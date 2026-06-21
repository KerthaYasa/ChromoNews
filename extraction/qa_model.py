"""
qa_model.py
===========
Wrapper extractive QA model (Rifky/Indobert-QA) sebagai FALLBACK untuk
WHY & HOW ketika rule-based pattern matching gagal.

REVISI (v4) -- PENTING, baca sebelum mengubah apa pun di file ini:

transformers v5.0+ MENGHAPUS TOTAL pipeline task "question-answering"
dari registry (bukan rename, dihapus -- lihat MIGRATION_GUIDE_V5.md
resmi HuggingFace). Awalnya project ini coba menahan transformers di
v4.x lewat pin requirements.txt, TAPI itu tidak bisa dipertahankan:
sentence-transformers (dependency lain di project ini, untuk
semantic_search.py) sejak versi 5.2.1 SUDAH MELEPAS upper bound
"transformers<5.0.0" dari metadata dependency-nya sendiri, supaya bisa
mendukung transformers v5. Begitu sentence-transformers di-upgrade
(dan pip akan terus menariknya karena tidak ada upper bound di
requirements.txt kita untuk sentence-transformers juga), pip resolver
bebas menarik transformers v5 lagi -- pin kita di sisi transformers
TIDAK PERNAH cukup, karena bukan transformers yang jadi sumber
constraint, melainkan sentence-transformers yang sudah tidak melarang.

KEPUTUSAN FINAL: berhenti melawan arah ekosistem ini. File ini SEKARANG
TIDAK PAKAI `pipeline()` SAMA SEKALI -- load model+tokenizer manual via
AutoModelForQuestionAnswering, jalankan forward pass sendiri, decode
span jawaban dari start/end logits sendiri. Ini valid di SEMUA versi
transformers (v4 maupun v5) karena tidak menyentuh task registry yang
jadi sumber breaking change.

Kenapa fallback, bukan pengganti total rule-based? (tetap berlaku)
- Rule-based jalan DULU untuk SEMUA artikel (cepat, explainable).
- Model HANYA dipanggil untuk artikel yang gagal di rule-based (hemat
  komputasi -- tidak semua artikel butuh inference model).
- Model ini EKSTRAKTIF (ambil span teks asli), BUKAN generatif -- jadi
  tidak akan "mengarang" fakta baru di luar artikel.
"""

import torch

QUESTIONS = {
    "why": "Mengapa peristiwa ini terjadi?",
    "how": "Bagaimana cara atau proses peristiwa ini terjadi?",
}

CONFIDENCE_THRESHOLD = 0.15
MIN_ANSWER_LENGTH = 5
MAX_CONTEXT_TOKENS = 384  # batas aman untuk model BERT-base (max_position 512,
                            # sisakan ruang utk pertanyaan + token spesial)


def load_qa_pipeline():
    """
    Loader MENTAH -- TIDAK membungkus exception (sengaja), supaya
    Streamlit @st.cache_resource yang atur caching & retry behavior.

    TIDAK lagi memanggil transformers.pipeline() (lihat penjelasan di
    docstring modul). Return tuple (model, tokenizer) mentah; semua
    logic forward-pass + span decoding ada di answer_question() di
    bawah, bukan di sini.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]

    Raises:
        Exception apapun saat load -- caller (app.py) WAJIB try/except.
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    model_name = "Rifky/Indobert-QA"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()  # mode inference, bukan training -- matikan dropout dll
    return (model, tokenizer)


def answer_question(pipe, field, context):
    """
    Args:
        pipe: tuple (model, tokenizer) hasil load_qa_pipeline() yang
              sudah di-cache (BUKAN None)
        field: "why" atau "how"
        context: isi artikel (str)

    Returns:
        str -- jawaban (span dari teks asli), atau None kalau tidak
        cukup yakin (skor < threshold) / pipe None.
    """
    if pipe is None:
        return None

    model, tokenizer = pipe

    question = QUESTIONS.get(field)
    if not question or not context:
        return None

    try:
        # encode question+context sekaligus, model BERT-QA butuh format
        # [CLS] question [SEP] context [SEP] -- tokenizer urus otomatis
        # lewat argumen text/text_pair. return_offsets_mapping supaya
        # kita bisa balik dari index token ke index karakter di context
        # asli (perlu untuk slice jawaban dari teks ASLI, bukan hasil
        # decode tokenizer yang bisa berubah whitespace/casing-nya).
        inputs = tokenizer(
            question,
            context[:2000],
            max_length=MAX_CONTEXT_TOKENS,
            truncation="only_second",  # kalau kepanjangan, potong context, JANGAN potong question
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0]  # (seq_len, 2)
        # sequence_ids: None utk token spesial, 0 utk question, 1 utk context
        # -- perlu ini supaya start/end span yg dipilih TIDAK jatuh di
        # bagian question (model kadang salah confidence ke situ)
        sequence_ids = inputs.sequence_ids(0)

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs.start_logits[0]  # (seq_len,)
        end_logits = outputs.end_logits[0]

        # Mask token yang BUKAN bagian context (question + token spesial)
        # dengan -inf, supaya tidak mungkin terpilih sebagai start/end.
        context_mask = torch.tensor(
            [sid == 1 for sid in sequence_ids], dtype=torch.bool
        )
        start_logits = start_logits.masked_fill(~context_mask, float("-inf"))
        end_logits = end_logits.masked_fill(~context_mask, float("-inf"))

        start_idx = int(torch.argmax(start_logits))
        end_idx = int(torch.argmax(end_logits))

        # Span tidak valid kalau end sebelum start -- batasi end ke
        # beberapa token setelah start sebagai fallback (jarang terjadi,
        # tapi harus ditangani supaya tidak crash/return teks acak)
        if end_idx < start_idx:
            end_idx = start_idx

        # Skor confidence: gabungan probabilitas start & end (softmax),
        # standar untuk extractive QA -- sebelumnya transformers.pipeline
        # menghitung ini otomatis, sekarang kita hitung manual.
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)
        score = float(start_probs[start_idx] * end_probs[end_idx])

        if score < CONFIDENCE_THRESHOLD:
            return None

        # Decode span jawaban dari TEKS ASLI (bukan tokenizer.decode(),
        # yang bisa mengubah whitespace/casing) via offset_mapping.
        char_start = int(offset_mapping[start_idx][0])
        char_end = int(offset_mapping[end_idx][1])
        if char_start == 0 and char_end == 0:
            # offset (0,0) artinya token spesial/padding -- tidak valid
            return None

        answer = context[:2000][char_start:char_end].strip()
    except Exception:
        return None

    if len(answer) < MIN_ANSWER_LENGTH:
        return None

    return answer
