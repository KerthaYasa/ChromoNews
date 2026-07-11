"""
how_extractor.py — v5 (POS Tagging + BERTScore sebagai Layer utama)
====================================================================
Arsitektur berlapis:

Layer 0 — POS Tagging (extraction.pos_tagger) untuk mengidentifikasi
          kalimat yang mengandung verba, lalu di-ranking dengan BERTScore
          (extraction.text_similarity) terhadap HOW_PROTOTYPES. Ambil
          top-3 kandidat tertinggi -- ini LAYER UTAMA sesuai revisi
          terbaru (catatan dosen).

Layer 1 — Pattern gramatikal (bersumber dari extraction.patterns:
          METHOD_CONNECTORS, METHOD_VERB_PATTERN, SECARA_METHOD_PATTERN).
          Jaring pengaman jika Layer 0 tidak menemukan kandidat.

Layer 2 — Semantic similarity (sentence-transformers) sebagai TIE-BREAKER
          untuk Layer 1.

Layer 3 — Fallback posisi: ambil kalimat ke-2/3 artikel jika Layer 0 & 1
          kosong, dengan filter kalimat status/fakta (mis. "X telah
          ditetapkan sebagai Y") yang bukan penjelasan cara/proses.
          Jika SEMUA kandidat fallback adalah status/fakta murni, extractor
          mengembalikan NOT_FOUND -- lebih jujur daripada memaksakan
          kalimat yang bukan HOW.

Anti-duplikasi: BERTScore (seluruh kalimat) kandidat HOW vs WHAT via
extraction.text_similarity; buang jika skor > WHAT_DEDUP_THRESHOLD.

Changelog v5.1 (bugfix):
  1. FIX false-positive Layer 0: syarat "ada verba" TERLALU LONGGAR (hampir
     semua kalimat berita punya verba, termasuk kalimat non-proses seperti
     "KPK belum bersedia mengumumkan..."). Kandidat sekarang WAJIB juga
     lolos _has_process_indicator() (verba proses konkret: menggeledah,
     menyita, memeriksa, menyelidiki, dst.) SEBELUM di-ranking BERTScore.
     POS tagging + BERTScore kini berfungsi sebagai RE-RANKER atas kandidat
     yang sudah masuk akal, bukan penyaring dari nol semua kalimat berverba.

Changelog v5:
  1. TAMBAH Layer 0: _find_verb_bertscore_candidates() -- POS tagging
     (extraction.pos_tagger.sentence_has_verb) + BERTScore terhadap
     HOW_PROTOTYPES, ambil top-3 kandidat berverba dengan skor tertinggi.
     Ini LAYER UTAMA baru; Layer 1 (regex pattern) & Layer 3 (posisi)
     jadi jaring pengaman jika Layer 0 kosong.
  2. TAMBAH extract_how_candidates() -- API publik untuk mengambil
     top-3 kandidat (bukan cuma jawaban tunggal), untuk keperluan
     UI/evaluasi.
  3. Anti-duplikasi WHAT vs HOW sekarang pakai BERTScore penuh
     (extraction.text_similarity), bukan cosine similarity 1 pasangan
     kalimat saja.

Changelog v4 (dipertahankan):
  1. _is_status_fact_sentence(): filter kalimat verba penetapan status
     (ditetapkan sebagai/menjadi, dst.) TANPA penanda proses/cara lain.
  2. _fallback_positional() skip kalimat status/fakta murni; kalau window
     habis tanpa kandidat valid, extractor mengembalikan NOT_FOUND.
  3. HOW_FALLBACK_CONNECTORS ("usai", "setelah") diprioritaskan lebih dulu.

Dependencies:
- transformers (opsional, POS tagging -- fallback heuristik morfologi
  jika model/koneksi tidak tersedia)
- bert-score (opsional, BERTScore -- fallback cosine similarity /
  word-overlap jika tidak tersedia)
- sentence-transformers  (opsional, reuse dari semantic_search.py)
- scikit-learn           (opsional, untuk cosine_similarity fallback)
"""

import re
from typing import List, Optional

from extraction.text_utils import split_sentences, is_scraping_artifact
from extraction.patterns import (
    METHOD_CONNECTORS,
    METHOD_VERB_PATTERN,
    HOW_FALLBACK_CONNECTORS,
    is_valid_how_secara,
    is_valid_how_dengan,
)
from extraction.text_similarity import bertscore_similarity
from extraction.pos_tagger import sentence_has_verb

# ---------------------------------------------------------------------------
# Lazy-loaded model
# ---------------------------------------------------------------------------
_EMBED_MODEL = None

# ---------------------------------------------------------------------------
# HOW prototype sentences (topic-neutral, for Layer 2 ranking)
# ---------------------------------------------------------------------------
HOW_PROTOTYPES = [
    "Peristiwa ini terjadi melalui serangkaian proses yang sistematis.",
    "Hal tersebut dilakukan dengan cara melakukan pendekatan bertahap.",
    "Pelaku menggunakan metode tertentu untuk mencapai tujuannya.",
    "Proses ini dilaksanakan secara bertahap dan terorganisir.",
    "Caranya adalah dengan memanfaatkan sarana yang tersedia.",
    "Tindakan dilakukan lewat mekanisme yang telah ditetapkan.",
    "Langkah-langkah yang diambil meliputi berbagai tahapan.",
    "Modus operandi yang digunakan cukup terstruktur.",
]

# ---------------------------------------------------------------------------
# Bobot per jenis pola instrumental
# ---------------------------------------------------------------------------
_HIGH_WEIGHT_CONNECTORS = {"dengan cara", "dengan modus", "modus operandi", "modusnya"}
_WEIGHT_HIGH = 3.0
_WEIGHT_METHOD_CONNECTOR = 2.0
_WEIGHT_DENGAN_VERB = 2.5
_WEIGHT_SECARA = 1.5
_WEIGHT_AWAL = 1.5

MIN_SENT_LEN = 25
MIN_WORD_COUNT = 5
WHAT_DEDUP_THRESHOLD = 0.85

NOT_FOUND = "Tidak disebutkan dalam artikel"

# =============================================================================
# FILTER STATUS/FAKTA — kalimat "X ditetapkan/dinyatakan sebagai Y"
# =============================================================================
# Pola verba penetapan status. Kalimat yang HANYA berisi pola ini (tanpa
# ada penanda proses/cara lain di kalimat yang sama) dianggap fakta latar,
# BUKAN penjelasan HOW, dan harus di-skip di Layer 3 fallback.
# Contoh yang di-skip: "Indonesia telah ditetapkan sebagai tuan rumah
# Piala Dunia U-20 2023."
# Contoh yang TETAP boleh lolos (karena ada penanda proses juga):
# "Ia ditetapkan sebagai tersangka setelah penyidik memeriksa 12 saksi."
#   -> tetap match _PAT_AWAL / HOW_FALLBACK_CONNECTORS lain, jadi aman.
_STATUS_FACT_PATTERN = re.compile(
    r"\b(?:telah\s+)?(?:ditetapkan|dinyatakan|ditunjuk|dinobatkan|ditasbihkan)\s+"
    r"(?:sebagai|menjadi)\s+\w+",
    re.IGNORECASE,
)


def _is_status_fact_sentence(sentence: str) -> bool:
    """
    True jika kalimat murni pernyataan status ("X ditetapkan sebagai Y")
    tanpa disertai penanda proses/cara lain (METHOD_CONNECTORS atau
    HOW_FALLBACK_CONNECTORS) di kalimat yang sama.
    """
    if not _STATUS_FACT_PATTERN.search(sentence):
        return False

    sent_lower = sentence.lower()
    # Jika ada penanda proses lain di kalimat yang sama, jangan anggap
    # sebagai status murni — biarkan tetap jadi kandidat.
    has_other_process_marker = any(conn in sent_lower for conn in METHOD_CONNECTORS)
    if has_other_process_marker:
        return False

    return True


# =============================================================================
# INJECT / LOAD FUNCTIONS
# =============================================================================
def inject_embed_model(model):
    global _EMBED_MODEL
    _EMBED_MODEL = model


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"[how_extractor] SentenceTransformer load gagal: {e}")
            return None
    return _EMBED_MODEL


# =============================================================================
# LAYER 0 — POS Tagging (verba) + BERTScore relevansi -> top-3 kandidat
# =============================================================================
# Sesuai catatan dosen:
#   1. Lakukan POS Tagging untuk mengidentifikasi kata kerja (verb).
#   2. Gunakan BERTScore untuk mencari kalimat yang paling relevan
#      (relevan terhadap konsep "cara/proses" -- diwakili HOW_PROTOTYPES).
#   3. Pilih 3 kalimat dengan skor tertinggi yang mengandung kata kerja
#      sebagai kandidat jawaban HOW.
HOW_TOP_N = 3


def _find_verb_bertscore_candidates(
    sentences: List[str],
    embed_model,
    top_n: int = HOW_TOP_N,
) -> List[dict]:
    """
    Layer 0: filter kalimat yang mengandung verba (POS tagging), lalu
    ranking berdasarkan BERTScore terhadap HOW_PROTOTYPES (kalimat
    prototipe "cara/proses"), ambil top-N sebagai kandidat jawaban HOW.
    """
    scored = []
    for idx, sent in enumerate(sentences):
        if len(sent) < MIN_SENT_LEN or len(sent.split()) < MIN_WORD_COUNT:
            continue
        if is_scraping_artifact(sent):
            continue
        if not sentence_has_verb(sent):
            continue

        # PENTING: "ada verba" saja terlalu longgar -- hampir semua kalimat
        # berita punya kata kerja (mis. "KPK belum bersedia mengumumkan...").
        # Kalimat WAJIB juga punya indikator proses/cara yang sudah dikenal
        # (pola instrumental Layer 1: "dengan cara", "melalui", "secara X",
        # dst. -- lihat _has_process_indicator). Verba + BERTScore hanya
        # dipakai untuk MERANKING kandidat yang SUDAH masuk akal ini, bukan
        # untuk menyaring dari nol semua kalimat berverba.
        if not _has_process_indicator(sent):
            continue

        # BERTScore relevansi: skor tertinggi kalimat ini vs SELURUH
        # kalimat prototipe HOW (bukan cuma dibandingkan ke 1 kalimat).
        proto_scores = [
            bertscore_similarity(sent, proto, embed_model=embed_model)
            for proto in HOW_PROTOTYPES
        ]
        best_score = max(proto_scores) if proto_scores else 0.0

        scored.append({
            "text": sent,
            "index": idx,
            "bertscore": best_score,
            "pattern_score": best_score,  # dipakai kompatibel dgn sort di layer lain
            "source": "pos_bertscore",
        })

    scored.sort(key=lambda c: (-c["bertscore"], c["index"]))
    return scored[:top_n]


def extract_how_candidates(
    content: str,
    what_sentence: str = "",
    top_n: int = HOW_TOP_N,
) -> List[dict]:
    """
    API publik: kembalikan top-N (default 3) kandidat jawaban HOW hasil
    POS Tagging (verba) + BERTScore, sudah difilter status/fakta &
    duplikasi vs WHAT. Berguna untuk ditampilkan di UI/evaluasi, bukan
    cuma jawaban tunggal seperti extract_how().

    Returns:
        List[dict] -- masing-masing {"text": str, "bertscore": float}
    """
    if not content:
        return []
    sentences = split_sentences(content)
    if not sentences:
        return []

    embed_model = _get_embed_model()
    candidates = _find_verb_bertscore_candidates(sentences, embed_model, top_n=top_n * 2)
    candidates = [c for c in candidates if not _is_status_fact_sentence(c["text"])]
    candidates = _filter_what_duplicates(candidates, what_sentence, embed_model)

    return [{"text": c["text"], "bertscore": c["bertscore"]} for c in candidates[:top_n]]



def _score_instrumental_patterns(sentence: str) -> float:
    total_score = 0.0
    sent_lower = sentence.lower()

    for conn in METHOD_CONNECTORS:
        if conn in sent_lower:
            if conn in _HIGH_WEIGHT_CONNECTORS:
                total_score += _WEIGHT_HIGH
            elif conn in ("berawal dari", "bermula dari", "diawali dari", "diawali"):
                total_score += _WEIGHT_AWAL
            else:
                total_score += _WEIGHT_METHOD_CONNECTOR

    if METHOD_VERB_PATTERN.search(sentence) and is_valid_how_dengan(sentence):
        total_score += _WEIGHT_DENGAN_VERB

    if is_valid_how_secara(sentence):
        total_score += _WEIGHT_SECARA

    return total_score


def _find_pattern_candidates(sentences: List[str]) -> List[dict]:
    candidates = []
    for idx, sent in enumerate(sentences):
        if len(sent) < MIN_SENT_LEN:
            continue
        if is_scraping_artifact(sent):
            continue

        score = _score_instrumental_patterns(sent)
        if score > 0:
            candidates.append({
                "text": sent,
                "index": idx,
                "pattern_score": score,
                "source": "pattern",
            })

    candidates.sort(key=lambda x: (-x["pattern_score"], x["index"]))
    return candidates


# =============================================================================
# LAYER 2 — Semantic Similarity Ranking (Tie-breaker)
# =============================================================================
def _rank_by_similarity_to_prototypes(candidates, embed_model, prototypes=HOW_PROTOTYPES):
    if not candidates or embed_model is None:
        return candidates

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    import numpy as np

    cand_texts = [c["text"] for c in candidates]
    cand_embeddings = embed_model.encode(cand_texts, convert_to_numpy=True)
    proto_embeddings = embed_model.encode(prototypes, convert_to_numpy=True)
    sim_matrix = cos_sim(cand_embeddings, proto_embeddings)

    for i, cand in enumerate(candidates):
        cand["proto_score"] = float(np.max(sim_matrix[i]))
    for cand in candidates:
        cand["combined_score"] = cand["pattern_score"] + (cand["proto_score"] * 0.5)

    candidates.sort(key=lambda x: -x["combined_score"])
    return candidates


# =============================================================================
# ANTI-DUPLIKASI WHAT vs HOW
# =============================================================================
def _is_similar_to_what(candidate_text, what_sentence, embed_model, threshold=WHAT_DEDUP_THRESHOLD):
    """
    Cek kemiripan kandidat HOW vs kalimat WHAT menggunakan BERTScore
    (extraction.text_similarity.bertscore_similarity), yang membandingkan
    SELURUH kalimat pada kedua teks -- bukan cuma 1 pasangan kalimat
    dibandingkan secara terbatas. Fallback otomatis ke cosine similarity
    SentenceTransformer, lalu word-overlap, jika BERTScore tidak tersedia.
    """
    if not what_sentence:
        return False
    sim = bertscore_similarity(candidate_text, what_sentence, embed_model=embed_model)
    return sim > threshold


def _filter_what_duplicates(candidates, what_sentence, embed_model, threshold=WHAT_DEDUP_THRESHOLD):
    if not what_sentence:
        return candidates
    return [
        c for c in candidates
        if not _is_similar_to_what(c["text"], what_sentence, embed_model, threshold)
    ]


# =============================================================================
# LAYER 3 — Positional Fallback (DIPERKETAT v4)
# =============================================================================
_PROCESS_VERBS = re.compile(
    r"\b(melakukan|menjalankan|menggelar|memeriksa|menggeledah|menyita|"
    r"memproses|mengumpulkan|menyelidiki|mengusut|menindaklanjuti|"
    r"menelusuri|memverifikasi|mengecek|menginvestigasi)\b",
    re.IGNORECASE,
)


def _has_process_indicator(sentence: str) -> bool:
    """
    Cek apakah kalimat punya indikasi proses/tindakan konkret, walau tidak
    match pola instrumental Layer 1. Dipakai sebagai syarat WAJIB untuk
    Layer 3 prioritas 2 (v4) — bukan lagi cukup asal posisi ke-2/3.

    CATATAN: HOW_FALLBACK_CONNECTORS ("usai", "setelah") SENGAJA TIDAK
    dicek di sini dengan substring bebas — itu sudah ditangani khusus di
    prioritas 1 (harus di AWAL kalimat). Jika dicek longgar ("mengandung
    kata setelah di mana saja"), kalimat non-HOW seperti "...diragukan
    setelah FIFA membatalkan..." ikut lolos padahal itu klausa waktu/WHY,
    bukan penjelasan cara. Di sini hanya verba tindakan konkret yang
    dihitung sebagai indikasi proses.
    """
    return bool(_PROCESS_VERBS.search(sentence))


def _fallback_positional(sentences: List[str], what_sentence: str, embed_model) -> Optional[str]:
    """
    Fallback: ambil kalimat ke-2 s/d ke-4 (skip kalimat pertama = WHAT).
    v4: kalimat status/fakta murni ("X ditetapkan sebagai Y") DI-SKIP.
    Kandidat fallback SEKARANG WAJIB punya indikasi proses
    (_has_process_indicator) — bukan lagi cukup asal posisi & panjang
    kata. Kalau tidak ada kandidat yang memenuhi, return None sehingga
    caller mengembalikan NOT_FOUND. Ini sengaja: untuk artikel yang
    memang tidak menjelaskan cara/metode (mis. artikel pernyataan/opini
    politik), lebih jujur mengosongkan HOW daripada memaksakan kalimat
    terdekat yang sebenarnya bukan penjelasan cara.
    """
    window = sentences[1:4]

    # Prioritas 1: kalimat yang diawali HOW_FALLBACK_CONNECTORS ("usai", "setelah")
    for sent in window:
        sent_lower = sent.lower().strip()
        if any(sent_lower.startswith(conn) for conn in HOW_FALLBACK_CONNECTORS):
            if len(sent.split()) < MIN_WORD_COUNT or is_scraping_artifact(sent):
                continue
            if _is_status_fact_sentence(sent):
                continue
            if _is_similar_to_what(sent, what_sentence, embed_model, WHAT_DEDUP_THRESHOLD):
                continue
            return sent

    # Prioritas 2: kalimat ke-2 s/d ke-4 dengan indikasi proses (WAJIB),
    # bukan status/fakta murni.
    for sent in window:
        if len(sent.split()) < MIN_WORD_COUNT:
            continue
        if is_scraping_artifact(sent):
            continue
        if _is_status_fact_sentence(sent):
            continue
        if not _has_process_indicator(sent):
            continue
        if _is_similar_to_what(sent, what_sentence, embed_model, WHAT_DEDUP_THRESHOLD):
            continue
        return sent

    # Tidak ada kandidat proses yang valid -> jangan paksakan.
    # Caller akan mengembalikan NOT_FOUND ("Tidak disebutkan dalam artikel").
    return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def extract_how(content: str, title: str = "", what_sentence: str = "") -> str:
    """
    Ekstraksi HOW dengan arsitektur berlapis.
    v5: Layer 0 baru -- POS Tagging (verba) + BERTScore terhadap
    HOW_PROTOTYPES, ambil top-3 kandidat, pilih yang berskor tertinggi.
    Layer 1 (pattern gramatikal) dan Layer 3 (fallback posisi) tetap
    dipertahankan sebagai jaring pengaman jika Layer 0 tidak menemukan
    kalimat berverba yang relevan sama sekali.
    """
    if not content:
        return NOT_FOUND

    sentences = split_sentences(content)
    if not sentences:
        return NOT_FOUND

    embed_model = _get_embed_model()

    # --- Layer 0: POS Tagging (verba) + BERTScore -> top-3 ---
    verb_candidates = _find_verb_bertscore_candidates(sentences, embed_model, top_n=HOW_TOP_N)
    verb_candidates = [c for c in verb_candidates if not _is_status_fact_sentence(c["text"])]
    verb_candidates = _filter_what_duplicates(verb_candidates, what_sentence, embed_model)
    if verb_candidates:
        return verb_candidates[0]["text"]

    # --- Layer 1+2: Pattern gramatikal + semantic similarity tie-breaker ---
    candidates = _find_pattern_candidates(sentences)
    candidates = _filter_what_duplicates(candidates, what_sentence, embed_model)

    if candidates:
        if len(candidates) > 1 and embed_model is not None:
            candidates = _rank_by_similarity_to_prototypes(candidates, embed_model)
        return candidates[0]["text"]

    # --- Layer 3: fallback posisi ---
    fallback = _fallback_positional(sentences, what_sentence, embed_model)
    if fallback:
        return fallback

    return NOT_FOUND


# =============================================================================
# TESTING MANDIRI
# =============================================================================
if __name__ == "__main__":
    test_articles = [
        {
            "title": "Rafael Alun Ngaku Ditarget Jadi Tersangka",
            "content": (
                "Komisi Pemberantasan Korupsi (KPK) menegaskan penetapan "
                "tersangka mantan pejabat pajak Rafael Alun Trisambodo memiliki "
                "landasan hukum. KPK melakukan penyelidikan melalui audit "
                "keuangan dan pemeriksaan rekening milik Rafael. "
                "Rafael diduga menerima gratifikasi selama periode 2011-2023."
            ),
            "what": (
                "Komisi Pemberantasan Korupsi (KPK) menegaskan penetapan "
                "tersangka mantan pejabat pajak Rafael Alun Trisambodo memiliki "
                "landasan hukum."
            ),
        },
        {
            "title": "Plt Menpora Sebut Piala Dunia U-20 Terancam Batal",
            "content": (
                "Pelaksana Tugas (PLT) Menteri Pemuda dan Olahraga Muhadjir Effendy "
                "mengatakan masyarakat tak perlu khawatir jika Piala Dunia U-20 batal "
                "digelar di Indonesia seolah bakal terjadi kiamat. "
                "Indonesia telah ditetapkan sebagai tuan rumah Piala Dunia U-20 2023. "
                "Namun, hal ini belakangan diragukan setelah FIFA membatalkan drawing "
                "Piala Dunia U-20 di Bali. "
                "Pembatalan ini berkaitan dengan sikap Gubernur Bali Wayan Koster yang "
                "menolak keikutsertaan Timnas Israel."
            ),
            "what": (
                "Pelaksana Tugas (PLT) Menteri Pemuda dan Olahraga Muhadjir Effendy "
                "mengatakan masyarakat tak perlu khawatir jika Piala Dunia U-20 batal "
                "digelar di Indonesia seolah bakal terjadi kiamat."
            ),
        },
        {
            "title": "Penipuan Online Modus Baru",
            "content": (
                "Polisi mengungkap sindikat penipuan online dengan skema investasi bodong. "
                "Pelaku menjaring korban melalui iklan di media sosial. "
                "Korban diminta mentransfer dana via rekening penampung."
            ),
            "what": (
                "Polisi mengungkap sindikat penipuan online dengan skema investasi bodong."
            ),
        },
    ]

    print("=" * 70)
    print("TEST HOW EXTRACTOR v4 (filter status/fakta di Layer 3)")
    print("=" * 70)

    for art in test_articles:
        print(f"\nJUDUL : {art['title']}")
        print(f"WHAT  : {art['what']}")
        result = extract_how(art["content"], title=art["title"], what_sentence=art.get("what", ""))
        print(f"HOW   : {result}")
        print("-" * 70)