"""
how_extractor.py — v4 (3-Layer Architecture, sinkron patterns.py)
====================================================================
Arsitektur tetap 3-layer sesuai plan-how-extractor.md:

Layer 1 — Pattern gramatikal (bersumber dari extraction.patterns:
          METHOD_CONNECTORS, METHOD_VERB_PATTERN, SECARA_METHOD_PATTERN).

Layer 2 — Semantic similarity (sentence-transformers) sebagai TIE-BREAKER.

Layer 3 — Fallback posisi: ambil kalimat ke-2/3 artikel jika Layer 1
          kosong, TAPI SEKARANG memfilter kalimat status/fakta (mis.
          "X telah ditetapkan sebagai Y", "X dinyatakan sebagai Y") yang
          bukan penjelasan cara/proses meski lolos syarat panjang/posisi.
          Jika SEMUA kandidat fallback adalah status/fakta murni, extractor
          mengembalikan NOT_FOUND — lebih jujur daripada memaksakan
          kalimat yang bukan HOW ("no answer" lebih baik dari jawaban
          salah, terutama untuk artikel yang memang tidak menjelaskan cara,
          seperti artikel pernyataan/opini politik).

Anti-duplikasi: cosine similarity kandidat HOW vs WHAT; buang jika > 0.85.

Changelog v4:
  1. TAMBAH _is_status_fact_sentence(): filter kalimat yang berisi
     verba penetapan status (ditetapkan sebagai/menjadi, dinyatakan
     sebagai, ditunjuk sebagai, dinobatkan sebagai, dll) TANPA disertai
     penanda proses/cara apapun. Kalimat semacam ini adalah FAKTA LATAR,
     bukan HOW, dan sebelumnya lolos ke Layer 3 fallback karena hanya
     dicek panjang kata & posisi.
  2. _fallback_positional() sekarang skip kalimat status/fakta murni.
     Jika window kalimat ke-2..4 habis tanpa kandidat valid, extractor
     mengembalikan NOT_FOUND alih-alih memaksakan kalimat ke-berapa pun
     yang tersisa (perilaku lama: "kalimat APAPUN yang valid" — DIHAPUS
     karena inilah sumber false positive "ditetapkan sebagai tuan rumah").
  3. HOW_FALLBACK_CONNECTORS ("usai", "setelah") tetap diprioritaskan
     lebih dulu sebelum fallback biasa.

Dependencies:
- sentence-transformers  (opsional, reuse dari semantic_search.py)
- scikit-learn           (opsional, untuk cosine_similarity)
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
# LAYER 1 — Grammatical Pattern Matching (sumber: patterns.py)
# =============================================================================
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
    if not what_sentence or embed_model is None:
        if what_sentence:
            cand_words = set(re.findall(r'\w{3,}', candidate_text.lower()))
            what_words = set(re.findall(r'\w{3,}', what_sentence.lower()))
            if what_words:
                overlap_ratio = len(cand_words & what_words) / len(what_words)
                return overlap_ratio > 0.8
        return False

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    embeddings = embed_model.encode([candidate_text, what_sentence], convert_to_numpy=True)
    sim = cos_sim([embeddings[0]], [embeddings[1]])[0][0]
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
    Ekstraksi HOW dengan arsitektur 3-layer.
    v4: fallback Layer 3 lebih jujur — mengembalikan NOT_FOUND untuk
    artikel yang memang tidak menjelaskan cara/metode/proses, daripada
    memaksakan kalimat status/fakta yang kebetulan berada di posisi awal.
    """
    if not content:
        return NOT_FOUND

    sentences = split_sentences(content)
    if not sentences:
        return NOT_FOUND

    embed_model = _get_embed_model()

    candidates = _find_pattern_candidates(sentences)
    candidates = _filter_what_duplicates(candidates, what_sentence, embed_model)

    if candidates:
        if len(candidates) > 1 and embed_model is not None:
            candidates = _rank_by_similarity_to_prototypes(candidates, embed_model)
        return candidates[0]["text"]

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