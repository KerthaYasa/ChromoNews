"""
how_extractor.py — v2 (3-Layer Architecture)
=============================================
Arsitektur baru sesuai plan-how-extractor.md:

Layer 1 — Pattern gramatikal: deteksi pola preposisi instrumental
          (melalui, dengan, lewat, via, secara) diikuti nomina/verba.
          Menggunakan regex gramatikal Bahasa Indonesia — lebih robust
          daripada spaCy dep-parse karena belum ada model resmi id
          untuk spaCy 3.8+. Tetap EXPLAINABLE untuk keperluan sidang.

Layer 2 — Semantic similarity (sentence-transformers): ranking kandidat
          Layer 1 terhadap kalimat prototipe HOW. Dipakai sebagai
          TIE-BREAKER, bukan penentu utama.

Layer 3 — Fallback posisi: ambil kalimat ke-2/3 artikel jika Layer 1
          kosong. Hindari kalimat pertama (cenderung = WHAT).

Anti-duplikasi: cosine similarity kandidat HOW vs WHAT; buang jika > 0.85.

Dependencies:
- sentence-transformers  (sudah ada di project via semantic_search.py)
- scikit-learn           (sudah ada, untuk cosine_similarity)
"""

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Lazy-loaded model
# ---------------------------------------------------------------------------
_EMBED_MODEL = None  # SentenceTransformer model

# ---------------------------------------------------------------------------
# HOW prototype sentences (topic-neutral, for Layer 2 ranking)
# Kalimat generik yang merepresentasikan konsep "cara/metode/proses"
# tanpa terikat domain/topik tertentu.
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
# Pola gramatikal instrumental (regex)
# ---------------------------------------------------------------------------
# Pattern 1: "dengan + me-VERB" — pola paling khas HOW
#   Contoh: "dengan menggunakan", "dengan memanfaatkan", "dengan menerapkan"
_PAT_DENGAN_VERB = re.compile(
    r"\bdengan\s+(?:meng|mem|men|meny|me|di|ber|ter|pe)\w+\b",
    re.IGNORECASE,
)

# Pattern 2: Preposisi instrumental + frasa nominal/verbal
#   Contoh: "melalui audit keuangan", "lewat mekanisme X", "secara bertahap"
_PAT_PREP_INSTRUMENTAL = re.compile(
    r"\b(?:melalui|lewat|via)\s+\w+(?:\s+\w+){0,3}",
    re.IGNORECASE,
)

# Pattern 3: "secara + ADJEKTIVA/NOMINA"
#   Contoh: "secara bertahap", "secara sistematis", "secara langsung"
_PAT_SECARA = re.compile(
    r"\bsecara\s+\w+",
    re.IGNORECASE,
)

# Pattern 4: "menggunakan/memanfaatkan + NOMINA"
#   Contoh: "menggunakan metode X", "memanfaatkan teknologi"
_PAT_MENGGUNAKAN = re.compile(
    r"\b(?:menggunakan|memanfaatkan|mempergunakan|memakai)\s+\w+(?:\s+\w+){0,3}",
    re.IGNORECASE,
)

# Pattern 5: "dengan cara + ..."
#   Contoh: "dengan cara memalsukan dokumen"
_PAT_DENGAN_CARA = re.compile(
    r"\bdengan\s+cara\s+\w+(?:\s+\w+){0,5}",
    re.IGNORECASE,
)

# Pattern 6: "modus operandi" / "modusnya"
_PAT_MODUS = re.compile(
    r"\b(?:modus\s+operandi|modusnya|modus\s+\w+)",
    re.IGNORECASE,
)

# Pattern 7: "berawal dari" / "bermula dari" / "diawali"
_PAT_AWAL = re.compile(
    r"\b(?:berawal|bermula|diawali)\s+(?:dari\s+)?\w+",
    re.IGNORECASE,
)

# Semua patterns dikumpulkan dengan bobot
INSTRUMENTAL_PATTERNS = [
    (_PAT_DENGAN_CARA,      3.0),  # paling eksplisit
    (_PAT_MODUS,            3.0),  # sangat spesifik HOW
    (_PAT_DENGAN_VERB,      2.5),  # kuat
    (_PAT_PREP_INSTRUMENTAL, 2.0),  # kuat
    (_PAT_MENGGUNAKAN,      2.0),  # kuat
    (_PAT_AWAL,             1.5),  # sedang
    (_PAT_SECARA,           1.5),  # sedang
]

# Minimum sentence length (chars) to be considered meaningful
MIN_SENT_LEN = 25
# Minimum word count for fallback sentences
MIN_WORD_COUNT = 5
# Cosine similarity threshold for WHAT dedup
WHAT_DEDUP_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOT_FOUND = "Tidak disebutkan dalam artikel"


# =============================================================================
# INJECT / LOAD FUNCTIONS
# =============================================================================
def inject_embed_model(model):
    """Inject a pre-loaded SentenceTransformer model (reuse dari semantic_search)."""
    global _EMBED_MODEL
    _EMBED_MODEL = model


def _get_embed_model():
    """Lazy-load embedding model jika belum di-inject."""
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
# TEXT UTILITIES
# =============================================================================
def _split_sentences(text: str) -> List[str]:
    """Pecah teks menjadi kalimat (reuse logic dari project)."""
    text = re.sub(r'(\b[A-Z]{1,4})\.(\s)', r'\1. \2', text)
    text = re.sub(r'([a-z0-9"\')\\]])(\\.)([A-Z])', r'\1.\n\3', text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences


def _is_scraping_artifact(text: str) -> bool:
    """Cek artefak scraping/boilerplate."""
    patterns = [
        r"^(baca juga|gambas|scroll|advertisement|lihat juga|simak juga)",
        r"(klik|follow|download|subscribe)",
        r"pilihan editor|trending",
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# =============================================================================
# LAYER 1 — Grammatical Pattern Matching (Instrumental Prepositions)
# =============================================================================
def _score_instrumental_patterns(sentence: str) -> float:
    """
    Hitung skor pola instrumental dalam satu kalimat.
    Return > 0 jika ada pola yang match, 0 jika tidak.
    
    Prinsip: deteksi pola gramatikal "preposisi instrumental + frasa",
    bukan hardcoded keyword per domain. Ini tetap explainable untuk
    keperluan akademis.
    """
    total_score = 0.0
    for pattern, weight in INSTRUMENTAL_PATTERNS:
        if pattern.search(sentence):
            total_score += weight
    return total_score


def _find_pattern_candidates(sentences: List[str]) -> List[dict]:
    """
    Layer 1: scan semua kalimat, return candidates yang match pola
    gramatikal instrumental.
    """
    candidates = []

    for idx, sent in enumerate(sentences):
        if len(sent) < MIN_SENT_LEN:
            continue
        if _is_scraping_artifact(sent):
            continue

        score = _score_instrumental_patterns(sent)
        if score > 0:
            candidates.append({
                "text": sent,
                "index": idx,
                "pattern_score": score,
                "source": "pattern",
            })

    # Sort by pattern_score descending, then by position (earlier = tiebreak)
    candidates.sort(key=lambda x: (-x["pattern_score"], x["index"]))

    return candidates


# =============================================================================
# LAYER 2 — Semantic Similarity Ranking (Tie-breaker)
# =============================================================================
def _rank_by_similarity_to_prototypes(
    candidates: List[dict],
    embed_model,
    prototypes: List[str] = HOW_PROTOTYPES,
) -> List[dict]:
    """
    Ranking kandidat berdasarkan cosine similarity ke kalimat prototipe HOW.
    Dipakai sebagai tie-breaker kalau Layer 1 menghasilkan > 1 kandidat
    dengan skor yang sama/mirip.
    """
    if not candidates or embed_model is None:
        return candidates

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    import numpy as np

    cand_texts = [c["text"] for c in candidates]

    # Encode candidates + prototypes
    cand_embeddings = embed_model.encode(cand_texts, convert_to_numpy=True)
    proto_embeddings = embed_model.encode(prototypes, convert_to_numpy=True)

    # Cosine similarity: each candidate vs all prototypes
    sim_matrix = cos_sim(cand_embeddings, proto_embeddings)

    # Score = max similarity ke salah satu prototype
    for i, cand in enumerate(candidates):
        cand["proto_score"] = float(np.max(sim_matrix[i]))

    # Combined score: pattern_score (utama) + proto_score (bonus kecil)
    # Proto score dinormalisasi agar tidak mendominasi pattern score
    for cand in candidates:
        cand["combined_score"] = cand["pattern_score"] + (cand["proto_score"] * 0.5)

    # Sort by combined score descending
    candidates.sort(key=lambda x: -x["combined_score"])

    return candidates


# =============================================================================
# ANTI-DUPLIKASI WHAT vs HOW
# =============================================================================
def _is_similar_to_what(
    candidate_text: str,
    what_sentence: str,
    embed_model,
    threshold: float = WHAT_DEDUP_THRESHOLD,
) -> bool:
    """
    Cek apakah kandidat HOW terlalu mirip dengan WHAT (duplikasi).
    Return True jika cosine similarity > threshold.
    """
    if not what_sentence or embed_model is None:
        # Fallback: simple text overlap check
        if what_sentence:
            cand_words = set(re.findall(r'\w{3,}', candidate_text.lower()))
            what_words = set(re.findall(r'\w{3,}', what_sentence.lower()))
            if what_words:
                overlap_ratio = len(cand_words & what_words) / len(what_words)
                return overlap_ratio > 0.8
        return False

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    embeddings = embed_model.encode(
        [candidate_text, what_sentence], convert_to_numpy=True
    )
    sim = cos_sim([embeddings[0]], [embeddings[1]])[0][0]

    return sim > threshold


def _filter_what_duplicates(
    candidates: List[dict],
    what_sentence: str,
    embed_model,
    threshold: float = WHAT_DEDUP_THRESHOLD,
) -> List[dict]:
    """Filter out kandidat yang terlalu mirip WHAT."""
    if not what_sentence:
        return candidates

    return [
        c for c in candidates
        if not _is_similar_to_what(c["text"], what_sentence, embed_model, threshold)
    ]


# =============================================================================
# LAYER 3 — Positional Fallback
# =============================================================================
def _fallback_positional(
    sentences: List[str],
    what_sentence: str,
    embed_model,
) -> Optional[str]:
    """
    Fallback: ambil kalimat ke-2 s/d ke-4 (skip kalimat pertama = WHAT).
    Filter: minimal 5 kata, bukan scraping artifact, bukan duplikat WHAT.
    """
    for sent in sentences[1:4]:
        if len(sent.split()) < MIN_WORD_COUNT:
            continue
        if _is_scraping_artifact(sent):
            continue
        if _is_similar_to_what(sent, what_sentence, embed_model, WHAT_DEDUP_THRESHOLD):
            continue
        return sent

    # Terakhir: kalimat APAPUN yang valid (skip pertama)
    for sent in sentences[1:]:
        if len(sent.split()) >= MIN_WORD_COUNT and not _is_scraping_artifact(sent):
            return sent

    return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def extract_how(content: str, title: str = "", what_sentence: str = "") -> str:
    """
    Ekstraksi HOW dengan arsitektur 3-layer:
    1. Pattern gramatikal (preposisi instrumental) → kandidat utama
    2. Semantic similarity → tie-breaker untuk ranking
    3. Positional fallback → jika Layer 1 kosong

    Args:
        content:        Teks artikel (sudah di-strip dateline)
        title:          Judul artikel (konteks)
        what_sentence:  Kalimat WHAT (untuk anti-duplikasi)

    Returns:
        str — kalimat HOW terbaik, atau NOT_FOUND
    """
    if not content:
        return NOT_FOUND

    sentences = _split_sentences(content)
    if not sentences:
        return NOT_FOUND

    embed_model = _get_embed_model()

    # ─── Layer 1: Grammatical Pattern Matching ─────────────────────────
    candidates = _find_pattern_candidates(sentences)

    # Anti-duplikasi dengan WHAT
    candidates = _filter_what_duplicates(
        candidates, what_sentence, embed_model
    )

    if candidates:
        # ─── Layer 2: Semantic Similarity Ranking (tie-breaker) ────────
        if len(candidates) > 1 and embed_model is not None:
            candidates = _rank_by_similarity_to_prototypes(
                candidates, embed_model
            )

        return candidates[0]["text"]

    # ─── Layer 3: Positional Fallback ──────────────────────────────────
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
                "Rafael diduga menerima gratifikasi selama periode 2011-2023. "
                "Penggeledahan dilakukan di kantor KPK, Jakarta."
            ),
            "what": (
                "Komisi Pemberantasan Korupsi (KPK) menegaskan penetapan "
                "tersangka mantan pejabat pajak Rafael Alun Trisambodo memiliki "
                "landasan hukum."
            ),
        },
        {
            "title": "Tips Mudik Aman Lebaran 2023",
            "content": (
                "Jutaan pemudik diperkirakan akan memadati jalur mudik Lebaran 2023. "
                "Untuk menghindari kemacetan, pemudik disarankan berangkat "
                "lebih awal dengan menggunakan jalur tol Trans Jawa. "
                "Pastikan kendaraan dalam kondisi prima sebelum berangkat. "
                "Beristirahat setiap 4 jam perjalanan untuk menghindari kelelahan."
            ),
            "what": (
                "Jutaan pemudik diperkirakan akan memadati jalur mudik Lebaran 2023."
            ),
        },
        {
            "title": "Dampak Ekonomi Silicon Valley Bank",
            "content": (
                "Silicon Valley Bank (SVB) kolaps secara mendadak pada Maret 2023. "
                "Kebangkrutan terjadi setelah nasabah menarik dana secara masif "
                "melalui bank run yang berlangsung dalam hitungan jam. "
                "Regulator AS kemudian mengambil alih bank tersebut. "
                "Kejadian ini berdampak pada sektor teknologi global."
            ),
            "what": (
                "Silicon Valley Bank (SVB) kolaps secara mendadak pada Maret 2023."
            ),
        },
    ]

    print("=" * 70)
    print("TEST HOW EXTRACTOR v2 (3-Layer Architecture)")
    print("=" * 70)

    for art in test_articles:
        print(f"\nJUDUL : {art['title']}")
        print(f"WHAT  : {art['what']}")
        result = extract_how(
            art["content"],
            title=art["title"],
            what_sentence=art.get("what", ""),
        )
        print(f"HOW   : {result}")
        print("-" * 70)