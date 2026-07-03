"""
why_extractor.py
================
Ekstraksi WHY (Mengapa) dengan konektor kausal.
Sesuai masukan dosen: gunakan kata konektor "karena", "akibat", "disebabkan".

Changelog:
  v3 — Patch anti-duplikasi WHY vs WHAT dan WHY vs HOW:
       1. Exclude what_sentence dan how_sentence dari pool kalimat yang di-scan
          (exact match + normalized match).
       2. Similarity check (cosine atau word-overlap fallback) terhadap WHAT
          dan HOW sebelum kandidat di-finalisasi — sama dengan pola di how_extractor.
       3. "untuk" dihapus dari PURPOSE_CONNECTORS (terlalu generik, high FPR).
          "untuk" hanya dipakai sebagai last-resort fallback jika tidak ada
          kandidat lain sama sekali.
       4. Tambah parameter how_sentence untuk exclude hasil HOW.
       5. Shared embed model via inject_embed_model() — tidak load ulang.
"""

import re
from typing import List, Optional

# =============================================================================
# CAUSAL CONNECTORS (Sesuai masukan dosen)
# =============================================================================
CAUSAL_CONNECTORS = [
    "karena", "sebab", "lantaran", "akibat", "disebabkan", "diakibatkan",
    "dipicu", "buntut", "imbas", "dampak", "berujung",
    "atas dasar", "dengan tujuan", "bertujuan", "dalam rangka",
]

INTER_SENTENCE_CAUSAL = [
    "pasalnya", "oleh karena itu", "sebab itu", "karenanya", "alhasil",
    "hal itu disebabkan", "hal tersebut disebabkan",
]

# "untuk" DIHAPUS — terlalu generik, muncul di hampir setiap kalimat berita.
# "untuk" hanya dipakai sebagai last-resort fallback (lihat bagian bawah).
PURPOSE_CONNECTORS = [
    "dengan tujuan", "bertujuan", "dalam rangka", "demi", "guna",
    "agar", "supaya", "berharap", "diharapkan",
]

# Threshold word-overlap untuk anti-duplikasi (fallback tanpa embed model)
_WORD_OVERLAP_THRESHOLD = 0.7

# Similarity threshold cosine (sama dengan how_extractor)
_COSINE_THRESHOLD = 0.85

# Shared embed model instance (di-inject dari luar agar tidak load dua kali)
_EMBED_MODEL = None


def inject_embed_model(model):
    """Inject shared SentenceTransformer model dari how_extractor / app.py."""
    global _EMBED_MODEL
    _EMBED_MODEL = model


def _get_embed_model():
    """Lazy-load embed model jika belum di-inject."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"[why_extractor] SentenceTransformer load gagal: {e}")
            return None
    return _EMBED_MODEL


# =============================================================================
# UTILITIES
# =============================================================================

def _split_sentences(text: str) -> List[str]:
    """Pecah teks menjadi kalimat."""
    text = re.sub(r'(\b[A-Z]{1,4})\.([\s])', r'\1. \2', text)
    text = re.sub(r'([a-z0-9"\')\]])(\.)([A-Z])', r'\1.\n\3', text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences


def _normalize(text: str) -> str:
    """Normalisasi untuk perbandingan: lowercase, hapus tanda baca & spasi ganda."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _is_excluded(sentence: str, exclusions: List[str]) -> bool:
    """
    Cek apakah sentence cocok (exact atau normalized) dengan salah satu
    string di exclusions. Dipakai untuk buang what_sentence / how_sentence
    dari pool kandidat WHY.
    """
    norm_sent = _normalize(sentence)
    for excl in exclusions:
        if not excl:
            continue
        # Exact match
        if sentence.strip() == excl.strip():
            return True
        # Normalized match
        if norm_sent == _normalize(excl):
            return True
        # Substring match: jika exclusion adalah bagian dari sentence atau sebaliknya
        # (menangani kasus WHAT yang sudah di-truncate)
        norm_excl = _normalize(excl)
        if norm_excl and (norm_excl in norm_sent or norm_sent in norm_excl):
            if len(norm_excl) > 30:   # hindari false-positive untuk string pendek
                return True
    return False


def _is_similar_to(
    candidate: str,
    reference: str,
    embed_model,
    threshold: float = _COSINE_THRESHOLD,
) -> bool:
    """
    Cek kemiripan kandidat WHY dengan reference (WHAT atau HOW).
    Pakai cosine similarity jika embed_model tersedia, word-overlap jika tidak.
    """
    if not reference:
        return False

    if embed_model is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            embeddings = embed_model.encode(
                [candidate, reference], convert_to_numpy=True
            )
            sim = cos_sim([embeddings[0]], [embeddings[1]])[0][0]
            return float(sim) > threshold
        except Exception:
            pass

    # Fallback: word overlap
    cand_words = set(re.findall(r'\w{3,}', candidate.lower()))
    ref_words  = set(re.findall(r'\w{3,}', reference.lower()))
    if not ref_words:
        return False
    overlap_ratio = len(cand_words & ref_words) / len(ref_words)
    return overlap_ratio > _WORD_OVERLAP_THRESHOLD


def _filter_duplicates(
    candidates: List[tuple],
    what_sentence: str,
    how_sentence: str,
    embed_model,
) -> List[tuple]:
    """
    Buang kandidat WHY yang terlalu mirip dengan WHAT atau HOW.
    Dijalankan SETELAH exclusion list — ini lapisan kedua yang lebih lenient
    (threshold cosine 0.85 / word-overlap 0.7).
    """
    filtered = []
    for ctype, ctext in candidates:
        if _is_similar_to(ctext, what_sentence, embed_model):
            continue
        if _is_similar_to(ctext, how_sentence, embed_model):
            continue
        filtered.append((ctype, ctext))
    return filtered


def _extract_clause_after_connector(text: str, connectors: List[str]) -> Optional[str]:
    """Ekstrak klausa setelah konektor."""
    text_lower = text.lower()

    for conn in connectors:
        pos = text_lower.find(conn)
        if pos == -1:
            continue

        after = text[pos + len(conn):].strip()

        end_match = re.search(r'[.!?]', after)
        if end_match:
            clause = after[:end_match.end()]
        else:
            clause = after[:200]

        clause = clause.strip(" ,:-\"")
        if len(clause) > 15:
            return clause

    return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def extract_why(
    content: str,
    what_sentence: str = "",
    how_sentence: str = "",
) -> str:
    """
    WHY: Deteksi alasan menggunakan konektor kausal.
    Sesuai masukan dosen: gunakan "karena", "akibat", "disebabkan".

    Args:
        content:       Teks artikel bersih.
        what_sentence: Hasil WHAT — dikecualikan dari kandidat WHY.
        how_sentence:  Hasil HOW  — dikecualikan dari kandidat WHY.

    Anti-duplikasi (dua lapis):
        Lapis 1 — exact/normalized exclusion: buang kalimat yang identik
                  dengan what_sentence atau how_sentence sebelum scan dimulai.
        Lapis 2 — similarity check: buang kandidat yang cosine similarity-nya
                  > 0.85 terhadap WHAT atau HOW (atau word-overlap > 0.7
                  jika embed model tidak tersedia).
    """
    if not content:
        return "Tidak disebutkan dalam artikel"

    sentences = _split_sentences(content)
    exclusions = [s for s in [what_sentence, how_sentence] if s]

    # ── Lapis 1: buang kalimat yang identik/mirip dengan WHAT atau HOW ──
    filtered_sentences = [
        s for s in sentences
        if not _is_excluded(s, exclusions)
    ]

    # Gunakan filtered_sentences untuk scan kandidat
    candidates = []

    # 1. Konektor antar-kalimat (pasalnya, oleh karena itu)
    for sent in filtered_sentences:
        sent_lower = sent.lower().strip()
        for conn in INTER_SENTENCE_CAUSAL:
            if sent_lower.startswith(conn) and len(sent) > 20:
                candidates.append(("inter", sent.strip()))
                break

    # 2. Konektor kausal di dalam kalimat
    for sent in filtered_sentences:
        sent_lower = sent.lower()
        for conn in CAUSAL_CONNECTORS:
            if conn in sent_lower and len(sent) > 20:
                candidates.append(("causal", sent.strip()))
                break

    # 3. Konektor tujuan (lebih spesifik — "untuk" sudah dibuang)
    for sent in filtered_sentences:
        sent_lower = sent.lower()
        for conn in PURPOSE_CONNECTORS:
            if conn in sent_lower and len(sent) > 20:
                candidates.append(("purpose", sent.strip()))
                break

    # ── Lapis 2: filter similarity vs WHAT dan HOW ──────────────────────
    if candidates and (what_sentence or how_sentence):
        embed_model = _get_embed_model()
        candidates = _filter_duplicates(
            candidates, what_sentence, how_sentence, embed_model
        )

    if not candidates:
        # Last-resort fallback 1: klausa setelah "karena"/"sebab" dari
        # filtered_sentences (WHAT/HOW sudah dikecualikan)
        filtered_text = " ".join(filtered_sentences)
        clause = _extract_clause_after_connector(filtered_text, ["karena", "sebab"])
        if clause:
            # Similarity check terhadap WHAT/HOW untuk fallback juga
            embed_model = _get_embed_model()
            if not _is_similar_to(clause, what_sentence, embed_model) and \
               not _is_similar_to(clause, how_sentence, embed_model):
                return clause

        # Last-resort fallback 2: "untuk" sebagai connector (paling lemah)
        for sent in filtered_sentences:
            if re.search(r'\buntuk\b', sent.lower()) and len(sent) > 20:
                embed_model = _get_embed_model()
                if not _is_similar_to(sent, what_sentence, embed_model) and \
                   not _is_similar_to(sent, how_sentence, embed_model):
                    return sent.strip()

        return "Tidak disebutkan dalam artikel"

    # Jika hanya satu kandidat, return langsung
    if len(candidates) == 1:
        return candidates[0][1]

    # Multi-event tie-breaking: prioritaskan kandidat yang share kata kunci
    # dengan what_sentence (event utama)
    if what_sentence:
        what_words = set(re.findall(r'\w{4,}', what_sentence.lower()))
        best = None
        best_score = -1
        for ctype, ctext in candidates:
            cand_words = set(re.findall(r'\w{4,}', ctext.lower()))
            overlap = len(what_words & cand_words)
            weight = 1 if ctype == "inter" else 0
            score = overlap + weight
            if score > best_score:
                best_score = score
                best = ctext
        if best:
            return best

    # Default: prioritas inter > causal > purpose
    priority = {"inter": 0, "causal": 1, "purpose": 2}
    candidates.sort(key=lambda x: priority.get(x[0], 9))
    return candidates[0][1]
