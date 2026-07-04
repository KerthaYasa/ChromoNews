"""
why_extractor.py
================
Ekstraksi WHY (Mengapa) dengan konektor kausal.
Sesuai masukan dosen: gunakan kata konektor "karena", "akibat", "disebabkan".

Changelog:
  v5 — Tambah marker relasional "berkaitan dengan" / "terkait dengan"
       dengan disambiguasi konteks (lihat is_valid_why_relational).
  v4 — Sinkronisasi penuh dengan patterns.py (REVISI v6).
  v3 — Patch anti-duplikasi WHY vs WHAT dan WHY vs HOW.
"""

import re
from typing import List, Optional

from extraction.patterns import (
    CAUSAL_CONNECTORS_TIER1,
    CAUSAL_CONNECTORS_TIER2,
    CAUSAL_CONNECTORS_AMBIGUOUS,
    INTER_SENTENCE_CAUSAL,
    PURPOSE_CONNECTORS,
    is_valid_why_sebab,
    is_valid_why_untuk,
)
from extraction.text_utils import split_sentences

# =============================================================================
# MARKER KAUSAL RELASIONAL — "berkaitan dengan" / "terkait dengan"
# =============================================================================
# Marker ini TIDAK ditambahkan ke patterns.py CAUSAL_CONNECTORS_TIER2 karena
# butuh disambiguasi khusus: "berkaitan dengan"/"terkait dengan" sangat umum
# di berita Indonesia untuk menyatakan SEBAB dari sebuah peristiwa
# ("Pembatalan ini berkaitan dengan sikap Gubernur..."), tapi juga sering
# dipakai sebagai penghubung topik generik non-kausal ("terkait dengan hal
# itu, DPR akan menggelar rapat"). Disambiguasi: valid sebagai WHY hanya
# jika didahului NOMINA PERISTIWA/KEPUTUSAN (mis. "pembatalan ini", "hal
# ini", "keputusan itu", "penolakan ini").
RELATIONAL_CAUSAL_CONNECTORS = ["berkaitan dengan", "terkait dengan"]

_RELATIONAL_SUBJECT_PATTERN = re.compile(
    r"\b(?:pembatalan|penolakan|penundaan|keputusan|kebijakan|langkah|"
    r"sikap|hal|kondisi|masalah|persoalan|dugaan|rencana|wacana)\s*"
    r"(?:ini|itu|tersebut)?\s*(?:yang\s+)?(?:berkaitan|terkait)\s+dengan\b",
    re.IGNORECASE,
)


def is_valid_why_relational(sentence: str) -> bool:
    """
    Cek apakah 'berkaitan dengan'/'terkait dengan' dalam kalimat menandakan
    WHY (sebab suatu peristiwa/keputusan), bukan sekadar penghubung topik
    generik. Valid jika didahului nomina peristiwa/keputusan.
    """
    return bool(_RELATIONAL_SUBJECT_PATTERN.search(sentence))


# =============================================================================
# THRESHOLDS
# =============================================================================
_WORD_OVERLAP_THRESHOLD = 0.7
_COSINE_THRESHOLD = 0.85

_TYPE_PRIORITY = {
    "inter": 0,
    "relational": 1,
    "tier1": 2,
    "tier2": 3,
    "ambiguous": 4,
    "purpose": 5,
    "untuk": 6,
}

_EMBED_MODEL = None


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
            print(f"[why_extractor] SentenceTransformer load gagal: {e}")
            return None
    return _EMBED_MODEL


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _is_excluded(sentence: str, exclusions: List[str]) -> bool:
    norm_sent = _normalize(sentence)
    for excl in exclusions:
        if not excl:
            continue
        if sentence.strip() == excl.strip():
            return True
        if norm_sent == _normalize(excl):
            return True
        norm_excl = _normalize(excl)
        if norm_excl and (norm_excl in norm_sent or norm_sent in norm_excl):
            if len(norm_excl) > 30:
                return True
    return False


def _is_similar_to(candidate, reference, embed_model, threshold=_COSINE_THRESHOLD):
    if not reference:
        return False
    if embed_model is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            embeddings = embed_model.encode([candidate, reference], convert_to_numpy=True)
            sim = cos_sim([embeddings[0]], [embeddings[1]])[0][0]
            return float(sim) > threshold
        except Exception:
            pass
    cand_words = set(re.findall(r'\w{3,}', candidate.lower()))
    ref_words = set(re.findall(r'\w{3,}', reference.lower()))
    if not ref_words:
        return False
    return len(cand_words & ref_words) / len(ref_words) > _WORD_OVERLAP_THRESHOLD


def _filter_duplicates(candidates, what_sentence, how_sentence, embed_model):
    filtered = []
    for ctype, ctext in candidates:
        if _is_similar_to(ctext, what_sentence, embed_model):
            continue
        if _is_similar_to(ctext, how_sentence, embed_model):
            continue
        filtered.append((ctype, ctext))
    return filtered


def _extract_clause_after_connector(text, connectors):
    text_lower = text.lower()
    for conn in connectors:
        pos = text_lower.find(conn)
        if pos == -1:
            continue
        after = text[pos + len(conn):].strip()
        end_match = re.search(r'[.!?]', after)
        clause = after[:end_match.end()] if end_match else after[:200]
        clause = clause.strip(" ,:-\"")
        if len(clause) > 15:
            return clause
    return None


def _find_causal_candidates(sentences: List[str]) -> List[tuple]:
    candidates = []

    for sent in sentences:
        sent_lower = sent.lower().strip()
        for conn in INTER_SENTENCE_CAUSAL:
            if sent_lower.startswith(conn.lower()) and len(sent) > 20:
                candidates.append(("inter", sent.strip()))
                break

    for sent in sentences:
        sent_lower = sent.lower()
        for conn in RELATIONAL_CAUSAL_CONNECTORS:
            if conn not in sent_lower or len(sent) <= 20:
                continue
            if is_valid_why_relational(sent):
                candidates.append(("relational", sent.strip()))
            break

    for sent in sentences:
        sent_lower = sent.lower()
        for conn in CAUSAL_CONNECTORS_TIER1:
            if conn in sent_lower and len(sent) > 20:
                candidates.append(("tier1", sent.strip()))
                break

    for sent in sentences:
        sent_lower = sent.lower()
        for conn in CAUSAL_CONNECTORS_TIER2:
            if conn in sent_lower and len(sent) > 20:
                candidates.append(("tier2", sent.strip()))
                break

    for sent in sentences:
        sent_lower = sent.lower()
        for conn in CAUSAL_CONNECTORS_AMBIGUOUS:
            if conn not in sent_lower or len(sent) <= 20:
                continue
            if conn == "sebab" and not is_valid_why_sebab(sent):
                continue
            candidates.append(("ambiguous", sent.strip()))
            break

    for sent in sentences:
        sent_lower = sent.lower()
        for conn in PURPOSE_CONNECTORS:
            if conn in sent_lower and len(sent) > 20:
                candidates.append(("purpose", sent.strip()))
                break

    for sent in sentences:
        if len(sent) <= 20:
            continue
        if re.search(r'\buntuk\b', sent.lower()) and is_valid_why_untuk(sent):
            candidates.append(("untuk", sent.strip()))

    return candidates


def extract_why(content: str, what_sentence: str = "", how_sentence: str = "") -> str:
    if not content:
        return "Tidak disebutkan dalam artikel"

    sentences = split_sentences(content)
    exclusions = [s for s in [what_sentence, how_sentence] if s]

    filtered_sentences = [s for s in sentences if not _is_excluded(s, exclusions)]

    candidates = _find_causal_candidates(filtered_sentences)

    if candidates and (what_sentence or how_sentence):
        embed_model = _get_embed_model()
        candidates = _filter_duplicates(candidates, what_sentence, how_sentence, embed_model)

    if not candidates:
        filtered_text = " ".join(filtered_sentences)
        clause = _extract_clause_after_connector(filtered_text, ["karena", "sebab"])
        if clause:
            embed_model = _get_embed_model()
            if not _is_similar_to(clause, what_sentence, embed_model) and \
               not _is_similar_to(clause, how_sentence, embed_model):
                return clause
        return "Tidak disebutkan dalam artikel"

    if len(candidates) == 1:
        return candidates[0][1]

    if what_sentence:
        what_words = set(re.findall(r'\w{4,}', what_sentence.lower()))
        best = None
        best_score = -1
        for ctype, ctext in candidates:
            cand_words = set(re.findall(r'\w{4,}', ctext.lower()))
            overlap = len(what_words & cand_words)
            weight = 1 if ctype in ("inter", "tier1", "relational") else 0
            score = overlap + weight
            if score > best_score:
                best_score = score
                best = ctext
        if best:
            return best

    candidates.sort(key=lambda x: _TYPE_PRIORITY.get(x[0], 9))
    return candidates[0][1]


if __name__ == "__main__":
    test_cases = [
        {
            "title": "Piala Dunia U-20 Terancam Batal",
            "content": (
                "Indonesia telah ditetapkan sebagai tuan rumah Piala Dunia U-20 2023. "
                "Namun, hal ini belakangan diragukan setelah FIFA membatalkan drawing "
                "Piala Dunia U-20 di Bali. "
                "Pembatalan ini berkaitan dengan sikap Gubernur Bali Wayan Koster yang "
                "menolak keikutsertaan Timnas Israel."
            ),
            "what": "Indonesia telah ditetapkan sebagai tuan rumah Piala Dunia U-20 2023.",
            "how": "",
        },
        {
            "title": "Kebakaran gudang gegara korsleting listrik",
            "content": (
                "Kebakaran hebat melanda gudang tekstil di Tangerang pada Kamis dini hari. "
                "Petugas menduga kebakaran terjadi gegara korsleting listrik pada instalasi tua."
            ),
            "what": "Kebakaran hebat melanda gudang tekstil di Tangerang pada Kamis dini hari.",
            "how": "",
        },
    ]

    print("=" * 70)
    print("TEST WHY EXTRACTOR v5 (+ marker relasional)")
    print("=" * 70)
    for case in test_cases:
        print(f"\nJUDUL : {case['title']}")
        result = extract_why(case["content"], what_sentence=case.get("what", ""), how_sentence=case.get("how", ""))
        print(f"WHY   : {result}")
        print("-" * 70)