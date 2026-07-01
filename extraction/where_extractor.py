"""
where_extractor.py - v5 (FIXED)
=========================
WHERE: NER primary + Gazetteer fallback + blacklist ketat
"""

import re
from collections import defaultdict

from extraction.ner_model import WHERE_LABEL_PRIORITY, run_ner_chunked

_NER_PIPELINE = None


def inject_ner_pipeline(pipeline):
    global _NER_PIPELINE
    _NER_PIPELINE = pipeline


# =============================================================================
# GAZETTEER -- validator tambahan
# =============================================================================

_GAZETTEER_CACHE = None


def _load_gazetteer():
    """Load lokasi dari CSV"""
    global _GAZETTEER_CACHE
    if _GAZETTEER_CACHE is not None:
        return _GAZETTEER_CACHE
    try:
        from extraction.gazetteer import load_locations
        _GAZETTEER_CACHE = {loc.lower() for loc in load_locations()}
    except Exception:
        _GAZETTEER_CACHE = set()
    return _GAZETTEER_CACHE


# =============================================================================
# FILTER RINGAN
# =============================================================================

GARBAGE_WORDS = {
    "media sosial", "dunia maya", "internet", "online", "daring",
    "ruang sidang", "meja hijau", "ruang tahanan", "ruang kerja",
    "ruang rapat", "ruang", "meja", "kursi", "panggung",
    "pembahasan", "pengumuman", "pemeriksaan", "penetapan",
    "penangkapan", "penahanan", "penggeledahan", "penyidikan",
    "persidangan", "pelimpahan", "putusan", "vonis",
}


def _basic_filter(name: str) -> bool:
    if not name or len(name) < 3:
        return False

    name_lower = name.lower().strip()

    if name_lower in GARBAGE_WORDS:
        return False

    if re.search(r'\d', name_lower):
        return False

    if len(name_lower.split()) > 6:
        return False

    if re.match(r'^(yang|untuk|dalam|dengan|pada|ini|itu)\b', name_lower):
        return False

    return True


# =============================================================================
# RANKING
# =============================================================================

def _normalize_key(name: str) -> str:
    return re.sub(r'\s+', ' ', name.lower().strip())


def _score_and_rank(candidates: list, content_len: int, max_results: int = 4) -> list:
    label_weight = {"GPE": 1.5, "FAC": 1.2, "LOC": 1.0}
    gazetteer = _load_gazetteer()

    groups = defaultdict(list)
    for c in candidates:
        groups[_normalize_key(c["text"])].append(c)

    keys_sorted = sorted(groups.keys(), key=len, reverse=True)
    merged = {}
    absorbed = set()
    for k in keys_sorted:
        if k in absorbed:
            continue
        merged[k] = list(groups[k])
        for k2 in keys_sorted:
            if k2 == k or k2 in absorbed:
                continue
            if k2 in k:
                merged[k].extend(groups[k2])
                absorbed.add(k2)

    scored = []
    for key, items in merged.items():
        freq = len(items)
        first_pos = min((it["start"] for it in items if it["start"] is not None),
                         default=content_len)
        avg_conf = sum(it["score"] for it in items) / len(items)
        position_bonus = max(0, 1 - (first_pos / max(content_len, 1)))

        label_counts = defaultdict(int)
        for it in items:
            label_counts[it["label"]] += 1
        dominant_label = max(label_counts, key=label_counts.get)
        weight = label_weight.get(dominant_label, 1.0)

        gazetteer_bonus = 0.5 if key in gazetteer else 0.0
        display_name = max((it["text"] for it in items), key=len)

        total_score = (freq * 1.0 * weight) + (position_bonus * 1.0) + \
                      (avg_conf * 0.5) + gazetteer_bonus
        scored.append((total_score, display_name))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in scored[:max_results]]


# =============================================================================
# MAIN FUNCTION (VERSI FIX)
# =============================================================================

def extract_where(content: str, max_results: int = 4, dateline_location: str = None) -> list:
    """
    WHERE: NER sebagai metode utama + blacklist ketat.
    Gazetteer hanya sebagai fallback jika NER tidak menghasilkan hasil valid.
    """
    global _NER_PIPELINE

    candidates = []
    
    # === NER PRIMARY ===
    if _NER_PIPELINE is not None:
        entities = run_ner_chunked(_NER_PIPELINE, content)
        candidates = [
            e for e in entities
            if e["label"] in WHERE_LABEL_PRIORITY and _basic_filter(e["text"])
        ]

    # === Blacklist tambahan dari patterns.py ===
    try:
        from extraction.patterns import NOT_LOCATION_PATTERNS
        filtered = []
        for e in candidates:
            if not any(re.search(p, e["text"], re.IGNORECASE) for p in NOT_LOCATION_PATTERNS):
                filtered.append(e)
        candidates = filtered
    except:
        pass

    # === Dateline boost (sangat reliable) ===
    if dateline_location and len(dateline_location) >= 3:
        candidates.append({
            "text": dateline_location, 
            "label": "GPE",
            "start": 0, 
            "end": len(dateline_location), 
            "score": 0.95
        })

    # Jika ada hasil dari NER + dateline
    if candidates:
        ranked = _score_and_rank(candidates, len(content), max_results)
        if ranked:
            return ranked

    # === FALLBACK ke Gazetteer ===
    return extract_where_fallback(content, max_results)


# =============================================================================
# FALLBACK GAZETTEER
# =============================================================================

def extract_where_fallback(content: str, max_results: int = 4) -> list:
    gazetteer = _load_gazetteer()
    content_lower = content.lower()
    candidates = []

    for loc in gazetteer:
        if len(loc) < 3:
            continue
        for m in re.finditer(re.escape(loc), content_lower):
            cap = " ".join(w.capitalize() for w in loc.split())
            candidates.append({
                "text": cap, 
                "label": "GPE",
                "start": m.start(), 
                "end": m.end(), 
                "score": 0.5
            })

    # Pola generik "di/ke/dari ..."
    pattern = r'\b(?:di|ke|dari)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
    for m in re.finditer(pattern, content):
        name = m.group(1)
        if _basic_filter(name):
            candidates.append({
                "text": name, 
                "label": "LOC",
                "start": m.start(), 
                "end": m.end(), 
                "score": 0.3
            })

    if not candidates:
        return ["Tidak disebutkan dalam artikel"]

    return _score_and_rank(candidates, len(content), max_results) or ["Tidak disebutkan dalam artikel"]