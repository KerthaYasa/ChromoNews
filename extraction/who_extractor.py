"""
who_extractor.py - v5 (FIXED)
======================
WHO: NER primary + prioritas title + clean gelar jabatan
"""

import re
from collections import defaultdict

from extraction.ner_model import WHO_LABELS, run_ner_chunked

_NER_PIPELINE = None


def inject_ner_pipeline(pipeline):
    global _NER_PIPELINE
    _NER_PIPELINE = pipeline


# =============================================================================
# BLACKLIST RINGAN
# =============================================================================

BLACKLIST_GENERIC = {
    "media sosial", "televisi", "internet", "online", "video", "foto",
    "tiktok", "instagram", "twitter", "facebook", "youtube", "whatsapp",
    "konferensi", "sidang", "rapat", "wawancara",
    "pihak", "pihak terkait", "oknum", "pejabat", "mantan pejabat",
    "aparat", "tokoh", "saksi", "tersangka", "terdakwa",
}

BLACKLIST_MEDIA_DOMAINS = (".com", ".co.id", ".org", ".net", ".id")

DAY_MONTH_WORDS = {
    "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
    "januari", "februari", "maret", "april", "mei", "juni", "juli",
    "agustus", "september", "oktober", "november", "desember",
}


def _basic_filter(name: str) -> bool:
    """Filter ringan"""
    if not name or len(name) < 2:
        return False

    name_lower = name.lower().strip()

    if name_lower in BLACKLIST_GENERIC:
        return False
    if name_lower in DAY_MONTH_WORDS:
        return False
    if any(d in name_lower for d in BLACKLIST_MEDIA_DOMAINS):
        return False
    if re.search(r'\d', name):
        return False
    if len(name.split()) > 5:
        return False

    return True


# =============================================================================
# RANKING
# =============================================================================

def _normalize_key(name: str) -> str:
    return re.sub(r'\s+', ' ', name.lower().strip())


def _score_and_rank(candidates: list, content_len: int, max_results: int = 5) -> list:
    groups = defaultdict(list)
    for c in candidates:
        groups[_normalize_key(c["text"])].append(c)

    # Merge varian nama
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
        display_name = max((it["text"] for it in items), key=len)

        total_score = (freq * 1.0) + (position_bonus * 1.5) + (avg_conf * 0.5)
        scored.append((total_score, freq, display_name))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, _, name in scored[:max_results]]


# =============================================================================
# MAIN - VERSI FIX
# =============================================================================

def extract_who(content: str, title: str = "", max_results: int = 4) -> list:
    global _NER_PIPELINE

    if _NER_PIPELINE is not None:
        entities = run_ner_chunked(_NER_PIPELINE, content)
        
        candidates = []
        for e in entities:
            if e["label"] != "PER":          # Hanya ambil yang benar-benar PERSON
                continue
            
            name = e["text"].strip()
            if not _basic_filter(name):
                continue

            # Filter sederhana berdasarkan pola nama wilayah
            if _looks_like_region(name):
                continue

            # Clean gelar jabatan
            clean_name = re.sub(r'^(Menteri|Direktur|Gubernur|Presiden|Ketua|Wakil|Founder)\s+', 
                              '', name, flags=re.IGNORECASE).strip()
            if clean_name:
                e["text"] = clean_name

            candidates.append(e)

        if candidates:
            ranked = _score_and_rank(candidates, len(content), max_results)
            if ranked:
                return ranked

    return extract_who_fallback(content, max_results)


def _looks_like_region(name: str) -> bool:
    """Deteksi nama wilayah secara sederhana tanpa daftar panjang."""
    lower = name.lower()
    # Pola umum wilayah Indonesia
    if re.search(r'\b(jawa|sumatera|kalimantan|sulawesi|bali|ntt|ntb|maluku|papua|yogyakarta|jakarta|bandung|surabaya|semarang)\b', lower):
        return True
    # Nama provinsi/kota tunggal yang panjang
    if len(name.split()) <= 2 and len(name) > 5 and name[0].isupper():
        return True
    return False


# =============================================================================
# FALLBACK REGEX
# =============================================================================

def extract_who_fallback(content: str, max_results: int = 4) -> list:
    gelar_list = (
        "Menteri|Kepala|Gubernur|Presiden|Wakil Presiden|Komisaris|"
        "Direktur|Direktur Utama|Jaksa|Hakim|Kombes|Brigjen|Irjen|"
        "AKBP|Kompol|Sekretaris|Ketua|Wakil|Staf|Asisten|"
        "Inspektur|Kepala Badan|Deputi"
    )
    candidates = []

    # Pola dengan gelar
    pattern1 = rf'\b({gelar_list})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{0,3}})\b'
    for m in re.finditer(pattern1, content):
        name = f"{m.group(1)} {m.group(2)}"
        if _basic_filter(name):
            candidates.append({"text": name, "label": "PER",
                                "start": m.start(), "end": m.end(), "score": 0.6})

    # Pola nama kapital tanpa gelar
    pattern2 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    for m in re.finditer(pattern2, content):
        name = m.group(1)
        if _basic_filter(name):
            candidates.append({"text": name, "label": "PER",
                                "start": m.start(), "end": m.end(), "score": 0.4})

    if not candidates:
        return ["Tidak disebutkan dalam artikel"]

    return _score_and_rank(candidates, len(content), max_results) or \
        ["Tidak disebutkan dalam artikel"]