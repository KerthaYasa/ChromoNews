"""
where_extractor.py - v6 (+ WHERE_EXACT_BLACKLIST)
=========================
WHERE: NER primary + Gazetteer fallback + blacklist ketat

Changelog:
  v6 — Tambah filter WHERE_EXACT_BLACKLIST dari patterns.py (institusi
       hukum/pemerintah, media, frasa non-lokasi seperti "sidang
       paripurna", "bursa efek indonesia") — sebelumnya hanya
       NOT_LOCATION_PATTERNS (regex) dan WHERE_AMBIGUOUS_NOUNS yang
       dipakai, exact-match blacklist belum tersambung sama sekali.
  v5 — WHERE_AMBIGUOUS_NOUNS filter + WHERE_LOCATION_NOUNS fallback
       pattern (ditambahkan sebelumnya).
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

# =============================================================================
# PENALTI LOKASI GENERIK (level negara)
# =============================================================================
# "Indonesia" valid secara NER tapi terlalu generik dibanding kandidat kota/
# provinsi yang lebih spesifik dalam artikel yang sama. Beri penalti kecil
# supaya kalah ranking KECUALI dia satu-satunya kandidat yang ada.
_COUNTRY_LEVEL_NAMES = {"indonesia"}
_COUNTRY_PENALTY = 0.8


# =============================================================================
# EXCLUDE KALIMAT TOPIC-SHIFT (event lain / masa lalu)
# =============================================================================
# Sama seperti _TOPIC_SHIFT_MARKERS di rule_based_5w1h.py (extract_what),
# tapi dipakai di sini untuk MENGECUALIKAN kalimat yang membicarakan
# peristiwa/lokasi lain dari pencarian WHERE utama. Contoh: "Sebelumnya,
# Anies mendapat penolakan di Aceh..." — Aceh bukan lokasi kejadian utama.
_WHERE_TOPIC_SHIFT_MARKERS = re.compile(
    r'^\s*(sebelumnya|selain itu|di tempat terpisah|di sisi lain|'
    r'pada kesempatan lain)\b',
    re.IGNORECASE,
)


def _get_excluded_char_ranges(content: str) -> list:
    """
    Kembalikan list (start, end) rentang karakter dari kalimat yang diawali
    topic-shift marker, untuk dikecualikan dari kandidat WHERE utama.
    """
    from extraction.text_utils import split_sentences
    ranges = []
    cursor = 0
    for sent in split_sentences(content):
        idx = content.find(sent, cursor)
        if idx == -1:
            continue
        end = idx + len(sent)
        if _WHERE_TOPIC_SHIFT_MARKERS.search(sent):
            ranges.append((idx, end))
        cursor = end
    return ranges


def _is_in_excluded_range(start, excluded_ranges: list) -> bool:
    if start is None:
        return False
    return any(r_start <= start < r_end for r_start, r_end in excluded_ranges)

def _load_gazetteer():
    """Load lokasi dari CSV — untuk gazetteer_bonus lookup saja (urutan tidak relevan di sini)."""
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


def _is_exact_blacklisted(name: str, blacklist: set) -> bool:
    """
    Cek exact-match terhadap WHERE_EXACT_BLACKLIST dari patterns.py
    (institusi hukum/pemerintah, media, frasa non-lokasi). Beda dengan
    NOT_LOCATION_PATTERNS (regex substring) — ini exact match penuh
    setelah normalisasi, supaya tidak overzealous membuang kandidat
    yang cuma sekadar MENGANDUNG kata serupa.
    """
    return _normalize_key(name) in blacklist


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
    country_level = []
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

        # Lokasi level-negara: pisahkan ke bucket sendiri, taruh di akhir
        # daftar kecuali dia satu-satunya kandidat yang ada. Skor tinggi
        # akibat freq/posisi tidak relevan untuk menentukan spesifisitas.
        if key in _COUNTRY_LEVEL_NAMES:
            country_level.append((total_score, display_name))
        else:
            scored.append((total_score, display_name))

    scored.sort(key=lambda x: x[0], reverse=True)
    country_level.sort(key=lambda x: x[0], reverse=True)

    # Kandidat non-negara diprioritaskan penuh; negara cuma dipakai buat
    # mengisi sisa slot kalau kandidat spesifik tidak cukup.
    combined = scored + country_level
    return [name for _, name in combined[:max_results]]


# =============================================================================
# MAIN FUNCTION (VERSI FIX)
# =============================================================================

def extract_where(content: str, max_results: int = 4, dateline_location: str = None) -> list:
    global _NER_PIPELINE

    candidates = []

    if _NER_PIPELINE is not None:
        entities = run_ner_chunked(_NER_PIPELINE, content)
        candidates = [
            e for e in entities
            if e["label"] in WHERE_LABEL_PRIORITY and _basic_filter(e["text"])
        ]

    try:
        from extraction.patterns import (
            NOT_LOCATION_PATTERNS,
            WHERE_AMBIGUOUS_NOUNS,
            WHERE_EXACT_BLACKLIST,
        )
        filtered = []
        for e in candidates:
            text_lower = e["text"].lower().strip()
            if text_lower in WHERE_AMBIGUOUS_NOUNS:
                continue
            if _is_exact_blacklisted(e["text"], WHERE_EXACT_BLACKLIST):
                continue
            if any(re.search(p, e["text"], re.IGNORECASE) for p in NOT_LOCATION_PATTERNS):
                continue
            filtered.append(e)
        candidates = filtered
    except ImportError:
        pass

    # --- BARU: exclude kandidat dari kalimat topic-shift (event lain) ---
    excluded_ranges = _get_excluded_char_ranges(content)
    if excluded_ranges:
        non_shifted = [c for c in candidates if not _is_in_excluded_range(c["start"], excluded_ranges)]
        # Hanya buang jika masih tersisa kandidat lain setelah exclusion —
        # kalau semua kandidat kebetulan ada di kalimat topic-shift
        # (artikel pendek/aneh), lebih baik tetap pakai semua drpd kosong.
        if non_shifted:
            candidates = non_shifted

    if dateline_location and len(dateline_location) >= 3:
        candidates.append({
            "text": dateline_location,
            "label": "GPE",
            "start": 0,
            "end": len(dateline_location),
            "score": 0.95
        })

    if candidates:
        ranked = _score_and_rank(candidates, len(content), max_results)
        if ranked:
            return ranked

    return extract_where_fallback(content, max_results)


# =============================================================================
# FALLBACK GAZETTEER
# =============================================================================

def extract_where_fallback(content: str, max_results: int = 4) -> list:
    from extraction.gazetteer import find_locations_in_text

    candidates = []

    # Pakai fungsi gazetteer resmi — sudah benar: word-boundary + longest-match-first
    # (list dari load_locations() sudah terurut panjang->pendek, jangan diubah jadi set).
    for loc in find_locations_in_text(content, max_results=10):
        for m in re.finditer(r"\b" + re.escape(loc) + r"\b", content, re.IGNORECASE):
            candidates.append({
                "text": loc,
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

    # Pola lokasi spesifik: daerah, wilayah, kawasan + Nama (misal: "kawasan Sudirman")
    try:
        from extraction.patterns import WHERE_LOCATION_NOUNS
        loc_nouns = "|".join(re.escape(n) for n in WHERE_LOCATION_NOUNS)
        pattern2 = rf'\b(?:{loc_nouns})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{0,2}})\b'
        for m in re.finditer(pattern2, content, re.IGNORECASE):
            name = m.group(1)
            # Karena ini mengikuti noun lokasi eksplisit, confidennya sedikit lebih tinggi
            if _basic_filter(name):
                candidates.append({
                    "text": name,
                    "label": "LOC",
                    "start": m.start(),
                    "end": m.end(),
                    "score": 0.4
                })
    except ImportError:
        pass

    # === Blacklist exact-match juga dipakai di fallback gazetteer ===
    try:
        from extraction.patterns import WHERE_EXACT_BLACKLIST
        candidates = [
            c for c in candidates
            if not _is_exact_blacklisted(c["text"], WHERE_EXACT_BLACKLIST)
        ]
    except ImportError:
        pass

    if not candidates:
        return ["Tidak disebutkan dalam artikel"]

    return _score_and_rank(candidates, len(content), max_results) or ["Tidak disebutkan dalam artikel"]