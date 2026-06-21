"""
gazetteer.py
============
Modul deteksi lokasi (WHERE) berbasis GAZETTEER (daftar kata baku) —
salah satu teknik klasik di Information Retrieval / IE untuk Named Entity
Recognition tanpa model machine learning: cocokkan substring teks dengan
daftar nama lokasi yang sudah diketahui (kota, kabupaten, provinsi, negara).

Sumber daftar: data/lokasi_indonesia.csv (38 provinsi + kota/kabupaten besar
+ negara yang sering disebut di berita Indonesia).
"""

import os
import csv
import re

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lokasi_indonesia.csv")

_locations_cache = None


def load_locations():
    """
    Memuat daftar lokasi dari CSV, diurutkan dari nama TERPANJANG ke
    TERPENDEK. Urutan ini penting: supaya "Jakarta Selatan" tertangkap
    duluan sebelum "Jakarta" (greedy longest-match, hindari match parsial).
    """
    global _locations_cache
    if _locations_cache is not None:
        return _locations_cache

    locations = []
    try:
        with open(_DATA_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["nama"].strip()
                if name and name not in locations:
                    locations.append(name)
    except FileNotFoundError:
        locations = []

    # Urutkan dari paling panjang agar longest-match-first
    locations.sort(key=len, reverse=True)
    _locations_cache = locations
    return locations


def find_locations_in_text(text, max_results=3):
    """
    Mencari semua nama lokasi yang muncul di `text` menggunakan
    word-boundary matching, urutan kemunculan dipertahankan.

    Returns:
        List[str] -- daftar lokasi unik yang ditemukan (maks `max_results`)
    """
    if not text:
        return []

    found = []
    locations = load_locations()
    text_check = text

    for loc in locations:
        if loc in found:
            continue
        pattern = r"\b" + re.escape(loc) + r"\b"
        if re.search(pattern, text_check):
            found.append(loc)
        if len(found) >= max_results:
            break

    return found


def find_location_near_dateline(content, dateline_location=None):
    """
    Strategi prioritas untuk WHERE:
    1. Lokasi dari dateline (mis. "TEMPO.CO, Jakarta -" -> "Jakarta") jika
       lokasi tsb memang dikenal di gazetteer -> paling reliable.
    2. Lokasi pertama yang disebut dalam pola "di <Lokasi>" pada 500 karakter
       awal artikel (lead paragraph), karena WHERE biasanya disebut di awal.
    3. Lokasi pertama yang ditemukan di mana saja dalam artikel.

    Returns:
        str atau None
    """
    locations = load_locations()
    loc_set_lower = {l.lower() for l in locations}

    # 1. Dari dateline
    if dateline_location:
        candidate = dateline_location.strip()
        if candidate.lower() in loc_set_lower:
            return candidate

    # 2. Pola "di <Lokasi>" di lead paragraph
    lead = content[:500] if content else ""
    di_pattern = re.compile(r"\bdi\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})")
    for match in di_pattern.finditer(lead):
        candidate_phrase = match.group(1)
        # cocokkan dengan gazetteer (longest match dalam frasa ini)
        for loc in locations:
            if candidate_phrase.startswith(loc):
                return loc

    # 3. Lokasi pertama di mana pun
    found = find_locations_in_text(content, max_results=1)
    if found:
        return found[0]

    return None
