"""
ner_helper.py
=============
Heuristik ringan untuk deteksi entitas (WHO) tanpa model NER machine learning.

Pendekatan: "Capitalization + Title/Org Gazetteer Heuristic" — teknik klasik
pra-deep-learning untuk Named Entity Recognition berbasis aturan:

  1. Cari frasa kapital berurutan (mis. "Zulkifli Hasan", "Rafael Alun
     Trisambodo") yang BUKAN di awal kalimat (untuk menghindari salah
     tangkap kata pertama kalimat yang otomatis kapital).
  2. Beri skor lebih tinggi pada kandidat yang:
       - didahului gelar/jabatan (Menteri, Presiden, Direktur, dst -> patterns.TITLE_PREFIXES)
       - berdekatan dengan kata kerja pelaporan (kata, ujar, ungkap -> patterns.REPORTING_VERBS)
       - cocok dengan daftar lembaga dikenal (KPK, Polri, dst -> patterns.KNOWN_ORGS)
       - muncul juga di judul artikel (judul biasanya memuat aktor utama)
  3. Kandidat dengan skor tertinggi dikembalikan sebagai WHO.

Ini BUKAN NER yang sempurna, tapi cukup efektif untuk berita Bahasa
Indonesia yang well-formed, dan yang lebih penting: hasilnya bisa
ditelusuri/dijelaskan baris per baris (beda dengan model NER black-box).
"""

import re

from .patterns import TITLE_PREFIXES, REPORTING_VERBS, KNOWN_ORGS, SCRAPING_ARTIFACTS
from .gazetteer import load_locations

# Pola frasa kapital berurutan: 2-4 kata diawali huruf besar
_CAP_PHRASE_PATTERN = re.compile(r"\b([A-Z][a-zA-Z\.]+(?:\s+[A-Z][a-zA-Z\.]+){1,3})\b")

# Kata-kata kapital umum yang BUKAN nama (awal kalimat, singkatan media, dll)
_BLACKLIST_WORDS = {
    "Ia", "Dia", "Mereka", "Namun", "Akan", "Hal", "Pada", "Dalam",
    "Menurut", "Setelah", "Sebelum", "Karena", "Selain", "Kemudian",
}

# Tambahan Baru: Filter kata depan yang sering salah terdeteksi sebagai entitas (False Positive)
_INVALID_STARTS = {
    "Pada", "Hari", "Di", "Dalam", "Menurut", "Berdasarkan", 
    "Sementara", "Untuk", "Ini", "Itu", "Dan", "Atau"
}


def _split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def find_who(content, title=""):
    """
    Mengekstrak kandidat WHO TERBAIK (1 kandidat) dari isi artikel.
    Lihat find_who_multi() untuk versi multi-kandidat.
    """
    candidates = _score_who_candidates(content, title)
    if not candidates:
        return None
    best = max(candidates.items(), key=lambda x: x[1])
    return best[0]


def find_who_multi(content, title="", max_who=3, min_score=1.5):
    """
    Versi MULTI-kandidat dari find_who() -- dipakai sebagai fallback saat
    model NER (ner_model.py) tidak tersedia, supaya WHO tetap bisa
    multi-entitas walau TANPA model machine learning sama sekali (murni
    heuristik kapitalisasi + skor, lihat docstring modul di atas).

    Args:
        max_who: maksimum kandidat yang dikembalikan
        min_score: skor minimum supaya kandidat dianggap cukup yakin
                   (hindari nangkap kata kapital acak yang skornya rendah)

    Returns:
        List[str] -- kandidat WHO diurutkan dari skor tertinggi, atau
        list kosong [] kalau tidak ada kandidat yang lolos min_score
    """
    candidates = _score_who_candidates(content, title)
    if not candidates:
        return []

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    results = []
    seen_lower = set()
    for name, score in sorted_candidates:
        if score < min_score:
            break
        key = name.lower()
        # Dedup: skip kalau ini cuma substring dari kandidat yang sudah
        # diambil (mis. "Jokowi" vs "Joko Widodo" / "JokoWidodo")
        if any(key in s or s in key for s in seen_lower):
            continue
        seen_lower.add(key)
        results.append(name)
        if len(results) >= max_who:
            break

    return results


def _score_who_candidates(content, title=""):
    """Logika inti penilaian kandidat WHO (dipakai find_who & find_who_multi)."""
    if not content:
        return None

    sentences = _split_sentences(content)
    candidates = {}  # nama -> skor

    for sent_idx, sent in enumerate(sentences[:6]):  # fokus 6 kalimat pertama
        # Cari kandidat, tapi lewati kata pertama kalimat (sering kapital krn posisi)
        for match in _CAP_PHRASE_PATTERN.finditer(sent):
            phrase = match.group(1).strip()
            first_word = phrase.split()[0]

            # ===== PERBAIKAN: Tolak konjungsi dan posisi awalan palsu =====
            if first_word in _BLACKLIST_WORDS or first_word in _INVALID_STARTS:
                continue
            # ==============================================================

            if phrase.lower() in SCRAPING_ARTIFACTS:
                continue  # "Baca Juga", "Gambas:Video", dst -- bukan entitas
            if phrase in load_locations():
                continue  # ini lokasi (sudah ditangani gazetteer.py utk WHERE), bukan WHO
            if match.start() == 0:
                # ini kata pertama kalimat -> skip kecuali frasa > 2 kata
                # (nama orang biasanya >=2 kata, kalimat awal jarang begitu
                #  kecuali memang nama)
                if len(phrase.split()) < 2:
                    continue

            score = 1.0
            # Boost: didahului gelar/jabatan
            preceding_text = sent[max(0, match.start() - 30):match.start()]
            if any(title_word.lower() in preceding_text.lower() for title_word in TITLE_PREFIXES):
                score += 2.0
            # Boost: dekat dengan kata kerja pelaporan dalam kalimat yang sama
            if any(rv in sent.lower() for rv in REPORTING_VERBS):
                score += 1.0
            # Boost: ada di kalimat pertama (lead) -> kemungkinan subjek utama
            if sent_idx == 0:
                score += 1.5
            # Boost: juga muncul di judul artikel
            if title and phrase.lower() in title.lower():
                score += 2.0

            candidates[phrase] = candidates.get(phrase, 0) + score

    # Tambahkan lembaga dikenal yang disebut eksplisit
    for org in KNOWN_ORGS:
        pattern = r"\b" + re.escape(org) + r"\b"
        if re.search(pattern, content):
            score = 1.5
            if title and org.lower() in title.lower():
                score += 2.0
            candidates[org] = candidates.get(org, 0) + score

    return candidates if candidates else None