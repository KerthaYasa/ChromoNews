"""
ner_helper.py
=============
Heuristik ringan untuk deteksi entitas (WHO) tanpa model NER ML.

REVISI v5:
  - Prioritaskan PERSON (PER) di atas ORG saat scoring
  - Perbaiki false positive: "Perumahan Rakyat", "Mahkamah Agung" dll
    yang terpotong dari frasa jabatan tidak boleh jadi WHO
  - Perkuat deteksi nama orang 2-4 kata
  - Tambah daftar blacklist false-positive organisasi yang sering muncul
"""

import re

from .patterns import TITLE_PREFIXES, REPORTING_VERBS, KNOWN_ORGS, SCRAPING_ARTIFACTS
from .gazetteer import load_locations
from extraction.text_utils import split_sentences

# Pola frasa kapital berurutan: 2-4 kata diawali huruf besar
_CAP_PHRASE_PATTERN = re.compile(r"\b([A-Z][a-zA-Z\.]+(?:\s+[A-Z][a-zA-Z\.]+){1,3})\b")

# Kata-kata kapital umum yang BUKAN nama
_BLACKLIST_WORDS = {
    "Ia", "Dia", "Mereka", "Namun", "Akan", "Hal", "Pada", "Dalam",
    "Menurut", "Setelah", "Sebelum", "Karena", "Selain", "Kemudian",
    "Sementara", "Untuk", "Ini", "Itu", "Dan", "Atau", "Dengan",
    "Ketika", "Bahwa", "Para", "Sebuah", "Berbagai", "Sejak",
    "Namun", "Selama", "Antara", "Bagi", "Atas", "Bawah",
}

_INVALID_STARTS = {
    "Pada", "Hari", "Di", "Dalam", "Menurut", "Berdasarkan",
    "Sementara", "Untuk", "Ini", "Itu", "Dan", "Atau",
    "Pasal", "Ayat", "Nomor", "No", "Undang", "Peraturan",
    "Bab", "Bagian", "Total", "Nilai", "Harga", "Jumlah",
    "Tahun", "Bulan", "Hari", "Waktu", "Senin", "Selasa",
    "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu",
    # Event / konferensi bukan entitas WHO
    "KTT", "Konferensi", "Summit", "Rapat", "Sidang", "Forum",
    "Pertemuan", "Partnership", "Country", "The", "Info",
    # Singkatan bukan nama
    "INFO",
}

# Frasa yang sering menjadi false positive (bagian dari jabatan / frasa lain)
# Kandidat ini HARUS didampingi nama setelahnya agar valid sebagai WHO
_PARTIAL_TITLE_FRAGMENTS = {
    "perumahan rakyat", "umum dan perumahan", "pekerjaan umum",
    "direktur utama pt", "wakil direktur utama pt",
    "komisaris pt", "mahkamah agung", "pengadilan negeri",
    "kementerian keuangan", "direktorat jenderal",
}

# Kata pertama yang menandakan ini nama ORANG (bukan ORG)
_PERSON_NAME_INDICATORS = {
    "achmad", "ahmad", "ali", "andi", "agus", "alex", "amir",
    "budi", "bambang", "basuki", "bahlil",
    "catur", "cipto",
    "dito", "doni", "djoko", "dedi",
    "erick", "edy", "edo",
    "ferry", "firman",
    "gatot", "guntur",
    "heru", "hadi",
    "irwan", "ivan",
    "joko", "jokowi", "james",
    "kombes", "koordinator",
    "luhut", "lutfi",
    "mario", "mahfud",
    "nanang",
    "prabowo", "puan", "piyush",
    "rafael", "rizal", "ridwan",
    "siti", "sri", "surya",
    "tito", "teguh",
    "umar",
    "victor",
    "wahyu", "wika",
    "yenny", "yusuf", "yustinus",
    "zulkifli",
}


def _is_partial_title(phrase):
    """Cek apakah phrase ini hanya potongan dari jabatan (false positive)."""
    pl = phrase.lower()
    for frag in _PARTIAL_TITLE_FRAGMENTS:
        if pl.startswith(frag) or pl == frag:
            return True
    return False


def _is_likely_person(phrase):
    """
    Heuristik: apakah phrase ini kemungkinan nama ORANG (bukan ORG)?
    Nama orang Indonesia biasanya:
    - 2-4 kata
    - Tidak mengandung kata2 seperti PT, CV, Tbk, Kementerian, dll
    - Sering diawali nama kecil / nama Arab / nama Jawa yang dikenal
    """
    words = phrase.split()
    if len(words) < 2 or len(words) > 5:
        return False

    # Cek apakah kata pertama adalah indikator nama orang
    first = words[0].lower()
    if first in _PERSON_NAME_INDICATORS:
        return True

    # Cek apakah mengandung kata2 penanda bukan nama orang
    org_indicators = {
        "pt", "cv", "tbk", "persero", "kementerian", "direktorat",
        "badan", "lembaga", "komisi", "dewan", "mahkamah", "pengadilan",
        "kepolisian", "kejaksaan", "pemerintah",
    }
    for w in words:
        if w.lower() in org_indicators:
            return False

    # Jika semua kata 2-4 huruf kapital di tengah kalimat, cenderung nama orang
    if all(w[0].isupper() for w in words) and 2 <= len(words) <= 4:
        return True

    return False


def find_who(content, title=""):
    """Mengekstrak kandidat WHO TERBAIK (1 kandidat)."""
    candidates = _score_who_candidates(content, title)
    if not candidates:
        return None
    best = max(candidates.items(), key=lambda x: x[1])
    return best[0]


def find_who_multi(content, title="", max_who=3, min_score=1.5):
    """
    Versi MULTI-kandidat dari find_who().
    REVISI v5: prioritaskan PER (nama orang) di atas ORG.
    """
    candidates = _score_who_candidates(content, title)
    if not candidates:
        return []

    # Pisahkan kandidat menjadi "likely person" dan "org/other"
    persons = {k: v for k, v in candidates.items() if _is_likely_person(k)}
    orgs = {k: v for k, v in candidates.items() if not _is_likely_person(k)}

    # Gabungkan: persons dulu, baru orgs
    merged = []
    for name, score in sorted(persons.items(), key=lambda x: x[1], reverse=True):
        merged.append((name, score))
    for name, score in sorted(orgs.items(), key=lambda x: x[1], reverse=True):
        merged.append((name, score))

    results = []
    seen_lower = set()
    for name, score in merged:
        if score < min_score:
            continue
        key = name.lower()

        # Dedup: jika ada yang sudah masuk dan ini LEBIH PANJANG,
        # ganti yang lama dengan yang ini (prefer frase lebih spesifik)
        replaced = False
        for i, existing in enumerate(results):
            existing_key = existing.lower()
            if existing_key in key and len(key) > len(existing_key) + 2:
                # nama baru lebih panjang & mencakup yang lama -> ganti
                results[i] = name
                seen_lower.discard(existing_key)
                seen_lower.add(key)
                replaced = True
                break
            elif key in existing_key:
                # nama baru lebih pendek, yang lama sudah lebih spesifik -> skip
                replaced = True
                break

        if not replaced:
            if key not in seen_lower:
                seen_lower.add(key)
                results.append(name)

        if len(results) >= max_who:
            break

    return results


def _score_who_candidates(content, title=""):
    """Logika inti penilaian kandidat WHO."""
    if not content:
        return None

    sentences = split_sentences(content)
    candidates = {}

    for sent_idx, sent in enumerate(sentences[:10]):  # fokus 10 kalimat pertama
        for match in _CAP_PHRASE_PATTERN.finditer(sent):
            phrase = match.group(1).strip()
            words = phrase.split()
            first_word = words[0] if words else ""

            # Tolak blacklist dan invalid starts
            if first_word in _BLACKLIST_WORDS or first_word in _INVALID_STARTS:
                continue

            if phrase.lower() in [a.lower() for a in SCRAPING_ARTIFACTS]:
                continue

            # Tolak lokasi
            loc_lower = {l.lower() for l in load_locations()}
            if phrase.lower() in loc_lower:
                continue

            # Tolak partial title fragments
            if _is_partial_title(phrase):
                continue

            # Tolak lokasi yang menyamar sebagai WHO
            if _is_location_masquerading_as_who(phrase):
                continue

            # Tolak frasa yang mengandung titik di tengah (typo/artefak scraping)
            # mis. "Zulkifli.Di India", "NamaOrang.Com Judul"
            if re.search(r'[a-zA-Z]\.[A-Z]', phrase):
                # Coba ambil bagian sebelum titik
                clean_ph = phrase.split('.')[0].strip()
                if len(clean_ph.split()) >= 2:
                    phrase = clean_ph
                else:
                    continue

            # Tolak kata yang berakhiran verba tergabung (typo scraping)
            if _has_fused_verb(phrase):
                # Coba potong kata terakhir
                words_ph = phrase.split()
                if len(words_ph) > 1:
                    phrase = " ".join(words_ph[:-1])
                else:
                    continue

            # Tolak jika setelah semua filter, phrase tidak valid
            if not phrase or len(phrase) < 3:
                continue

            # Skip kata tunggal di awal kalimat
            if match.start() == 0 and len(words) < 2:
                continue

            # Hitung skor dasar
            score = 1.0

            # BONUS BESAR: kemungkinan nama orang
            if _is_likely_person(phrase):
                score += 2.5

            # Boost: didahului gelar/jabatan
            preceding_text = sent[max(0, match.start() - 60):match.start()]
            if any(tp.lower() in preceding_text.lower() for tp in TITLE_PREFIXES):
                score += 2.0

            # Boost: dekat dengan reporting verb
            if any(rv in sent.lower() for rv in REPORTING_VERBS):
                score += 1.0

            # Boost: kalimat lead (pertama atau kedua)
            if sent_idx == 0:
                score += 2.0
            elif sent_idx == 1:
                score += 1.0

            # Boost: muncul di judul
            if title and phrase.lower() in title.lower():
                score += 2.5

            # Boost: 2-3 kata (nama orang Indonesia paling umum 2-3 kata)
            if 2 <= len(words) <= 3:
                score += 0.5

            candidates[phrase] = candidates.get(phrase, 0) + score

    # Tambahkan lembaga dikenal
    for org in KNOWN_ORGS:
        pattern = r"\b" + re.escape(org) + r"\b"
        if re.search(pattern, content):
            score = 1.5
            if title and org.lower() in title.lower():
                score += 2.0
            candidates[org] = candidates.get(org, 0) + score

    return candidates if candidates else None


# =============================================================================
# TAMBAHAN: Filter untuk mendeteksi kata yang "tergabung" dengan verba
# Contoh: "Puromenyebut" = "Puro" + "menyebut" (typo scraping)
# =============================================================================
_VERB_FUSED_PATTERN = re.compile(
    r'(?:menyebut|mengatakan|menyatakan|mengaku|menilai|meminta|'
    r'melakukan|memeriksa|menjelaskan|menuturkan|menegaskan|'
    r'mengungkapkan|menambahkan)$',
    re.IGNORECASE,
)

_LOCATION_WORDS_IN_WHO = {
    "asia selatan", "asia tenggara", "asia timur", "asia barat",
    "eropa", "afrika", "Amerika", "australia",
    "pulau jawa", "pulau sumatera", "pulau kalimantan",
    "jawa barat", "jawa timur", "jawa tengah",
}


def _has_fused_verb(phrase):
    """Cek apakah ada kata dalam phrase yang berupa verba tergabung (typo scraping)."""
    words = phrase.split()
    if not words:
        return False
    verb_suffix_re = re.compile(
        r'[A-Z][a-z]+(menyebut|mengatakan|mengaku|meminta|menjelas|mengung|menambah)',
        re.IGNORECASE,
    )
    for w in words:
        if len(w) > 8 and verb_suffix_re.search(w):
            return True
    return False


def _is_location_masquerading_as_who(phrase):
    """Cek apakah phrase ini sebenarnya lokasi geografis, bukan WHO."""
    pl = phrase.lower()
    if pl in _LOCATION_WORDS_IN_WHO:
        return True
    # Wilayah dengan pola "X Selatan/Utara/Timur/Barat/Tengah"
    if re.search(r'\b(selatan|utara|timur|barat|tengah)\b', pl):
        return True
    return False
