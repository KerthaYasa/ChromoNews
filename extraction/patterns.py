"""
patterns.py
============
Kumpulan pattern (regex) dan kamus kata kunci yang dipakai oleh
`rule_based_5w1h.py` untuk mengekstrak elemen 5W1H secara ALGORITMIK
(tanpa AI/LLM).

Kenapa pendekatan rule-based?
-----------------------------
1. Transparan & bisa dijelaskan langkah demi langkah (cocok untuk laporan STKI,
   beda dengan AI generatif yang prosesnya black-box).
2. Deterministik -> hasil bisa diuji presisi/recall-nya terhadap ground truth.
3. Murah & cepat -> tidak butuh panggilan API per artikel untuk EKSTRAKSI
   (API hanya dipakai belakangan, sekali, untuk merangkai hasil ekstraksi
   jadi kalimat natural di `paraphraser.py`).
4. Berita Bahasa Indonesia (terutama media online) punya struktur penulisan
   yang relatif konsisten (piramida terbalik, dateline, pola atribusi kutipan),
   sehingga pattern matching sederhana sudah cukup efektif menangkap elemen
   5W1H tanpa perlu model NLP berat (NER/POS tagger) yang butuh instalasi
   model besar & lebih sulit dijelaskan ke pembaca awam.
"""

import re

# =============================================================================
# 1. POLA TANGGAL (untuk WHEN)
# =============================================================================
BULAN_INDO = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]
_BULAN_REGEX = "|".join(BULAN_INDO)

# "13 Maret 2023" / "13-14 Maret 2023" / "1314 Maret 2023" (typo umum di scraping)
DATE_PATTERN_FULL = re.compile(
    rf"\b\d{{1,2}}(?:[\s\-–]*\d{{1,2}})?\s+(?:{_BULAN_REGEX})\s+\d{{4}}\b",
    re.IGNORECASE,
)

# "Senin, 13 Maret 2023" / "Senin (13/4)" / "Senin (13/4/2023)"
DAY_NAMES = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_DAY_REGEX = "|".join(DAY_NAMES)
DATE_PATTERN_WITH_DAY = re.compile(
    rf"\b(?:{_DAY_REGEX})[,]?\s*(?:\(?\d{{1,2}}[/\-]\d{{1,2}}(?:[/\-]\d{{2,4}})?\)?|\d{{1,2}}\s+(?:{_BULAN_REGEX})\s+\d{{4}})",
    re.IGNORECASE,
)

# dd/mm/yyyy atau dd-mm-yyyy
DATE_PATTERN_NUMERIC = re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b")

DATE_PATTERN_SHORTHAND = re.compile(r"\b(?:Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)?\s*\(\d{1,2}/\d{1,2}(?:/\d{2,4})?\)", re.IGNORECASE)
DATE_PATTERN_RELATIVE = re.compile(r"(?i)\b(?:hari ini|kemarin|besok)\b")

DATE_PATTERNS = [
    DATE_PATTERN_WITH_DAY, 
    DATE_PATTERN_FULL, 
    DATE_PATTERN_NUMERIC, 
    DATE_PATTERN_SHORTHAND, 
    DATE_PATTERN_RELATIVE
]


# =============================================================================
# 2. KATA PENGHUBUNG KAUSAL (untuk WHY) — "apa SEBAB-nya"
# =============================================================================
CAUSAL_CONNECTORS = [
    "atas dasar itulah", "atas dasar itu", "atas dasar",
    "disebabkan oleh", "disebabkan karena", "disebabkan",
    "diakibatkan oleh", "diakibatkan",
    "akibat dari", "akibat",
    "karena", "lantaran", "sebab", "imbas dari", "imbas",
    "dipicu oleh", "dipicu",
    "buntut dari", "buntut",
]
INTER_SENTENCE_CAUSAL = [
    "pasalnya", "oleh karena itu", "sebab itu", "karenanya", "alhasil"
]

# =============================================================================
# 2b. KATA PENGHUBUNG TUJUAN/MOTIVASI (untuk WHY) — beda dari kausal di atas:
#     kausal = "apa SEBAB sebuah peristiwa terjadi" (ke belakang/retrospektif)
#     purposive = "apa TUJUAN/HARAPAN dari sebuah tindakan" (ke depan/prospektif)
#     Banyak berita non-kriminal (kunjungan, pernyataan, klarifikasi) menulis
#     motivasi dengan penanda tujuan, bukan penanda sebab-akibat — kalau
#     hanya CAUSAL_CONNECTORS yang dicek, kasus seperti ini akan lolos
#     tak terdeteksi (recall rendah). Ditangani terpisah dari "untuk" yang
#     terlalu generik (lihat extract_why: "untuk" dicari TERBATAS di
#     1-2 kalimat awal, dekat kata kerja utama, agar tidak overmatch).
# =============================================================================
PURPOSE_CONNECTORS = [
    "dengan tujuan", "bertujuan untuk", "bertujuan",
    "dalam rangka", "demi", "guna",
    "agar", "supaya", "berharap", "diharapkan",
]

# =============================================================================
# 3. KATA PENGHUBUNG CARA/MODUS (untuk HOW) — "BAGAIMANA caranya"
# =============================================================================
METHOD_CONNECTORS = [
    "dengan cara", "dengan modus", "modus operandi", "modusnya",
    "melalui", "lewat", "menggunakan", "dengan menggunakan",
    "dengan memanfaatkan", "memanfaatkan",
    "secara langsung", "langsung mendatangi", "dengan mendatangi",
    "dalam akun", "melalui akun", "via akun", "lewat akun",
    "berawal dari", "bermula dari", "diawali dari", "diawali",
]

METHOD_VERB_PATTERN = re.compile(r"\bdengan\s+(?:meng|mem|men|meny|me|di|ber|ter|pe)[a-z]+\b", re.IGNORECASE)

# Kata penghubung generik penanda proses/urutan kejadian ("usai", "setelah").
# DIPISAH dari METHOD_CONNECTORS karena terlalu umum/sering muncul di teks
# berita apa pun -- kalau dicari di SELURUH artikel akan banyak salah
# tangkap (overmatch). Maka HANYA dipakai sebagai fallback TERAKHIR dan
# DIBATASI pada 3 kalimat awal saja (lihat extract_how()).
HOW_FALLBACK_CONNECTORS = ["usai", "setelah"]

# =============================================================================
# 4. KATA KERJA PELAPORAN (reporting verbs) — penanda kalimat kutipan,
#    dipakai untuk membantu deteksi WHO (nama yang dikutip biasanya pelaku
#    utama/narasumber kunci dalam berita)
# =============================================================================
REPORTING_VERBS = [
    "kata", "ujar", "ungkap", "jelas", "tutur", "tegas", "papar",
    "terang", "imbuh", "tambah", "lanjut", "kata dia", "menurut",
    "tuturnya", "katanya", "ujarnya", "jelasnya", "tegasnya",
]

# =============================================================================
# 5. GELAR / JABATAN — penanda kuat bahwa frasa di sekitarnya adalah nama
#    orang/lembaga (membantu WHO), sekaligus dipakai memfilter agar tidak
#    salah tangkap kata kapital biasa (mis. awal kalimat)
# =============================================================================
TITLE_PREFIXES = [
    "Presiden", "Wakil Presiden", "Menteri", "Wakil Menteri", "Mendag",
    "Menko", "Menkeu", "Menkumham", "Mendagri", "Menhan", "Gubernur",
    "Wali Kota", "Walikota", "Bupati", "Wakil Bupati", "Camat",
    "Komisaris", "Direktur", "Direktur Utama", "Kepala", "Ketua",
    "Wakil Ketua", "Anggota DPR", "Anggota DPRD", "Jaksa", "Hakim",
    "Polisi", "Kapolri", "Kapolda", "Kapolres", "Brigjen", "Irjen",
    "Kombes", "AKP", "Mayor", "Kolonel", "Jenderal", "Letnan",
    "Dokter", "Profesor", "Prof", "Dr",
]

# Lembaga/instansi umum (sering jadi subjek WHO juga, bukan cuma orang)
KNOWN_ORGS = [
    "KPK", "Polri", "TNI", "DPR", "MPR", "MK", "MA", "BPK", "BPOM",
    "KPU", "Bawaslu", "OJK", "BI", "Bank Indonesia", "Kemenkeu",
    "Kemendagri", "Kemenkes", "Kemendikbud", "Kementerian Keuangan",
    "PPATK", "Bareskrim", "Kejaksaan Agung", "Kejagung", "Pemprov",
    "Pemkot", "Pemkab", "BNPB", "BMKG", "BPBD",
]

# =============================================================================
# 6. POLA DATELINE — prefix khas media online di awal artikel, mis:
#    "TEMPO.CO, Jakarta -" / "Suara.com -" / "INFO NASIONAL -"
#    Berguna untuk: (a) membersihkan kalimat lead sebelum dipakai sbg WHAT,
#                   (b) lokasi setelah koma sering kali adalah WHERE.
# =============================================================================
DATELINE_PATTERN = re.compile(
    r"^([A-Za-z\.\s]{2,25})\s*,\s*([A-Za-z\.\s]{2,25})\s*[-–]\s*"
)
DATELINE_SIMPLE_PATTERN = re.compile(r"^([A-Z][\w\.\s]{2,25})\s*[-–]\s*")

# =============================================================================
# 7. ARTEFAK NAVIGASI/UI SCRAPING — label seperti "Baca Juga", "Gambas:Video"
#    sering ikut ter-scrape dari halaman berita, dan karena ditulis kapital
#    sering salah tertangkap heuristik kapitalisasi sebagai "entitas" WHO.
#    Daftar ini dipakai untuk MEMBUANG kandidat semacam itu dari hasil WHO.
# =============================================================================
SCRAPING_ARTIFACTS = [
    "baca juga", "gambas video", "gambas:video",
    "scroll untuk melanjutkan", "scroll untuk lanjutkan",
    "advertisement", "lihat juga", "simak juga",
]
