"""
patterns.py
============
Kumpulan pattern (regex) dan kamus kata kunci untuk ekstraksi 5W1H.
REVISI v5: Perbaikan DATE_PATTERNS (hapus relatif), WHERE filter organisasi,
           WHO gelar diperluas, WHY konektor diperkuat.
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

# "13 Maret 2023" / "13-14 Maret 2023"
DATE_PATTERN_FULL = re.compile(
    rf"\b\d{{1,2}}(?:[\s\-–]*\d{{1,2}})?\s+(?:{_BULAN_REGEX})\s+\d{{4}}\b",
    re.IGNORECASE,
)

# "Senin, 13 Maret 2023" / "Senin (13/4/2023)"
DAY_NAMES = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_DAY_REGEX = "|".join(DAY_NAMES)
DATE_PATTERN_WITH_DAY = re.compile(
    rf"\b(?:{_DAY_REGEX})[,]?\s*(?:\(?\d{{1,2}}[/\-]\d{{1,2}}(?:[/\-]\d{{2,4}})?\)?|\d{{1,2}}\s+(?:{_BULAN_REGEX})\s+\d{{4}})",
    re.IGNORECASE,
)

# dd/mm/yyyy atau dd-mm-yyyy (hanya jika 4 digit tahun, agar tidak ambil angka lain)
DATE_PATTERN_NUMERIC = re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\b")

DATE_PATTERN_SHORTHAND = re.compile(
    r"\b(?:Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu)?\s*\(\d{1,2}/\d{1,2}(?:/\d{2,4})?\)",
    re.IGNORECASE,
)

# CATATAN: DATE_PATTERN_RELATIVE DIHAPUS — "hari ini", "kemarin", "besok"
# tidak akan dimasukkan ke hasil WHEN karena tidak informatif (tidak absolut).
# Akan digantikan oleh metadata_date jika tersedia.
DATE_PATTERN_RELATIVE = re.compile(
    r"(?i)\b(?:hari ini|kemarin|besok|tadi malam|malam tadi|sore tadi|pagi tadi)\b"
)

DATE_PATTERNS = [
    DATE_PATTERN_WITH_DAY,
    DATE_PATTERN_FULL,
    DATE_PATTERN_NUMERIC,
    DATE_PATTERN_SHORTHAND,
    # DATE_PATTERN_RELATIVE SENGAJA TIDAK DIMASUKKAN
]


# =============================================================================
# 2. KATA PENGHUBUNG KAUSAL (untuk WHY)
# =============================================================================
CAUSAL_CONNECTORS = [
    "atas dasar itulah", "atas dasar itu", "atas dasar",
    "disebabkan oleh", "disebabkan karena", "disebabkan",
    "diakibatkan oleh", "diakibatkan",
    "akibat dari", "akibat",
    "karena", "lantaran", "sebab", "imbas dari", "imbas",
    "dipicu oleh", "dipicu",
    "buntut dari", "buntut",
    "diduga karena", "dikarenakan",
]
INTER_SENTENCE_CAUSAL = [
    "pasalnya", "oleh karena itu", "sebab itu", "karenanya", "alhasil",
    "hal itu disebabkan", "hal tersebut disebabkan",
]

# =============================================================================
# 2b. KATA PENGHUBUNG TUJUAN/MOTIVASI (untuk WHY)
# =============================================================================
PURPOSE_CONNECTORS = [
    "dengan tujuan", "bertujuan untuk", "bertujuan",
    "dalam rangka", "demi", "guna",
    "agar", "supaya", "berharap", "diharapkan",
]

# =============================================================================
# 3. KATA PENGHUBUNG CARA/MODUS (untuk HOW)
# =============================================================================
METHOD_CONNECTORS = [
    "dengan cara", "dengan modus", "modus operandi", "modusnya",
    "melalui", "lewat", "menggunakan", "dengan menggunakan",
    "dengan memanfaatkan", "memanfaatkan",
    "secara langsung", "langsung mendatangi", "dengan mendatangi",
    "berawal dari", "bermula dari", "diawali dari", "diawali",
]

METHOD_VERB_PATTERN = re.compile(
    r"\bdengan\s+(?:meng|mem|men|meny|me|di|ber|ter|pe)[a-z]+\b", re.IGNORECASE
)

HOW_FALLBACK_CONNECTORS = ["usai", "setelah"]

# =============================================================================
# 4. KATA KERJA PELAPORAN (reporting verbs)
# =============================================================================
REPORTING_VERBS = [
    "kata", "ujar", "ungkap", "jelas", "tutur", "tegas", "papar",
    "terang", "imbuh", "tambah", "lanjut", "kata dia", "menurut",
    "tuturnya", "katanya", "ujarnya", "jelasnya", "tegasnya",
    "menyebut", "mengatakan", "menyatakan", "menuturkan",
]

# =============================================================================
# 5. GELAR / JABATAN
# =============================================================================
TITLE_PREFIXES = [
    "Presiden", "Wakil Presiden", "Menteri", "Wakil Menteri", "Mendag",
    "Menko", "Menkeu", "Menkumham", "Mendagri", "Menhan", "Gubernur",
    "Wali Kota", "Walikota", "Bupati", "Wakil Bupati", "Camat",
    "Komisaris", "Direktur", "Direktur Utama", "Kepala", "Ketua",
    "Wakil Ketua", "Anggota DPR", "Anggota DPRD", "Jaksa", "Hakim",
    "Polisi", "Kapolri", "Kapolda", "Kapolres", "Brigjen", "Irjen",
    "Kombes", "AKP", "Mayor", "Kolonel", "Jenderal", "Letnan",
    "Dokter", "Profesor", "Prof", "Dr", "Sekretaris", "Deputi",
    "Mantan", "Eks", "Tersangka", "Terdakwa", "Saksi",
    "Kabid", "Kasatgas", "Kasatreskrim", "Pimpinan",
]

# Lembaga/instansi dikenal
KNOWN_ORGS = [
    "KPK", "Polri", "TNI", "DPR", "MPR", "MK", "MA", "BPK", "BPOM",
    "KPU", "Bawaslu", "OJK", "BI", "Bank Indonesia", "Kemenkeu",
    "Kemendagri", "Kemenkes", "Kemendikbud", "Kementerian Keuangan",
    "PPATK", "Bareskrim", "Kejaksaan Agung", "Kejagung", "Pemprov",
    "Pemkot", "Pemkab", "BNPB", "BMKG", "BPBD",
]

# Kata/pola yang PASTI bukan lokasi geografis (untuk filter WHERE)
NOT_LOCATION_PATTERNS = [
    # Perusahaan / institusi (awalan umum)
    r"\bPT\b", r"\bCV\b", r"\bTbk\b", r"\bPersero\b",
    r"\bKPK\b", r"\bBUMN\b", r"\bDPR\b", r"\bMPR\b",
    r"\bKemenkeu\b", r"\bKementerian\b", r"\bKejaksaan\b",
    r"\bBareskrim\b", r"\bMahkamah\b", r"\bPengadilan\b",
    r"\bBank\s+\w+", r"\bUniversitas\b", r"\bInstitut\b",
    r"\bSekolah\b", r"\bRumah\s+Sakit\b",
    # Nama-nama media
    r"\bTempo\b", r"\bKompas\b", r"\bDetik\b",
    # Kata-kata yang sering muncul dalam nama organisasi tapi bukan lokasi
    r"\bBursa\b", r"\bInformasi\b", r"\bKeterbukaan\b",
]

# =============================================================================
# 6. POLA DATELINE
# =============================================================================
DATELINE_PATTERN = re.compile(
    r"^([A-Za-z\.]{2,25})\s*,\s*([A-Za-z\.\s]{2,25})\s*[-–]\s*"
)
DATELINE_SIMPLE_PATTERN = re.compile(r"^([A-Z][\w\.\s]{2,25})\s*[-–]\s*")

# =============================================================================
# 7. ARTEFAK NAVIGASI/UI SCRAPING
# =============================================================================
SCRAPING_ARTIFACTS = [
    "baca juga", "gambas video", "gambas:video",
    "scroll untuk melanjutkan", "scroll untuk lanjutkan",
    "advertisement", "lihat juga", "simak juga",
    "pilihan editor", "ikuti berita", "trending",
    "klik di sini", "klikdi sini",
]

# =============================================================================
# 8. BLACKLIST TAMBAHAN untuk WHERE (non-lokasi geografis)
# =============================================================================
WHERE_EXACT_BLACKLIST = {
    # Institusi hukum dan pemerintah
    "komisi pemberantasan korupsi", "mahkamah agung", "mahkamah konstitusi",
    "pengadilan negeri", "pengadilan tinggi", "kejaksaan agung",
    "kementerian keuangan", "kementerian hukum", "kemenkumham", "kemenkeu",
    "kementerian dalam negeri", "kemendagri",
    "badan pemeriksa keuangan", "badan intelijen negara",
    "dewan perwakilan rakyat", "majelis permusyawaratan rakyat",
    # Media
    "tempo co", "detik com", "kompas com", "suara com", "cnbc indonesia",
    # Frasa non-lokasi
    "paripurna", "sidang paripurna", "rapat paripurna",
    "corporate banking", "investment banking", "retail banking",
    # Nama yang sering jadi false positive
    "keterbukaan informasi", "bursa efek", "bursa efek indonesia",
}
