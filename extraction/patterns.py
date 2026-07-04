"""
patterns.py
============
Kumpulan pattern (regex) dan kamus kata kunci untuk ekstraksi 5W1H.
REVISI v6: Update berdasarkan analisis dataset preprocessed_news_sample.csv
           (3.000 artikel) — lihat linguistic_analysis_5w1h.md section 6a.
           - Tambah marker WHY baru: gegara, gara-gara, alasannya, penyebab, faktor, soalnya
           - Tambah marker HOW baru: dengan skema, dengan metode, yakni dengan, yaitu dengan
           - Tambah reporting verbs baru: pungkas, sambung, bebernya, tutupnya
           - Tambah disambiguation rule untuk marker ambigu: untuk, secara, dengan
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

# Pola tambahan: "pukul HH.MM" / "pukul HH:MM" — cukup sering (516x di sampel)
DATE_PATTERN_PUKUL = re.compile(
    r"\bpukul\s+\d{1,2}[.:]\d{2}(?:\s*(?:WIB|WITA|WIT))?\b",
    re.IGNORECASE,
)

# CATATAN: DATE_PATTERN_RELATIVE DIHAPUS dari DATE_PATTERNS — "hari ini", "kemarin", "besok"
# tidak akan dimasukkan ke hasil WHEN karena tidak informatif (tidak absolut).
# Akan digantikan oleh metadata_date jika tersedia.
DATE_PATTERN_RELATIVE = re.compile(
    r"(?i)\b(?:hari ini|kemarin|besok|tadi malam|malam tadi|sore tadi|pagi tadi|dini hari|saat ini|tahun ini|tahun lalu)\b"
)

DATE_PATTERNS = [
    DATE_PATTERN_WITH_DAY,
    DATE_PATTERN_FULL,
    DATE_PATTERN_NUMERIC,
    DATE_PATTERN_SHORTHAND,
    DATE_PATTERN_PUKUL,
    # DATE_PATTERN_RELATIVE SENGAJA TIDAK DIMASUKKAN
]


# =============================================================================
# 2. KATA PENGHUBUNG KAUSAL (untuk WHY)
# =============================================================================
# Tier 1: frekuensi tinggi & ambiguitas rendah -> confidence tinggi
CAUSAL_CONNECTORS_TIER1 = [
    "karena", "dikarenakan",
    "sehingga",
    "akibat dari", "akibat",
    "lantaran",
    "menyebabkan", "mengakibatkan",
    "disebabkan oleh", "disebabkan karena", "disebabkan",
    "diakibatkan oleh", "diakibatkan",
    "atas dasar itulah", "atas dasar itu", "atas dasar",
]

# Tier 2: frekuensi lebih rendah / perlu konteks, tapi cukup reliabel
CAUSAL_CONNECTORS_TIER2 = [
    "imbas dari", "imbas",
    "dipicu oleh", "dipicu", "memicu",
    "buntut dari", "buntut",
    "diduga karena",
    "gara-gara", "gegara",  # BARU — ditemukan 31x di sampel, belum ada sebelumnya
    "penyebab",              # BARU — 129x, biasanya nominal ("penyebab kebakaran")
    "faktor",                 # BARU — 101x, biasanya nominal ("faktor utama")
    "alasannya",              # BARU — 43x, penanda motif eksplisit
]

# "sebab" sengaja dipisah: ambigu antara nomina ("alasan") dan konektor.
# Gunakan hanya jika TIDAK diikuti langsung oleh kata benda umum seperti
# "itu", "tersebut", "ini" tanpa verba setelahnya (lihat disambiguation di bawah).
CAUSAL_CONNECTORS_AMBIGUOUS = ["sebab", "dampak"]

# Gabungan untuk kompatibilitas mundur dengan kode lama yang memakai CAUSAL_CONNECTORS
CAUSAL_CONNECTORS = CAUSAL_CONNECTORS_TIER1 + CAUSAL_CONNECTORS_TIER2

INTER_SENTENCE_CAUSAL = [
    "pasalnya", "oleh karena itu", "sebab itu", "karenanya", "alhasil",
    "hal itu disebabkan", "hal tersebut disebabkan",
    "soalnya",  # BARU — 14x, informal tapi reliabel sebagai penanda kausal awal kalimat
]

# =============================================================================
# 2b. KATA PENGHUBUNG TUJUAN/MOTIVASI (untuk WHY)
# =============================================================================
PURPOSE_CONNECTORS = [
    "dengan tujuan", "bertujuan untuk", "bertujuan",
    "dalam rangka", "demi", "guna",
    "agar", "supaya", "berharap", "diharapkan",
]

# "untuk" SENGAJA TIDAK dimasukkan langsung ke PURPOSE_CONNECTORS.
# Frekuensi 8.344x di sampel, tapi hanya ~15-20% benar-benar menandai WHY;
# sisanya preposisi biasa ("rumah untuk dijual"). Gunakan UNTUK_PURPOSE_PATTERN
# di bagian disambiguation rules di bawah untuk memvalidasi konteksnya.


# =============================================================================
# 3. KATA PENGHUBUNG CARA/MODUS (untuk HOW)
# =============================================================================
METHOD_CONNECTORS = [
    "dengan cara", "dengan modus", "modus operandi", "modusnya",
    "melalui", "lewat", "menggunakan", "dengan menggunakan",
    "dengan memanfaatkan", "memanfaatkan",
    "secara langsung", "langsung mendatangi", "dengan mendatangi",
    "berawal dari", "bermula dari", "diawali dari", "diawali",
    # --- BARU (dari analisis dataset, section 6a) ---
    "dengan skema",     # 10x
    "dengan metode",    # 14x
    "yakni dengan",     # 14x
    "yaitu dengan",     # 9x
    "via",               # 40x — umum untuk saluran/media ("via telepon", "via WhatsApp")
]

METHOD_VERB_PATTERN = re.compile(
    r"\bdengan\s+(?:meng|mem|men|meny|me|di|ber|ter|pe)[a-z]+\b", re.IGNORECASE
)

HOW_FALLBACK_CONNECTORS = ["usai", "setelah"]

# "secara" TIDAK dimasukkan sebagai method connector langsung karena sangat ambigu
# (1.567x di sampel, sering hanya adverbia: "secara resmi", "secara bertahap").
# Gunakan SECARA_METHOD_PATTERN di bawah untuk memvalidasi apakah benar menjelaskan
# mekanisme, bukan sekadar adverbia formalitas.


# =============================================================================
# 4. KATA KERJA PELAPORAN (reporting verbs)
# =============================================================================
REPORTING_VERBS = [
    "kata", "ujar", "ungkap", "jelas", "tutur", "tegas", "papar",
    "terang", "imbuh", "tambah", "lanjut", "kata dia", "menurut",
    "tuturnya", "katanya", "ujarnya", "jelasnya", "tegasnya",
    "menyebut", "mengatakan", "menyatakan", "menuturkan",
    "tandas", "tandasnya", "pungkas",  # "tandas"/"tandasnya" sudah lazim; "pungkas" BARU (60x)
    "sambung", "sambungnya",           # BARU — 44x, menandai kutipan lanjutan
    "bebernya", "membeberkan",         # BARU — 17x
    "tutupnya",                          # BARU — 34x, menandai kutipan penutup
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
    "Juru Bicara",  # BARU — 85x, cukup umum ("Juru Bicara KPK Ali Fikri")
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
    # Nomina lokasi generik yang sering ambigu dengan organisasi (mis. "kantor KPK")
    # -> ditangani lewat WHERE_AMBIGUOUS_NOUNS di bawah, bukan hard-blacklist,
    #    karena "kantor" & "gedung" TETAP valid untuk lokasi fisik pada banyak kasus.
]

# Nomina lokasi yang sering mengikuti preposisi "di/ke/dari" tapi berpotensi ambigu
# (butuh cek apakah diikuti nama organisasi atau nama tempat asli).
WHERE_AMBIGUOUS_NOUNS = ["kantor", "gedung", "lokasi"]

# Nomina lokasi yang cukup reliabel sebagai penanda WHERE eksplisit
WHERE_LOCATION_NOUNS = ["daerah", "wilayah", "kawasan", "kompleks", "setempat"]


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


# =============================================================================
# 9. DISAMBIGUATION RULES — untuk marker frekuensi tinggi tapi ambigu
# =============================================================================
# Ketiga marker ini TIDAK dipakai sebagai trigger langsung, karena mayoritas
# kemunculannya di dataset BUKAN sinyal 5W1H yang valid (lihat catatan
# ambiguitas di linguistic_analysis_5w1h.md section 1a/2a). Gunakan pattern
# di bawah untuk memvalidasi konteks sebelum menandai sebagai WHY/HOW.

# "untuk" -> valid sebagai WHY (tujuan) hanya jika diikuti verba (imbuhan
# me-/di-/ber-/ter-/per-), bukan langsung diikuti nomina/objek biasa.
# Contoh valid: "... dibentuk untuk mengusut ..." (untuk + verba)
# Contoh TIDAK valid: "rumah untuk dijual", "dana untuk masyarakat" (nomina)
UNTUK_PURPOSE_PATTERN = re.compile(
    r"\buntuk\s+(?:meng|mem|men|meny|me|di|ber|ter|per)[a-z]+\b", re.IGNORECASE
)

# "secara" -> valid sebagai HOW (mekanisme) hanya jika diikuti kata yang
# menjelaskan cara/metode konkret (bukan adverbia formalitas generik seperti
# "resmi", "bertahap", "keseluruhan"). Whitelist kata setelah "secara" yang
# benar-benar menjelaskan mekanisme:
SECARA_METHOD_WHITELIST = [
    "online", "daring", "manual", "elektronik", "digital", "tunai",
    "bertahap", "paksa", "sembunyi-sembunyi", "diam-diam", "terang-terangan",
    "langsung", "tatap muka", "virtual", "kolektif", "mandiri",
]
SECARA_METHOD_PATTERN = re.compile(
    r"\bsecara\s+(" + "|".join(SECARA_METHOD_WHITELIST) + r")\b",
    re.IGNORECASE,
)
# Kata setelah "secara" yang menandakan BUKAN How (adverbia formalitas/derajat,
# skip jika match ini) — dipakai sebagai negative filter tambahan.
SECARA_NON_METHOD_WORDS = [
    "resmi", "keseluruhan", "umum", "khusus", "pasti", "otomatis",
    "bersamaan", "bergantian", "signifikan", "drastis", "perlahan",
]

# "dengan" tanpa verba setelahnya -> kemungkinan besar WHO (kebersamaan),
# bukan HOW. Contoh: "Jokowi dengan Prabowo" vs "dengan menggunakan kunci T".
# Gunakan METHOD_VERB_PATTERN (poin 3) sebagai validator utama; jika "dengan"
# diikuti langsung oleh nama orang/gelar (lihat TITLE_PREFIXES) atau nomina
# tanpa imbuhan verba, klasifikasikan sebagai non-HOW.
DENGAN_COMPANION_PATTERN = re.compile(
    r"\bdengan\s+(?:" + "|".join(re.escape(t) for t in TITLE_PREFIXES) + r")\b",
    re.IGNORECASE,
)


def is_valid_why_untuk(text_snippet: str) -> bool:
    """Cek apakah kemunculan 'untuk' dalam snippet menandakan WHY (tujuan)."""
    return bool(UNTUK_PURPOSE_PATTERN.search(text_snippet))


def is_valid_how_secara(text_snippet: str) -> bool:
    """Cek apakah kemunculan 'secara' dalam snippet menandakan HOW (mekanisme)."""
    if SECARA_METHOD_PATTERN.search(text_snippet):
        return True
    return False


def is_valid_how_dengan(text_snippet: str) -> bool:
    """Cek apakah 'dengan' menandakan HOW (bukan kebersamaan/WHO)."""
    if DENGAN_COMPANION_PATTERN.search(text_snippet):
        return False
    return bool(METHOD_VERB_PATTERN.search(text_snippet))


def is_valid_why_sebab(text_snippet: str) -> bool:
    """Cek apakah 'sebab' berfungsi sebagai konektor kausal, bukan nomina 'alasan'."""
    # Jika langsung diikuti "itu"/"tersebut"/"ini" TANPA verba setelahnya,
    # kemungkinan besar nomina ("sebab itu tidak jelas"), bukan konektor.
    if re.search(r"\bsebab\s+(itu|tersebut|ini)\b(?!\s+(?:meng|mem|men|meny|me|di|ber|ter))",
                 text_snippet, re.IGNORECASE):
        return False
    return True