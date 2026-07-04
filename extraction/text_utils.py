import re
from typing import List

from extraction.patterns import SCRAPING_ARTIFACTS

# Singkatan umum Bahasa Indonesia yang diikuti titik tapi BUKAN akhir kalimat.
# Sengaja daftar kecil & konservatif — lebih baik under-protect drpd
# over-protect (over-protect bisa menggabungkan 2 kalimat yang harusnya terpisah).
_ABBREVIATIONS = [
    "Dr", "Prof", "Ir", "Drs", "Dra", "H", "Hj", "Sdr", "Sdri",
    "Kol", "Jend", "Mayjen", "Brigjen", "Irjen", "Kombes", "AKP",
    "No", "Jl", "Kec", "Kab", "Kel", "RT", "RW", "dll", "dsb", "tsb", "yth",
]
_ABBREV_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in _ABBREVIATIONS) + r')\.',
    re.IGNORECASE,
)
_ABBREV_PLACEHOLDER = "\x00DOT\x00"


def split_sentences(text: str) -> List[str]:
    """
    Pecah teks menjadi kalimat dengan mempertimbangkan singkatan dan
    batas akhir kalimat.

    Strategi: titik setelah singkatan dikenal (Dr., No., dll.) disamarkan
    sementara dengan placeholder supaya tidak ikut dipecah oleh regex
    split utama, lalu dikembalikan lagi setelah split selesai.
    """
    # Samarkan titik singkatan agar tidak dianggap akhir kalimat
    protected = _ABBREV_PATTERN.sub(lambda m: m.group(1) + _ABBREV_PLACEHOLDER, text)

    protected = re.sub(r'([a-z0-9"\')\]])(\.)([A-Z])', r'\1.\n\3', protected)
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", protected) if s.strip()]

    # Kembalikan titik singkatan yang disamarkan tadi
    sentences = [s.replace(_ABBREV_PLACEHOLDER, ".") for s in raw_sentences]
    return sentences


def is_scraping_artifact(text: str) -> bool:
    """Deteksi teks yang berupa artefak scraping (misal "Baca juga:", "Halaman selanjutnya")."""
    for pattern in SCRAPING_ARTIFACTS:
        if pattern.lower() in text.lower():
            return True
    return False