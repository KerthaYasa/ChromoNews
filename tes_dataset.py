import re
import pandas as pd
from pathlib import Path

# ======================================================
# CLEAN DATASET UNTUK CHROMONEWS
# ======================================================

# Folder project (otomatis)
BASE_DIR = Path(__file__).resolve().parent

# File input & output
INPUT_FILE  = BASE_DIR / "data" / "data.csv"
OUTPUT_FILE = BASE_DIR / "cleaned_news_sample.csv"

print("=" * 60)
print("MEMBUAT DATASET CHROMONEWS")
print("=" * 60)

# ======================================================
# 1. Load Dataset
# ======================================================

print("\n[1/8] Memuat dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Jumlah data awal : {len(df):,}")

# ======================================================
# 2. Ambil kolom yang dibutuhkan
# ======================================================

print("\n[2/8] Memilih kolom...")

df = df[["title", "date", "content"]].copy()

# ======================================================
# 3. Cleaning
# ======================================================

print("\n[3/8] Cleaning data...")

df = df.dropna(subset=["title", "date", "content"])
df = df.reset_index(drop=True)

print(f"Jumlah setelah cleaning : {len(df):,}")

# ======================================================
# 4. Filter Rentang Tanggal (hanya 2023)
# ======================================================

print("\n[4/8] Filter rentang tanggal...")

df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

sebelum = len(df)
df = df[df["date"].dt.year == 2023].reset_index(drop=True)

print(f"Artikel di luar 2023 dibuang  : {sebelum - len(df)}")
print(f"Jumlah setelah filter tanggal : {len(df):,}")

# ======================================================
# 5. Hapus Duplikat
# ======================================================

print("\n[5/8] Menghapus duplikat...")

sebelum = len(df)

df = df.drop_duplicates(subset=["content"], keep="first")
df = df.drop_duplicates(subset=["title", "date"], keep="first")
df = df.reset_index(drop=True)

print(f"Duplikat dibuang     : {sebelum - len(df)}")
print(f"Jumlah setelah dedup : {len(df):,}")

# ======================================================
# 6. Filter Boilerplate
# ======================================================

print("\n[6/8] Membersihkan boilerplate...")

# -- Dateline sumber media (di awal ATAU di tengah konten) --
DATELINE = re.compile(
    r"(?:^|\s)(?:"
    r"TEMPO\.CO\s*,\s*\w+\s*[-]\s*"
    r"|Suara\.com\s*[-]\s*"
    r"|(?:\w+\s*,\s*)?CNBC\s*Indonesia\s*[-]\s*"
    r"|(?:\w+\s*,\s*)?CNN\s*Indonesia\s*[-]\s*"
    r"|JAKARTA\s*[-]\s*"
    r"|SURABAYA\s*[-]\s*"
    r"|BANDUNG\s*[-]\s*"
    r"|MEDAN\s*[-]\s*"
    r"|BALI\s*[-]\s*"
    r"|YOGYAKARTA\s*[-]\s*"
    r"|Okezone\.com\s*[-]\s*"
    r"|Kumparan\.com\s*[-]\s*"
    r"|JawaPos\.com\s*[-]\s*"
    r")",
    re.IGNORECASE,
)

# -- Pola lain yang ditemukan di analisis data.csv --
BACA_JUGA   = re.compile(r"(?:Baca\s+[Jj]uga|BACA\s+JUGA)\s*:\s*[^\n]*", re.IGNORECASE)
GAMBAS      = re.compile(r"\[Gambas:[^\]]*\]", re.IGNORECASE)
EDITOR      = re.compile(r"(?:Editor|Penulis|Reporter|Redaktur)\s*:\s*[^\n]*", re.IGNORECASE)
INITIALS    = re.compile(r"\s*\([A-Z]{2,4}\)\s*$")
NAVIGASI    = re.compile(
    r"klik\s+di\s+sini[^.]*[.]?"
    r"|scroll\s+ke\s+bawah[^.]*[.]?"
    r"|simak\s+breaking\s+news[^.]*[.]?"
    r"|ikuti\s+kami\s+di[^.]*[.]?"
    r"|follow\s+us[^.]*[.]?"
    r"|lihat\s+juga\s*:[^\n]*"
    r"|tonton\s+video[^\n]*"
    r"|video\s+pilihan[^\n]*",
    re.IGNORECASE,
)
URL_INLINE  = re.compile(r"https?://\S+", re.IGNORECASE)
INFO_BISNIS = re.compile(r"^INFO\s+BISNIS\s*[-]?\s*", re.IGNORECASE)
WHITESPACE  = re.compile(r"\s{2,}")


def bersihkan_boilerplate(text):
    if not isinstance(text, str):
        return ""

    text = INFO_BISNIS.sub("", text)   # prefix advertorial
    text = DATELINE.sub(" ", text)     # dateline sumber media (di mana saja)
    text = BACA_JUGA.sub("", text)     # link rekomendasi artikel
    text = GAMBAS.sub("", text)        # embed widget media
    text = EDITOR.sub("", text)        # label editor/penulis
    text = NAVIGASI.sub("", text)      # teks navigasi / CTA
    text = URL_INLINE.sub("", text)    # URL inline
    text = INITIALS.sub("", text)      # inisial reporter di akhir
    text = WHITESPACE.sub(" ", text)   # normalisasi whitespace

    return text.strip()


sebelum = len(df)

df["content"] = df["content"].apply(bersihkan_boilerplate)

# Buang artikel yang terlalu pendek setelah cleaning
MIN_LEN = 150
df = df[df["content"].str.len() >= MIN_LEN].reset_index(drop=True)

print(f"Artikel dibuang (konten < {MIN_LEN} char) : {sebelum - len(df)}")
print(f"Jumlah setelah filter boilerplate        : {len(df):,}")

# ======================================================
# 7. Format sesuai ChromoNews
# ======================================================

print("\n[7/8] Menyesuaikan format...")

# Kembalikan date ke format string bersih
df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Tambahkan ID
df.insert(0, "id", range(1, len(df) + 1))

# Tambahkan summary kosong (None agar konsisten null semua)
df["summary"] = None

# ======================================================
# 8. Sampling 15000 berita
# ======================================================

print("\n[8/8] Sampling dataset...")

SAMPLE_SIZE = min(15000, len(df))

df_sample = (
    df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )
    .reset_index(drop=True)
)

df_sample["id"] = range(1, len(df_sample) + 1)

print(f"Jumlah sample : {len(df_sample):,}")

# Simpan
df_sample.to_csv(OUTPUT_FILE, index=False)

# Cek hasil simpan
cek = pd.read_csv(OUTPUT_FILE)
print("Jumlah baris file yang disimpan:", len(cek))

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)

print(f"File berhasil disimpan di:\n{OUTPUT_FILE}")
print(f"Total berita: {len(df_sample):,}")
print("\nSelanjutnya jalankan:")
print("  python preprocess.py")
print("  python regenerate_embeddings.py")
