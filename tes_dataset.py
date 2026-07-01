import pandas as pd
from pathlib import Path

# ======================================================
# CLEAN DATASET UNTUK CHROMONEWS
# ======================================================

# Folder project (otomatis)
BASE_DIR = Path(__file__).resolve().parent

# File input & output
INPUT_FILE = BASE_DIR / "data" / "news.csv"
OUTPUT_FILE = BASE_DIR / "cleaned_news_sample.csv"

print("=" * 60)
print("MEMBUAT DATASET CHROMONEWS")
print("=" * 60)

# ======================================================
# 1. Load Dataset
# ======================================================

print("\n[1/5] Memuat dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Jumlah data awal : {len(df):,}")

# ======================================================
# 2. Ambil kolom yang dibutuhkan
# ======================================================

print("\n[2/5] Memilih kolom...")

df = df[["Judul", "Waktu", "Content"]].copy()

# ======================================================
# 3. Cleaning
# ======================================================

print("\n[3/5] Cleaning data...")

df = df.dropna(subset=["Judul", "Waktu", "Content"])
df = df.reset_index(drop=True)

print(f"Jumlah setelah cleaning : {len(df):,}")

# ======================================================
# 4. Format sesuai ChromoNews
# ======================================================

print("\n[4/5] Menyesuaikan format...")

df.rename(columns={
    "Judul": "title",
    "Waktu": "date",
    "Content": "content"
}, inplace=True)

# Tambahkan ID
df.insert(0, "id", range(1, len(df) + 1))

# Tambahkan summary kosong
df["summary"] = ""

# ======================================================
# 5. Sampling 5000 berita
# ======================================================

print("\n[5/5] Sampling dataset...")

SAMPLE_SIZE = min(5000, len(df))

df_sample = (
    df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )
    .reset_index(drop=True)
)

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