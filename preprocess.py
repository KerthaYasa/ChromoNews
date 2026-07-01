# preprocess.py
# ======================================================
# PREPROCESS DATASET UNTUK BM25 & SEMANTIC SEARCH
# ======================================================

import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory,
)

print("Menyiapkan Sastrawi...")

stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

# Aktifkan progress bar untuk pandas
tqdm.pandas()


# ======================================================
# BM25 PREPROCESSING
# ======================================================
def preprocess_for_bm25(text):
    """
    Preprocessing untuk BM25:
    - lowercase
    - hapus karakter selain huruf/angka
    - stopword removal
    - stemming
    """

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = stopword.remove(text)
    text = stemmer.stem(text)

    return text


# ======================================================
# SEMANTIC PREPROCESSING
# ======================================================
def preprocess_for_semantic(text):
    """
    Semantic Search:
    hanya normalize whitespace
    """

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    INPUT_FILE = BASE_DIR / "cleaned_news_sample.csv"
    OUTPUT_FILE = BASE_DIR / "preprocessed_news_sample.csv"

    print("=" * 70)
    print("MEMPROSES DATASET (BM25 + SEMANTIC SEARCH)")
    print("=" * 70)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    print("\n[1/3] Memuat dataset...")

    print("Lokasi dataset:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"✓ {len(df):,} artikel dimuat")

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    print("\n[2/3] Preprocessing BM25")
    print("Lowercase → Cleaning → Stopword Removal → Stemming")

    start = time.time()

    df["processed_content"] = df["content"].progress_apply(
        preprocess_for_bm25
    )

    end = time.time()

    print(f"✓ BM25 selesai ({end-start:.2f} detik)")

    # --------------------------------------------------
    # SEMANTIC
    # --------------------------------------------------

    print("\n[3/3] Preprocessing Semantic Search")
    print("Normalize whitespace")

    start = time.time()

    df["content_semantic"] = df["content"].progress_apply(
        preprocess_for_semantic
    )

    end = time.time()

    print(f"✓ Semantic selesai ({end-start:.2f} detik)")

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    print("\nMenyimpan dataset...")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✓ Dataset disimpan:\n{OUTPUT_FILE}")

    # --------------------------------------------------
    # PREVIEW
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PREVIEW HASIL")
    print("=" * 70)

    print("\nContent Asli")
    print("-" * 70)
    print(df["content"].iloc[0][:200])

    print("\nContent Semantic")
    print("-" * 70)
    print(df["content_semantic"].iloc[0][:200])

    print("\nContent BM25")
    print("-" * 70)
    print(df["processed_content"].iloc[0][:200])

    print("\n" + "=" * 70)
    print("SELESAI")
    print("=" * 70)

    print(f"Jumlah artikel        : {len(df):,}")
    print(f"Output               : {OUTPUT_FILE}")

    print("\nKolom dataset:")
    for col in df.columns:
        print(f" - {col}")

    print("\nSelanjutnya jalankan:")
    print("python run_eval.py")