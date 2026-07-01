"""
regenerate_embeddings.py
========================
Script untuk regenerasi ulang corpus_embeddings.npy dari dataset yang ada.

Jalankan ini jika:
- Dataset (preprocessed_news_sample.csv) sudah diperbarui/ditambah
- File corpus_embeddings.npy terhapus atau corrupt
- Ingin me-refresh embedding setelah update model

Perintah:
    python regenerate_embeddings.py
"""

import pandas as pd
import os
from semantic_search import load_embedding_model, encode_corpus

CACHE_FILE = "corpus_embeddings.npy"
DATA_FILE = "preprocessed_news_sample.csv"


def main():
    print(f"Memuat dataset '{DATA_FILE}'...")
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: File '{DATA_FILE}' tidak ditemukan.")
        print("Pastikan preprocess.py sudah dijalankan terlebih dahulu.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Dataset dimuat: {len(df)} artikel")

    # Hapus cache lama
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"Cache lama '{CACHE_FILE}' dihapus.")

    print("\nMemuat model embedding...")
    model = load_embedding_model()

    print(f"\nMen-encode {len(df)} artikel...")
    print("(Ini butuh 1-3 menit tergantung spesifikasi mesin)")
    embeddings = encode_corpus(model, df['content'].tolist(), cache_path=CACHE_FILE)

    print(f"\n✓ Selesai! Embeddings tersimpan di '{CACHE_FILE}'")
    print(f"  Shape: {embeddings.shape}")
    print(f"  File size: {os.path.getsize(CACHE_FILE) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
