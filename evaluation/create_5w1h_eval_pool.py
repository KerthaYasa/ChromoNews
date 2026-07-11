"""
create_5w1h_eval_pool.py — Step 1: Sampling 50 berita + jalankan ekstraksi
============================================================================
Sesuai catatan dosen (Analisis 5W+1H):
    "Lakukan ekstraksi untuk seluruh komponen 5W+1H ... Gunakan 50 berita
    sebagai sampel."

Skrip ini:
1. Ambil 50 berita SECARA ACAK (random sample, seed tetap agar reproducible)
   dari dataset CSV yang sudah ada (default: preprocessed_news_sample.csv).
2. Jalankan extract_5w1h() (pipeline lengkap: who/what/when/where/why/how)
   pada setiap berita.
3. Simpan hasilnya sebagai template CSV (5w1h_eval_template.csv) dengan
   kolom "*_pred" (hasil sistem, sudah terisi) dan "*_truth" (KOSONG --
   wajib diisi manual oleh anotator/manusia sebagai ground truth) untuk
   masing-masing dari 6 komponen: who, what, when, where, why, how.

Cara pakai:
    python evaluation/create_5w1h_eval_pool.py

Setelah itu:
    1. Buka 5w1h_eval_template.csv (mis. di Excel/Google Sheets).
    2. Baca kolom "content", isi kolom "*_truth" untuk who/what/when/where/
       why/how secara manual berdasarkan isi artikel (ground truth).
       - Untuk who/where/when yang bisa multi-entitas, pisahkan dengan ";"
         (konsisten dengan format evaluate_who_where.py yang sudah ada).
       - Kalau memang tidak disebutkan di artikel, isi:
         "Tidak disebutkan dalam artikel"
    3. Simpan sebagai 5w1h_eval_ground_truth.csv (kolom sama, "*_truth" sudah
       terisi).
    4. Jalankan evaluation/compute_mape_5w1h.py untuk hitung skor MAPE.
"""

import os
import sys
import random
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.rule_based_5w1h import extract_5w1h

SEED = 42
N_SAMPLE = 50
DEFAULT_DATA_PATH = "preprocessed_news_sample.csv"
OUTPUT_PATH = "5w1h_eval_template.csv"


def _join_list(value):
    """who/when/where hasil extract_5w1h berupa List[str] -> gabung jadi 1 string."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return value


def build_eval_template(data_path=DEFAULT_DATA_PATH, n_sample=N_SAMPLE, seed=SEED, output_path=OUTPUT_PATH):
    print("=" * 70)
    print(f"STEP 1: SAMPLING {n_sample} BERITA + EKSTRAKSI 5W1H")
    print("=" * 70)

    print(f"\n📄 Memuat dataset dari '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"   Total berita di dataset: {len(df)}")

    if len(df) < n_sample:
        print(f"⚠ Dataset lebih kecil dari {n_sample}, memakai semua ({len(df)}) berita.")
        n_sample = len(df)

    random.seed(seed)
    sampled_indices = random.sample(range(len(df)), n_sample)
    df_sample = df.iloc[sampled_indices].reset_index(drop=True)
    print(f"   Diambil {n_sample} berita secara acak (seed={seed}, reproducible).")

    rows = []
    print(f"\n🔎 Menjalankan ekstraksi 5W1H untuk {n_sample} berita...")
    for i, row in df_sample.iterrows():
        article = {
            "title": row.get("title", ""),
            "content": row.get("content", ""),
            "date": row.get("date", ""),
        }
        result = extract_5w1h(article)

        rows.append({
            "id": row.get("id", i),
            "title": article["title"],
            "content": article["content"],

            "what_pred": result["what"],
            "what_truth": "",

            "who_pred": _join_list(result["who"]),
            "who_truth": "",

            "when_pred": _join_list(result["when"]),
            "when_truth": "",

            "where_pred": _join_list(result["where"]),
            "where_truth": "",

            "why_pred": result["why"],
            "why_truth": "",

            "how_pred": result["how"],
            "how_truth": "",
        })

        if (i + 1) % 10 == 0:
            print(f"   ...{i + 1}/{n_sample} selesai")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✓ Template evaluasi disimpan ke '{output_path}'")
    print(f"  Kolom '*_pred' sudah terisi otomatis (hasil sistem).")
    print(f"  Kolom '*_truth' MASIH KOSONG -- isi manual sebagai ground truth,")
    print(f"  lalu simpan sebagai '5w1h_eval_ground_truth.csv' dan jalankan")
    print(f"  evaluation/compute_mape_5w1h.py.")

    return out_df


if __name__ == "__main__":
    build_eval_template()
