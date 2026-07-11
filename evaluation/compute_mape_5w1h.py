"""
compute_mape_5w1h.py — Step 2: Evaluasi MAPE hasil ekstraksi 5W1H
====================================================================
Sesuai catatan dosen:
    "Evaluasi hasil ekstraksi menggunakan MAPE."

MAPE (Mean Absolute Percentage Error) aslinya untuk data numerik. Karena
jawaban 5W1H berupa TEKS, kita adaptasi dengan cara yang lazim dipakai di
penelitian ekstraksi teks: skor kemiripan (BERTScore, seluruh kalimat --
extraction.text_similarity) antara jawaban SISTEM dan GROUND TRUTH dianggap
sebagai "nilai aktual" dengan nilai ideal 100% (kalau sistem persis sama
dengan ground truth, error = 0%). Rumus:

    error_i   = |100% - skor_kemiripan_i| / 100%   (dalam skala 0..1)
    MAPE_kat  = (1/n) * sum(error_i) * 100%

Skor kemiripan dihitung PER KOMPONEN (who/what/when/where/why/how):
    - what/why/how: bertscore_similarity(pred, truth) langsung (teks bebas).
    - who/when/where: bisa multi-entitas (dipisah ";"). Digabung dulu jadi
      1 string ("; ".join(...)) SEBELUM dibandingkan, supaya urutan/jumlah
      entitas yang berbeda tetap tertangkap sebagai penalti kemiripan
      (bukan cuma dicocokkan per-entitas seperti di evaluate_who_where.py
      yang pakai P/R/F1 -- MAPE di sini melihat kemiripan keseluruhan
      jawaban sebagai satu teks, sesuai permintaan "gunakan BERTScore").

Selain MAPE per kategori (kuantitatif), skrip ini juga mencetak 2 CONTOH
kasus (query/kasus uji) per kategori sebagai ilustrasi kualitatif di
laporan -- 1 kasus dengan skor kemiripan TERTINGGI dan 1 dengan skor
TERENDAH, supaya terlihat contoh sukses vs contoh gagal ekstraksi.

Cara pakai:
    python evaluation/compute_mape_5w1h.py 5w1h_eval_ground_truth.csv

Kalau argumen path tidak diberikan, default: 5w1h_eval_ground_truth.csv
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.text_similarity import bertscore_similarity

CATEGORIES = ["what", "who", "when", "where", "why", "how"]
NOT_FOUND = "Tidak disebutkan dalam artikel"
DEFAULT_GT_PATH = "5w1h_eval_ground_truth.csv"


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _similarity_score(pred: str, truth: str) -> float:
    """
    Skor kemiripan [0, 1] antara prediksi & ground truth untuk 1 komponen.
    Kasus khusus: kalau keduanya sama-sama "Tidak disebutkan dalam artikel"
    (atau salah satu kosong dan yang lain juga menandakan kosong), anggap
    kemiripan sempurna (1.0) -- sistem benar bilang "tidak ada".
    """
    pred, truth = _clean(pred), _clean(truth)

    pred_empty = (not pred) or (NOT_FOUND.lower() in pred.lower())
    truth_empty = (not truth) or (NOT_FOUND.lower() in truth.lower())

    if pred_empty and truth_empty:
        return 1.0
    if pred_empty or truth_empty:
        return 0.0  # salah satu bilang "tidak ada" tapi yang lain ada isinya -> mismatch total

    return bertscore_similarity(pred, truth)


def compute_mape(gt_path: str = DEFAULT_GT_PATH):
    print("=" * 70)
    print("STEP 2: EVALUASI MAPE HASIL EKSTRAKSI 5W1H")
    print("=" * 70)

    print(f"\n📄 Memuat ground truth dari '{gt_path}'...")
    df = pd.read_csv(gt_path)
    print(f"   Total baris: {len(df)}")

    missing_truth = []
    for cat in CATEGORIES:
        col = f"{cat}_truth"
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di file ground truth.")
        n_empty = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        if n_empty > 0:
            missing_truth.append((cat, n_empty))

    if missing_truth:
        print("\n⚠ PERINGATAN: ada baris ground truth yang masih kosong:")
        for cat, n in missing_truth:
            print(f"   - {cat}_truth: {n} baris kosong (akan tetap dihitung, "
                  f"dianggap 'Tidak disebutkan dalam artikel').")

    per_row_scores = {cat: [] for cat in CATEGORIES}
    examples = {cat: [] for cat in CATEGORIES}

    print(f"\n🔎 Menghitung skor kemiripan (BERTScore) per komponen...")
    for i, row in df.iterrows():
        for cat in CATEGORIES:
            pred = row.get(f"{cat}_pred", "")
            truth = row.get(f"{cat}_truth", "")
            score = _similarity_score(pred, truth)
            per_row_scores[cat].append(score)
            examples[cat].append({
                "id": row.get("id", i),
                "title": row.get("title", ""),
                "pred": _clean(pred),
                "truth": _clean(truth),
                "score": score,
            })
        if (i + 1) % 10 == 0:
            print(f"   ...{i + 1}/{len(df)} selesai")

    # --- Ringkasan MAPE per kategori ---
    print("\n" + "=" * 70)
    print("HASIL MAPE PER KATEGORI")
    print("=" * 70)
    summary = {}
    for cat in CATEGORIES:
        scores = per_row_scores[cat]
        n = len(scores)
        errors_pct = [abs(100.0 - (s * 100.0)) for s in scores]
        mape = sum(errors_pct) / n if n else 0.0
        avg_similarity = (sum(scores) / n) * 100 if n else 0.0
        summary[cat] = {"mape": mape, "avg_similarity_pct": avg_similarity, "n": n}
        print(f"  {cat.upper():6s} | Rata-rata kemiripan: {avg_similarity:6.2f}% "
              f"| MAPE: {mape:6.2f}% | n={n}")

    overall_mape = sum(v["mape"] for v in summary.values()) / len(summary)
    print(f"\n  {'RATA2':6s} | {'':6s}   {'':6s}   MAPE keseluruhan (6 kategori): "
          f"{overall_mape:.2f}%")

    # --- 2 contoh kasus per kategori (tertinggi & terendah) ---
    print("\n" + "=" * 70)
    print("CONTOH KASUS PER KATEGORI (2 query/kasus uji -- terbaik & terburuk)")
    print("=" * 70)
    for cat in CATEGORIES:
        ex_sorted = sorted(examples[cat], key=lambda e: -e["score"])
        best = ex_sorted[0]
        worst = ex_sorted[-1]
        print(f"\n[{cat.upper()}]")
        print(f"  Contoh TERBAIK  (id={best['id']}, skor={best['score']*100:.1f}%):")
        print(f"    Judul : {best['title']}")
        print(f"    Pred  : {best['pred']}")
        print(f"    Truth : {best['truth']}")
        print(f"  Contoh TERBURUK (id={worst['id']}, skor={worst['score']*100:.1f}%):")
        print(f"    Judul : {worst['title']}")
        print(f"    Pred  : {worst['pred']}")
        print(f"    Truth : {worst['truth']}")

    # --- Simpan detail per baris untuk lampiran laporan ---
    detail_rows = []
    for i in range(len(df)):
        row_detail = {"id": df.iloc[i].get("id", i), "title": df.iloc[i].get("title", "")}
        for cat in CATEGORIES:
            row_detail[f"{cat}_similarity_pct"] = round(per_row_scores[cat][i] * 100, 2)
        detail_rows.append(row_detail)
    detail_path = os.path.splitext(gt_path)[0] + "_mape_detail.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Detail skor per baris disimpan ke '{detail_path}'")

    return summary


if __name__ == "__main__":
    gt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GT_PATH
    if not os.path.exists(gt_path):
        print(f"❌ File '{gt_path}' tidak ditemukan.")
        print(f"   Jalankan dulu evaluation/create_5w1h_eval_pool.py, isi kolom")
        print(f"   '*_truth' secara manual, simpan sebagai '{gt_path}', baru jalankan skrip ini.")
        sys.exit(1)
    compute_mape(gt_path)
