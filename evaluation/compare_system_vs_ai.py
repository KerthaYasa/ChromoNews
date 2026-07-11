"""
compare_system_vs_ai.py — Validasi AI, Step B: Sistem vs AI
====================================================================
Sesuai catatan dosen (Validasi dengan AI):
    "Bandingkan hasil sistem dengan hasil AI menggunakan Word Embedding
    (BERTScore) atau Cosine Similarity. Semakin tinggi nilai kemiripan,
    semakin relevan hasil ekstraksi sistem."

Berbeda dari compute_mape_5w1h.py (yang mengukur error terhadap ground
truth MANUSIA), skrip ini murni melaporkan SKOR KEMIRIPAN sistem vs AI
per kategori (tanpa MAPE) -- sesuai instruksi di atas, karena di sini AI
bukan "kebenaran mutlak", jadi yang dilaporkan adalah derajat relevansi/
konsistensi, bukan "error".

Cara pakai:
    python evaluation/compare_system_vs_ai.py 5w1h_eval_ai.csv

Kalau argumen path tidak diberikan, default: 5w1h_eval_ai.csv (hasil dari
evaluation/generate_ai_5w1h.py, kolom "*_pred" = sistem, "*_truth" = AI).
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.text_similarity import bertscore_similarity

CATEGORIES = ["what", "who", "when", "where", "why", "how"]
NOT_FOUND = "Tidak disebutkan dalam artikel"
DEFAULT_PATH = "5w1h_eval_ai.csv"


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _similarity_score(pred: str, truth: str) -> float:
    pred, truth = _clean(pred), _clean(truth)
    pred_empty = (not pred) or (NOT_FOUND.lower() in pred.lower())
    ai_empty = (not truth) or (NOT_FOUND.lower() in truth.lower())

    if pred_empty and ai_empty:
        return 1.0
    if pred_empty or ai_empty:
        return 0.0

    return bertscore_similarity(pred, truth)


def _interpret(pct: float) -> str:
    if pct >= 80:
        return "Sangat relevan (konsisten dengan AI)"
    if pct >= 60:
        return "Relevan"
    if pct >= 40:
        return "Cukup relevan"
    return "Kurang relevan (banyak perbedaan dengan AI)"


def compare(path: str = DEFAULT_PATH):
    print("=" * 70)
    print("VALIDASI AI - STEP B: PERBANDINGAN SISTEM vs AI (BERTScore)")
    print("=" * 70)

    print(f"\n📄 Memuat '{path}'...")
    df = pd.read_csv(path)
    print(f"   Total baris: {len(df)}")

    per_row_scores = {cat: [] for cat in CATEGORIES}
    examples = {cat: [] for cat in CATEGORIES}

    print(f"\n🔎 Menghitung skor kemiripan sistem vs AI per komponen...")
    for i, row in df.iterrows():
        for cat in CATEGORIES:
            pred = row.get(f"{cat}_pred", "")
            ai_answer = row.get(f"{cat}_truth", "")
            score = _similarity_score(pred, ai_answer)
            per_row_scores[cat].append(score)
            examples[cat].append({
                "id": row.get("id", i),
                "title": row.get("title", ""),
                "pred": _clean(pred),
                "ai": _clean(ai_answer),
                "score": score,
            })
        if (i + 1) % 10 == 0:
            print(f"   ...{i + 1}/{len(df)} selesai")

    print("\n" + "=" * 70)
    print("HASIL: RATA-RATA SKOR KEMIRIPAN SISTEM vs AI PER KATEGORI")
    print("=" * 70)
    summary = {}
    for cat in CATEGORIES:
        scores = per_row_scores[cat]
        n = len(scores)
        avg_pct = (sum(scores) / n) * 100 if n else 0.0
        summary[cat] = {"avg_similarity_pct": avg_pct, "n": n}
        print(f"  {cat.upper():6s} | Kemiripan rata-rata: {avg_pct:6.2f}% "
              f"| {_interpret(avg_pct)} | n={n}")

    overall_avg = sum(v["avg_similarity_pct"] for v in summary.values()) / len(summary)
    print(f"\n  {'RATA2':6s} | Kemiripan keseluruhan (6 kategori): {overall_avg:.2f}%")
    print(f"           {_interpret(overall_avg)}")

    # --- 2 contoh kasus per kategori (paling konsisten & paling beda) ---
    print("\n" + "=" * 70)
    print("CONTOH KASUS PER KATEGORI (2 kasus -- paling konsisten & paling beda)")
    print("=" * 70)
    for cat in CATEGORIES:
        ex_sorted = sorted(examples[cat], key=lambda e: -e["score"])
        best = ex_sorted[0]
        worst = ex_sorted[-1]
        print(f"\n[{cat.upper()}]")
        print(f"  Paling KONSISTEN (id={best['id']}, skor={best['score']*100:.1f}%):")
        print(f"    Judul  : {best['title']}")
        print(f"    Sistem : {best['pred']}")
        print(f"    AI     : {best['ai']}")
        print(f"  Paling BEDA      (id={worst['id']}, skor={worst['score']*100:.1f}%):")
        print(f"    Judul  : {worst['title']}")
        print(f"    Sistem : {worst['pred']}")
        print(f"    AI     : {worst['ai']}")

    detail_rows = []
    for i in range(len(df)):
        row_detail = {"id": df.iloc[i].get("id", i), "title": df.iloc[i].get("title", "")}
        for cat in CATEGORIES:
            row_detail[f"{cat}_similarity_pct"] = round(per_row_scores[cat][i] * 100, 2)
        detail_rows.append(row_detail)
    detail_path = os.path.splitext(path)[0] + "_comparison_detail.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Detail skor per baris disimpan ke '{detail_path}'")

    return summary


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        print(f"❌ File '{path}' tidak ditemukan.")
        print(f"   Jalankan dulu evaluation/generate_ai_5w1h.py untuk membuatnya.")
        sys.exit(1)
    compare(path)
