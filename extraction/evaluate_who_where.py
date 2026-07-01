"""
evaluate_who_where.py
======================
Evaluasi WHO & WHERE pakai set-based Precision/Recall/F1 (metrik yang
TEPAT untuk daftar entitas pendek -- bukan ROUGE, yang didesain untuk
ringkasan paragraf panjang).

Cara pakai:
1. Siapkan file ground_truth.csv dengan kolom: id, who_truth, where_truth
   (pisahkan multi-entitas dengan ';', contoh: "Sri Mulyani;Jokowi")
2. Jalankan: python evaluate_who_where.py
3. Hasil: precision/recall/F1 per artikel + rata-rata keseluruhan,
   dan opsional skor ROUGE-L kalau memang diminta laporan.

CATATAN ROUGE: kalau kamu tetap mau lapor angka ROUGE (mis. karena
mengikuti format penelitian/skripsi orang lain), fungsi rouge_score_who_where()
di bawah menggabungkan list jadi 1 string lalu hitung ROUGE-L -- tapi
INTERPRETASINYA TERBATAS karena ROUGE sensitif urutan & tidak peduli soal
"apakah ini benar nama orang", cuma overlap token. F1 set-based lebih
jujur menggambarkan kualitas ekstraksi.
"""

import re
from typing import List


def normalize_entity(s: str) -> str:
    """Normalisasi ringan supaya 'Sri Mulyani' == 'sri  mulyani' dst."""
    return re.sub(r'\s+', ' ', s.lower().strip())


def fuzzy_match(a: str, b: str) -> bool:
    """Match longgar: exact, atau salah satu substring dari yang lain
    (menangani 'Sri Mulyani' vs 'Sri Mulyani Indrawati')."""
    a, b = normalize_entity(a), normalize_entity(b)
    if a == b:
        return True
    return a in b or b in a


def set_based_prf1(predicted: List[str], truth: List[str]):
    """
    Precision/Recall/F1 dengan fuzzy matching satu-ke-satu (greedy).
    predicted, truth: list string entitas untuk SATU artikel.
    """
    if not truth or truth == ["Tidak disebutkan dalam artikel"]:
        # Kalau ground truth memang kosong, predicted idealnya juga kosong
        if not predicted or predicted == ["Tidak disebutkan dalam artikel"]:
            return 1.0, 1.0, 1.0
        return 0.0, 1.0, 0.0  # over-predicting saat seharusnya kosong

    pred_remaining = list(predicted)
    matched_truth = 0
    matched_pred = set()

    for t in truth:
        for i, p in enumerate(pred_remaining):
            if i in matched_pred:
                continue
            if fuzzy_match(p, t):
                matched_truth += 1
                matched_pred.add(i)
                break

    precision = len(matched_pred) / len(predicted) if predicted else 0.0
    recall = matched_truth / len(truth) if truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_dataset(rows):
    """
    rows: List[dict] dengan keys: who_pred, who_truth, where_pred, where_truth
          (masing-masing List[str])
    Returns: dict ringkasan rata-rata + per-baris detail.
    """
    results = {"who": [], "where": []}

    for row in rows:
        p, r, f = set_based_prf1(row["who_pred"], row["who_truth"])
        results["who"].append({"precision": p, "recall": r, "f1": f})

        p, r, f = set_based_prf1(row["where_pred"], row["where_truth"])
        results["where"].append({"precision": p, "recall": r, "f1": f})

    summary = {}
    for key in ("who", "where"):
        n = len(results[key])
        summary[key] = {
            "precision": sum(x["precision"] for x in results[key]) / n if n else 0,
            "recall": sum(x["recall"] for x in results[key]) / n if n else 0,
            "f1": sum(x["f1"] for x in results[key]) / n if n else 0,
            "n": n,
        }
    return summary, results


def rouge_score_who_where(predicted: List[str], truth: List[str]):
    """Opsional: ROUGE-L kalau memang dibutuhkan untuk laporan.
    Butuh: pip install rouge-score --break-system-packages"""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    pred_str = "; ".join(predicted)
    truth_str = "; ".join(truth)
    scores = scorer.score(truth_str, pred_str)
    return scores['rougeL'].fmeasure


# =============================================================================
# CONTOH PAKAI
# =============================================================================
if __name__ == "__main__":
    # Contoh dummy -- ganti dengan data asli + ground truth manual kamu.
    # Jalankan dari root project (folder yang berisi app.py) supaya
    # import "extraction.xxx" di file lain tetap konsisten.
    sample_rows = [
        {
            "who_pred": ["Sri Mulyani Indrawati", "Joko Widodo"],
            "who_truth": ["Sri Mulyani", "Jokowi"],  # fuzzy match akan gagal
                                                       # utk "Jokowi" vs "Joko Widodo"
                                                       # -- ini contoh kenapa
                                                       # ground truth harus konsisten
            "where_pred": ["Jakarta", "Gedung KPK"],
            "where_truth": ["Jakarta", "KPK"],
        },
    ]

    summary, details = evaluate_dataset(sample_rows)
    print("=== RINGKASAN EVALUASI ===")
    for key, vals in summary.items():
        print(f"{key.upper()}: Precision={vals['precision']:.3f} "
              f"Recall={vals['recall']:.3f} F1={vals['f1']:.3f} (n={vals['n']})")
