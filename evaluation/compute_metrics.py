"""
compute_metrics.py — Step 4: Hitung metrik evaluasi dari file hasil

Membaca hasil pencarian dari results/ dan ground truth,
menghitung Precision@5, Recall@5, Hit Rate@5, dan MRR
untuk setiap metode, lalu menyimpan ringkasan ke summary_metrics.csv.
"""

import os
import sys
import json
import csv

# Fix encoding untuk Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def precision_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Precision@K: Berapa proporsi dari top-K yang relevan.
    Formula: |relevant ∩ retrieved[:k]| / k
    """
    retrieved_top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_top_k if doc_id in relevant_set)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Recall@K: Berapa proporsi dari artikel relevan yang berhasil ditemukan di top-K.
    Formula: |relevant ∩ retrieved[:k]| / |relevant|
    """
    if len(relevant_ids) == 0:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_top_k if doc_id in relevant_set)
    return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Hit Rate@K: 1 jika ada minimal 1 artikel relevan di top-K, else 0.
    Metrik yang mudah dipahami: "apakah sistem berhasil menemukan sesuatu yang relevan?"
    """
    retrieved_top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return 1.0 if len(retrieved_top_k & relevant_set) > 0 else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Mean Reciprocal Rank (per query): 1 / rank artikel relevan pertama.
    Jika tidak ada artikel relevan di hasil, return 0.
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_all_metrics(results_path, k=5):
    """
    Hitung semua metrik untuk satu metode pencarian.
    
    Args:
        results_path: Path ke file results JSON (misal results_bm25.json)
        k: Cutoff untuk Precision@K, Recall@K, Hit Rate@K
    
    Returns:
        Dict berisi metrik per query dan rata-rata
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    per_query_metrics = []

    for query_id, query_data in results_data.items():
        retrieved_ids = [r["article_id"] for r in query_data["results"]]
        relevant_ids = query_data["relevant_article_ids"]

        p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)
        r_at_k = recall_at_k(retrieved_ids, relevant_ids, k)
        hr_at_k = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        per_query_metrics.append({
            "query_id": query_id,
            "query_text": query_data["query_text"],
            "category": query_data["category"],
            "num_relevant": len(relevant_ids),
            f"precision@{k}": round(p_at_k, 4),
            f"recall@{k}": round(r_at_k, 4),
            f"hit_rate@{k}": round(hr_at_k, 4),
            "mrr": round(rr, 4),
        })

    # Hitung rata-rata
    n = len(per_query_metrics)
    if n == 0:
        return {"per_query": [], "average": {}}

    avg_metrics = {
        f"precision@{k}": round(sum(m[f"precision@{k}"] for m in per_query_metrics) / n, 4),
        f"recall@{k}": round(sum(m[f"recall@{k}"] for m in per_query_metrics) / n, 4),
        f"hit_rate@{k}": round(sum(m[f"hit_rate@{k}"] for m in per_query_metrics) / n, 4),
        "mrr": round(sum(m["mrr"] for m in per_query_metrics) / n, 4),
    }

    return {
        "per_query": per_query_metrics,
        "average": avg_metrics
    }


def compute_category_metrics(per_query_metrics, k=5):
    """Hitung rata-rata metrik per kategori query."""
    categories = {}
    for m in per_query_metrics:
        cat = m["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(m)

    category_averages = {}
    for cat, metrics_list in categories.items():
        n = len(metrics_list)
        category_averages[cat] = {
            "count": n,
            f"precision@{k}": round(sum(m[f"precision@{k}"] for m in metrics_list) / n, 4),
            f"recall@{k}": round(sum(m[f"recall@{k}"] for m in metrics_list) / n, 4),
            f"hit_rate@{k}": round(sum(m[f"hit_rate@{k}"] for m in metrics_list) / n, 4),
            "mrr": round(sum(m["mrr"] for m in metrics_list) / n, 4),
        }

    return category_averages


def save_summary_csv(all_metrics, output_path, k=5):
    """Simpan ringkasan metrik ke CSV."""
    rows = []

    # --- Per-query detail ---
    for method_name, method_metrics in all_metrics.items():
        for m in method_metrics["per_query"]:
            rows.append({
                "method": method_name,
                "query_id": m["query_id"],
                "query_text": m["query_text"],
                "category": m["category"],
                "num_relevant": m["num_relevant"],
                f"precision@{k}": m[f"precision@{k}"],
                f"recall@{k}": m[f"recall@{k}"],
                f"hit_rate@{k}": m[f"hit_rate@{k}"],
                "mrr": m["mrr"],
            })

    # --- Rata-rata per metode ---
    for method_name, method_metrics in all_metrics.items():
        avg = method_metrics["average"]
        rows.append({
            "method": method_name,
            "query_id": "AVERAGE",
            "query_text": "",
            "category": "ALL",
            "num_relevant": "",
            f"precision@{k}": avg[f"precision@{k}"],
            f"recall@{k}": avg[f"recall@{k}"],
            f"hit_rate@{k}": avg[f"hit_rate@{k}"],
            "mrr": avg["mrr"],
        })

    fieldnames = [
        "method", "query_id", "query_text", "category", "num_relevant",
        f"precision@{k}", f"recall@{k}", f"hit_rate@{k}", "mrr"
    ]

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    output_csv = os.path.join(script_dir, "summary_metrics.csv")
    output_json = os.path.join(script_dir, "detailed_metrics.json")

    K = 5  # Cutoff

    print("=" * 60)
    print("STEP 4: MENGHITUNG METRIK EVALUASI")
    print("=" * 60)

    # --- Hitung metrik untuk setiap metode ---
    methods = {
        "BM25": os.path.join(results_dir, "results_bm25.json"),
        "Semantic": os.path.join(results_dir, "results_semantic.json"),
        "Hybrid": os.path.join(results_dir, "results_hybrid.json"),
    }

    all_metrics = {}

    for method_name, results_path in methods.items():
        if not os.path.exists(results_path):
            print(f"⚠️  File {results_path} tidak ditemukan, skip {method_name}")
            continue

        print(f"\n📊 Menghitung metrik untuk {method_name}...")
        metrics = compute_all_metrics(results_path, k=K)
        all_metrics[method_name] = metrics

        avg = metrics["average"]
        print(f"   Precision@{K} : {avg[f'precision@{K}']:.4f}")
        print(f"   Recall@{K}    : {avg[f'recall@{K}']:.4f}")
        print(f"   Hit Rate@{K}  : {avg[f'hit_rate@{K}']:.4f}")
        print(f"   MRR          : {avg['mrr']:.4f}")

    # --- Tampilkan tabel perbandingan ---
    print(f"\n{'=' * 60}")
    print(f"  RINGKASAN PERBANDINGAN METRIK (rata-rata dari semua query)")
    print(f"{'=' * 60}")
    print(f"\n  {'Metode':<12} {'Precision@5':>12} {'Recall@5':>10} {'Hit Rate@5':>12} {'MRR':>8}")
    print(f"  {'─' * 54}")

    for method_name, metrics in all_metrics.items():
        avg = metrics["average"]
        print(f"  {method_name:<12} {avg[f'precision@{K}']:>12.4f} {avg[f'recall@{K}']:>10.4f} {avg[f'hit_rate@{K}']:>12.4f} {avg['mrr']:>8.4f}")

    # --- Metrik per kategori ---
    print(f"\n{'=' * 60}")
    print(f"  PERBANDINGAN PER KATEGORI QUERY")
    print(f"{'=' * 60}")

    categories_order = ["keyword_spesifik", "natural_language", "sinonim", "typo"]
    category_labels = {
        "keyword_spesifik": "Keyword Spesifik",
        "natural_language": "Natural Language",
        "sinonim": "Sinonim/Tidak Eksak",
        "typo": "Typo Ringan",
    }

    for cat in categories_order:
        print(f"\n  📂 {category_labels.get(cat, cat)}")
        print(f"  {'Metode':<12} {'P@5':>8} {'R@5':>8} {'HR@5':>8} {'MRR':>8}")
        print(f"  {'─' * 40}")

        for method_name, metrics in all_metrics.items():
            cat_metrics = compute_category_metrics(metrics["per_query"], k=K)
            if cat in cat_metrics:
                cm = cat_metrics[cat]
                print(f"  {method_name:<12} {cm[f'precision@{K}']:>8.4f} {cm[f'recall@{K}']:>8.4f} {cm[f'hit_rate@{K}']:>8.4f} {cm['mrr']:>8.4f}")

    # --- Simpan hasil ---
    save_summary_csv(all_metrics, output_csv, k=K)
    print(f"\n✅ Ringkasan metrik disimpan ke: {output_csv}")

    # Simpan detailed metrics ke JSON (untuk analyze_results.py)
    # Tambahkan metrik per kategori
    detailed = {}
    for method_name, metrics in all_metrics.items():
        cat_metrics = compute_category_metrics(metrics["per_query"], k=K)
        detailed[method_name] = {
            "average": metrics["average"],
            "per_query": metrics["per_query"],
            "per_category": cat_metrics
        }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)
    print(f"✅ Detail metrik disimpan ke: {output_json}")

    print(f"\n👉 Langkah selanjutnya: jalankan 'python analyze_results.py'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
