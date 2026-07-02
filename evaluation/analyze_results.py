"""
analyze_results.py — Step 5: Analisis & Visualisasi hasil evaluasi

Menghasilkan:
1. Tabel perbandingan metrik (Markdown + CSV)
2. Bar chart perbandingan metrik rata-rata per metode
3. Tabel performa per kategori query
4. Studi kasus: query dengan perbedaan terbesar antar metode
5. Analisis similarity score (metrik pendukung)
"""

import os
import sys
import json
import csv
import math

# Fix encoding untuk Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Coba import matplotlib, jika tidak ada gunakan text-based output
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib tidak terinstall. Grafik akan diskip.")
    print("   Install dengan: pip install matplotlib")


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_comparison_table(detailed_metrics, k=5):
    """Generate tabel perbandingan dalam format Markdown."""
    lines = []
    lines.append("# Tabel Perbandingan Metrik Evaluasi")
    lines.append("")
    lines.append(f"| Metode | Precision@{k} | Recall@{k} | Hit Rate@{k} | MRR |")
    lines.append("|--------|-------------|----------|------------|-----|")

    for method in ["BM25", "Semantic", "Hybrid"]:
        if method not in detailed_metrics:
            continue
        avg = detailed_metrics[method]["average"]
        lines.append(
            f"| **{method}** | "
            f"{avg[f'precision@{k}']:.4f} | "
            f"{avg[f'recall@{k}']:.4f} | "
            f"{avg[f'hit_rate@{k}']:.4f} | "
            f"{avg['mrr']:.4f} |"
        )

    lines.append("")

    # Cari metode terbaik per metrik
    lines.append("### Metode Terbaik per Metrik")
    lines.append("")
    for metric_name in [f"precision@{k}", f"recall@{k}", f"hit_rate@{k}", "mrr"]:
        best_method = None
        best_value = -1
        for method, data in detailed_metrics.items():
            val = data["average"].get(metric_name, 0)
            if val > best_value:
                best_value = val
                best_method = method
        display_name = metric_name.upper().replace("@", "@").replace("_", " ")
        lines.append(f"- **{display_name}**: {best_method} ({best_value:.4f})")

    return "\n".join(lines)


def generate_category_table(detailed_metrics, k=5):
    """Generate tabel performa per kategori query."""
    lines = []
    lines.append("# Perbandingan Performa per Kategori Query")
    lines.append("")

    category_labels = {
        "keyword_spesifik": "Keyword Spesifik",
        "natural_language": "Natural Language",
        "sinonim": "Sinonim/Tidak Eksak",
        "typo": "Typo Ringan",
    }

    for cat_key, cat_label in category_labels.items():
        lines.append(f"## {cat_label}")
        lines.append("")
        lines.append(f"| Metode | P@{k} | R@{k} | HR@{k} | MRR |")
        lines.append("|--------|-------|-------|--------|-----|")

        for method in ["BM25", "Semantic", "Hybrid"]:
            if method not in detailed_metrics:
                continue
            cat_data = detailed_metrics[method].get("per_category", {}).get(cat_key)
            if cat_data:
                lines.append(
                    f"| {method} | "
                    f"{cat_data[f'precision@{k}']:.4f} | "
                    f"{cat_data[f'recall@{k}']:.4f} | "
                    f"{cat_data[f'hit_rate@{k}']:.4f} | "
                    f"{cat_data['mrr']:.4f} |"
                )

        lines.append("")

    return "\n".join(lines)


def find_case_studies(detailed_metrics, k=5, num_cases=3):
    """
    Temukan query yang menunjukkan perbedaan paling signifikan antar metode.
    Fokus pada query di mana ada gap besar antara metode terbaik dan terburuk.
    """
    cases = []

    # Kumpulkan semua query dengan metrik per metode
    query_ids_set = set()
    for method, data in detailed_metrics.items():
        for m in data["per_query"]:
            query_ids_set.add(m["query_id"])

    for qid in sorted(query_ids_set):
        query_metrics = {}
        query_text = ""
        category = ""

        for method, data in detailed_metrics.items():
            for m in data["per_query"]:
                if m["query_id"] == qid:
                    query_text = m["query_text"]
                    category = m["category"]
                    query_metrics[method] = m[f"precision@{k}"]
                    break

        if len(query_metrics) < 2:
            continue

        values = list(query_metrics.values())
        gap = max(values) - min(values)

        best_method = max(query_metrics, key=query_metrics.get)
        worst_method = min(query_metrics, key=query_metrics.get)

        cases.append({
            "query_id": qid,
            "query_text": query_text,
            "category": category,
            "gap": gap,
            "best_method": best_method,
            "worst_method": worst_method,
            "metrics": query_metrics
        })

    # Sort by gap (perbedaan terbesar)
    cases.sort(key=lambda x: x["gap"], reverse=True)
    return cases[:num_cases]


def generate_case_studies_text(cases, detailed_metrics, results_dir, k=5):
    """Generate teks studi kasus."""
    lines = []
    lines.append("# Studi Kasus: Query dengan Perbedaan Signifikan")
    lines.append("")

    for i, case in enumerate(cases, 1):
        lines.append(f"## Kasus {i}: \"{case['query_text']}\"")
        lines.append(f"- **Kategori**: {case['category']}")
        lines.append(f"- **Metode terbaik**: {case['best_method']} (P@{k}={case['metrics'][case['best_method']]:.4f})")
        lines.append(f"- **Metode terburuk**: {case['worst_method']} (P@{k}={case['metrics'][case['worst_method']]:.4f})")
        lines.append(f"- **Gap**: {case['gap']:.4f}")
        lines.append("")

        # Tampilkan top-5 hasil dari setiap metode
        lines.append(f"| Rank | BM25 | Semantic | Hybrid |")
        lines.append("|------|------|----------|--------|")

        # Load results files untuk mendapatkan detail
        for rank_idx in range(k):
            row_parts = [str(rank_idx + 1)]
            for method_file, method_name in [
                ("results_bm25.json", "BM25"),
                ("results_semantic.json", "Semantic"),
                ("results_hybrid.json", "Hybrid")
            ]:
                filepath = os.path.join(results_dir, method_file)
                if os.path.exists(filepath):
                    results_data = load_json(filepath)
                    query_data = results_data.get(case["query_id"], {})
                    results_list = query_data.get("results", [])
                    relevant_ids = set(query_data.get("relevant_article_ids", []))

                    if rank_idx < len(results_list):
                        r = results_list[rank_idx]
                        is_rel = "✅" if r["article_id"] in relevant_ids else "❌"
                        title_short = r["title"][:30] + "..." if len(r["title"]) > 30 else r["title"]
                        row_parts.append(f"{is_rel} {title_short}")
                    else:
                        row_parts.append("-")
                else:
                    row_parts.append("-")

            lines.append("| " + " | ".join(row_parts) + " |")

        lines.append("")

        # Analisis
        lines.append(f"**Analisis**: ", )
        if case["category"] == "keyword_spesifik":
            lines.append(f"Query keyword spesifik ini menunjukkan bahwa {case['best_method']} lebih efektif dalam mencocokkan kata kunci eksak.")
        elif case["category"] == "natural_language":
            lines.append(f"Query natural language ini menunjukkan bahwa {case['best_method']} lebih mampu memahami makna kalimat secara keseluruhan.")
        elif case["category"] == "sinonim":
            lines.append(f"Query dengan sinonim ini menunjukkan bahwa {case['best_method']} lebih toleran terhadap variasi istilah.")
        elif case["category"] == "typo":
            lines.append(f"Query dengan typo ini menunjukkan bahwa {case['best_method']} lebih robust terhadap kesalahan ketik.")
        lines.append("")

    return "\n".join(lines)


def generate_bar_chart(detailed_metrics, output_path, k=5):
    """Generate bar chart perbandingan metrik rata-rata."""
    if not HAS_MATPLOTLIB:
        print("   ⏭️ Skip bar chart (matplotlib tidak tersedia)")
        return

    methods = []
    precision_vals = []
    recall_vals = []
    hitrate_vals = []
    mrr_vals = []

    for method in ["BM25", "Semantic", "Hybrid"]:
        if method not in detailed_metrics:
            continue
        avg = detailed_metrics[method]["average"]
        methods.append(method)
        precision_vals.append(avg[f"precision@{k}"])
        recall_vals.append(avg[f"recall@{k}"])
        hitrate_vals.append(avg[f"hit_rate@{k}"])
        mrr_vals.append(avg["mrr"])

    x = range(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar([i - 1.5 * width for i in x], precision_vals, width, label=f'Precision@{k}', color='#00d2ff')
    bars2 = ax.bar([i - 0.5 * width for i in x], recall_vals, width, label=f'Recall@{k}', color='#7b2ff7')
    bars3 = ax.bar([i + 0.5 * width for i in x], hitrate_vals, width, label=f'Hit Rate@{k}', color='#10b981')
    bars4 = ax.bar([i + 1.5 * width for i in x], mrr_vals, width, label='MRR', color='#f59e0b')

    ax.set_xlabel('Metode Pencarian', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Perbandingan Metrik Evaluasi per Metode Pencarian', fontsize=14, fontweight='bold')
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Tambahkan nilai di atas bar
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Bar chart disimpan ke: {output_path}")


def generate_category_chart(detailed_metrics, output_path, k=5):
    """Generate grouped bar chart per kategori query."""
    if not HAS_MATPLOTLIB:
        print("   ⏭️ Skip category chart (matplotlib tidak tersedia)")
        return

    category_labels = {
        "keyword_spesifik": "Keyword\nSpesifik",
        "natural_language": "Natural\nLanguage",
        "sinonim": "Sinonim",
        "typo": "Typo",
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    metric_names = [f"precision@{k}", f"recall@{k}", f"hit_rate@{k}", "mrr"]
    metric_labels = [f"Precision@{k}", f"Recall@{k}", f"Hit Rate@{k}", "MRR"]
    colors = {"BM25": "#00d2ff", "Semantic": "#7b2ff7", "Hybrid": "#10b981"}

    for ax, metric_name, metric_label in zip(axes, metric_names, metric_labels):
        categories = list(category_labels.keys())
        x = range(len(categories))
        width = 0.25

        for mi, method in enumerate(["BM25", "Semantic", "Hybrid"]):
            if method not in detailed_metrics:
                continue
            vals = []
            for cat in categories:
                cat_data = detailed_metrics[method].get("per_category", {}).get(cat, {})
                vals.append(cat_data.get(metric_name, 0))

            offset = (mi - 1) * width
            ax.bar([i + offset for i in x], vals, width, label=method, color=colors[method])

        ax.set_title(metric_label, fontsize=11, fontweight='bold')
        ax.set_xticks(list(x))
        ax.set_xticklabels([category_labels[c] for c in categories], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Skor', fontsize=11)
    axes[-1].legend(loc='upper right', fontsize=9)

    fig.suptitle('Perbandingan Performa per Kategori Query', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Category chart disimpan ke: {output_path}")


def analyze_score_distributions(results_dir):
    """
    Analisis distribusi similarity score (metrik pendukung).
    CATATAN: Skor antar metode TIDAK boleh dibandingkan langsung
    karena skalanya berbeda (BM25=TF-IDF, Semantic=cosine, Hybrid=RRF).
    """
    lines = []
    lines.append("# Analisis Distribusi Skor (Metrik Pendukung)")
    lines.append("")
    lines.append("> **Catatan**: Skor antar metode TIDAK dapat dibandingkan langsung")
    lines.append("> karena menggunakan skala berbeda (BM25 = TF-IDF score, Semantic = cosine similarity 0-1,")
    lines.append("> Hybrid = RRF score). Analisis ini hanya menunjukkan distribusi internal masing-masing metode.")
    lines.append("")

    for method_file, method_name in [
        ("results_bm25.json", "BM25"),
        ("results_semantic.json", "Semantic"),
        ("results_hybrid.json", "Hybrid")
    ]:
        filepath = os.path.join(results_dir, method_file)
        if not os.path.exists(filepath):
            continue

        data = load_json(filepath)
        all_scores = []
        for qid, qdata in data.items():
            for r in qdata["results"]:
                all_scores.append(r["score"])

        if not all_scores:
            continue

        lines.append(f"## {method_name}")
        lines.append(f"- Jumlah skor: {len(all_scores)}")
        lines.append(f"- Min: {min(all_scores):.4f}")
        lines.append(f"- Max: {max(all_scores):.4f}")
        lines.append(f"- Mean: {sum(all_scores)/len(all_scores):.4f}")

        # Median
        sorted_scores = sorted(all_scores)
        mid = len(sorted_scores) // 2
        median = sorted_scores[mid] if len(sorted_scores) % 2 == 1 else (sorted_scores[mid-1] + sorted_scores[mid]) / 2
        lines.append(f"- Median: {median:.4f}")
        lines.append("")

    return "\n".join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    detailed_metrics_path = os.path.join(script_dir, "detailed_metrics.json")
    output_dir = os.path.join(script_dir, "results")

    K = 5

    print("=" * 60)
    print("STEP 5: ANALISIS & VISUALISASI HASIL EVALUASI")
    print("=" * 60)

    if not os.path.exists(detailed_metrics_path):
        print("❌ File detailed_metrics.json tidak ditemukan!")
        print("   Jalankan 'python compute_metrics.py' terlebih dahulu.")
        sys.exit(1)

    detailed_metrics = load_json(detailed_metrics_path)

    # --- 1. Tabel perbandingan ---
    print("\n📊 Generating tabel perbandingan...")
    comparison_table = generate_comparison_table(detailed_metrics, K)

    # --- 2. Tabel per kategori ---
    print("📊 Generating tabel per kategori...")
    category_table = generate_category_table(detailed_metrics, K)

    # --- 3. Studi kasus ---
    print("🔍 Mencari studi kasus...")
    cases = find_case_studies(detailed_metrics, K, num_cases=3)
    case_studies_text = generate_case_studies_text(cases, detailed_metrics, results_dir, K)

    # --- 4. Analisis skor ---
    print("📈 Analisis distribusi skor...")
    score_analysis = analyze_score_distributions(results_dir)

    # --- Gabungkan semua ke satu laporan ---
    report = "\n\n---\n\n".join([
        comparison_table,
        category_table,
        case_studies_text,
        score_analysis
    ])

    report_path = os.path.join(script_dir, "evaluation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ Laporan lengkap disimpan ke: {report_path}")

    # --- 5. Grafik ---
    print("\n🎨 Generating grafik...")
    chart_path = os.path.join(output_dir, "chart_comparison.png")
    generate_bar_chart(detailed_metrics, chart_path, K)

    category_chart_path = os.path.join(output_dir, "chart_per_category.png")
    generate_category_chart(detailed_metrics, category_chart_path, K)

    # --- Tampilkan ringkasan ---
    print(f"\n{'=' * 60}")
    print(f"  ✅ ANALISIS SELESAI!")
    print(f"{'=' * 60}")
    print(f"\n  📄 Laporan    : {report_path}")
    if HAS_MATPLOTLIB:
        print(f"  📊 Bar chart  : {chart_path}")
        print(f"  📊 Per kategori: {category_chart_path}")
    print(f"\n  File-file ini bisa langsung digunakan untuk bab hasil pengujian skripsi.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
