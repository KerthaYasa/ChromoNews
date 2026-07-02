"""
annotate_helper.py — Step 2: CLI interaktif untuk anotasi ground truth

Membaca annotation_pool.json dan menampilkan artikel satu per satu
agar user bisa menandai relevan (y) atau tidak relevan (n).
Hasil anotasi disimpan ke ground_truth.json.

Progress otomatis disimpan — bisa dilanjutkan jika berhenti di tengah jalan.
"""

import os
import sys
import json

# Fix encoding untuk Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_pool(pool_path):
    """Load annotation pool dari file JSON."""
    with open(pool_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pool(pool_data, pool_path):
    """Simpan annotation pool (dengan progress anotasi) ke file JSON."""
    with open(pool_path, 'w', encoding='utf-8') as f:
        json.dump(pool_data, f, ensure_ascii=False, indent=2)


def save_ground_truth(pool_data, output_path):
    """
    Konversi annotation pool yang sudah dianotasi menjadi ground_truth.json.
    Hanya menyimpan query yang sudah selesai dianotasi (semua artikel sudah ditandai).
    """
    ground_truth = {
        "description": "Ground truth untuk evaluasi metode pencarian ChromoNews",
        "annotation_method": "pooling (union top-10 BM25 + Semantic + Hybrid), binary relevance",
        "queries": []
    }

    for q in pool_data["queries"]:
        # Cek apakah semua artikel sudah dianotasi
        all_annotated = all(item["is_relevant"] is not None for item in q["pool"])
        if not all_annotated:
            continue

        relevant_ids = [
            item["article_id"] for item in q["pool"]
            if item["is_relevant"] is True
        ]

        ground_truth["queries"].append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "category": q["category"],
            "relevant_article_ids": relevant_ids,
            "total_pool_size": len(q["pool"]),
            "total_relevant": len(relevant_ids)
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    return len(ground_truth["queries"])


def count_progress(pool_data):
    """Hitung progress anotasi keseluruhan."""
    total_articles = 0
    annotated_articles = 0
    completed_queries = 0

    for q in pool_data["queries"]:
        all_done = True
        for item in q["pool"]:
            total_articles += 1
            if item["is_relevant"] is not None:
                annotated_articles += 1
            else:
                all_done = False
        if all_done:
            completed_queries += 1

    return total_articles, annotated_articles, completed_queries


def run_annotation(pool_path, output_path):
    """Jalankan proses anotasi interaktif."""
    pool_data = load_pool(pool_path)

    total_queries = len(pool_data["queries"])
    total_articles, annotated_articles, completed_queries = count_progress(pool_data)

    print("=" * 65)
    print("  ANNOTATE HELPER — Anotasi Ground Truth untuk Evaluasi")
    print("=" * 65)
    print(f"\n  Total query       : {total_queries}")
    print(f"  Total artikel     : {total_articles}")
    print(f"  Sudah dianotasi   : {annotated_articles}/{total_articles}")
    print(f"  Query selesai     : {completed_queries}/{total_queries}")
    print(f"\n  Perintah:")
    print(f"    y = relevan")
    print(f"    n = tidak relevan")
    print(f"    s = skip (lanjut ke artikel berikutnya)")
    print(f"    q = quit (simpan progress & keluar)")
    print(f"    j = jump ke query tertentu (misal: j q05)")
    print("=" * 65)

    if annotated_articles == total_articles:
        print("\n🎉 Semua artikel sudah dianotasi!")
        saved = save_ground_truth(pool_data, output_path)
        print(f"✅ Ground truth disimpan ke: {output_path}")
        print(f"   ({saved} query tersimpan)")
        return

    for qi, q in enumerate(pool_data["queries"]):
        # Skip query yang sudah selesai
        all_done = all(item["is_relevant"] is not None for item in q["pool"])
        if all_done:
            continue

        print(f"\n{'─' * 65}")
        print(f"  📋 Query [{qi+1}/{total_queries}]: {q['query_id']}")
        print(f"  🔍 \"{q['query_text']}\"")
        print(f"  📂 Kategori: {q['category']}")
        print(f"  📊 Pool: {len(q['pool'])} artikel")
        print(f"{'─' * 65}")

        for ai, item in enumerate(q["pool"]):
            # Skip artikel yang sudah dianotasi
            if item["is_relevant"] is not None:
                status = "✅ relevan" if item["is_relevant"] else "❌ tidak"
                print(f"\n  [{ai+1}/{len(q['pool'])}] (sudah: {status}) {item['title'][:60]}")
                continue

            # Tampilkan info artikel
            found_by = item.get("found_by", {})
            sources = []
            if found_by.get("bm25"):
                sources.append("BM25")
            if found_by.get("semantic"):
                sources.append("Semantic")
            if found_by.get("hybrid"):
                sources.append("Hybrid")

            print(f"\n  [{ai+1}/{len(q['pool'])}] Artikel #{item['article_id']}")
            print(f"  {'=' * 55}")
            print(f"  Judul  : {item['title']}")
            if item.get('date'):
                print(f"  Tanggal: {item['date']}")
            print(f"  Ditemukan oleh: {', '.join(sources)}")
            print(f"  {'─' * 55}")
            
            # Tampilkan konten penuh artikel
            content = item.get('content', item.get('snippet', ''))
            # Wrap text agar mudah dibaca di terminal
            words = content.split()
            line = "  "
            for word in words:
                if len(line) + len(word) + 1 > 80:
                    print(line)
                    line = "  " + word
                else:
                    line += " " + word if line.strip() else "  " + word
            if line.strip():
                print(line)
            print(f"  {'─' * 55}")

            while True:
                try:
                    answer = input(f"\n  Relevan untuk \"{q['query_text'][:40]}...\"? (y/n/s/q/j): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "q"

                if answer == "y":
                    item["is_relevant"] = True
                    print("  → ✅ Ditandai RELEVAN")
                    break
                elif answer == "n":
                    item["is_relevant"] = False
                    print("  → ❌ Ditandai TIDAK RELEVAN")
                    break
                elif answer == "s":
                    print("  → ⏭️ Diskip")
                    break
                elif answer == "q":
                    # Simpan progress dan keluar
                    save_pool(pool_data, pool_path)
                    saved = save_ground_truth(pool_data, output_path)
                    _, done, completed = count_progress(pool_data)
                    print(f"\n  💾 Progress disimpan! ({done}/{total_articles} artikel, {completed}/{total_queries} query selesai)")
                    if saved > 0:
                        print(f"  ✅ Ground truth ({saved} query) disimpan ke: {output_path}")
                    print(f"  👉 Jalankan ulang script ini untuk melanjutkan.")
                    return
                elif answer.startswith("j "):
                    target_qid = answer.split(" ", 1)[1].strip()
                    print(f"  → 🔀 Jump ke {target_qid} (nanti setelah simpan)")
                    # Simpan dulu, lalu skip ke query yang diminta
                    save_pool(pool_data, pool_path)
                    # Cari index query target
                    found = False
                    for tqi, tq in enumerate(pool_data["queries"]):
                        if tq["query_id"] == target_qid:
                            found = True
                            break
                    if not found:
                        print(f"  ⚠️ Query {target_qid} tidak ditemukan. Lanjut...")
                    break
                else:
                    print("  ⚠️ Input tidak valid. Gunakan y/n/s/q/j")

            # Auto-save setiap 5 artikel
            if (ai + 1) % 5 == 0:
                save_pool(pool_data, pool_path)

        # Simpan setelah selesai satu query
        save_pool(pool_data, pool_path)
        _, done, completed = count_progress(pool_data)
        print(f"\n  💾 Auto-save! Progress: {done}/{total_articles} artikel, {completed}/{total_queries} query selesai")

    # Selesai semua
    save_pool(pool_data, pool_path)
    saved = save_ground_truth(pool_data, output_path)
    print(f"\n{'=' * 65}")
    print(f"  🎉 ANOTASI SELESAI!")
    print(f"  ✅ Ground truth ({saved} query) disimpan ke: {output_path}")
    print(f"  👉 Langkah selanjutnya: jalankan 'python run_eval.py'")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    pool_path = os.path.join(script_dir, "annotation_pool.json")
    output_path = os.path.join(script_dir, "ground_truth.json")

    if not os.path.exists(pool_path):
        print("❌ File annotation_pool.json tidak ditemukan!")
        print("   Jalankan 'python create_pool.py' terlebih dahulu.")
        sys.exit(1)

    run_annotation(pool_path, output_path)
