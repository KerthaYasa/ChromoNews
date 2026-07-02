"""
run_eval.py — Step 3: Jalankan ketiga metode pencarian pada semua query

Membaca ground_truth.json, menjalankan BM25, Semantic, dan Hybrid search
untuk setiap query, lalu menyimpan top-5 hasil ke file JSON terpisah.
"""

import os
import sys
import json
import time
import pandas as pd

# Fix encoding untuk Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Tambahkan parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import preprocess_for_bm25
from bm25_search import build_bm25_index, search_bm25
from semantic_search import load_embedding_model, encode_corpus, search_semantic
from hybrid_search import reciprocal_rank_fusion


def load_ground_truth(gt_path):
    """Load ground truth queries."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_evaluation(data_path, gt_path, results_dir, top_k=5, rrf_k=60):
    """
    Jalankan evaluasi ketiga metode pencarian.
    
    Args:
        data_path: Path ke preprocessed_news_sample.csv
        gt_path: Path ke ground_truth.json
        results_dir: Direktori untuk menyimpan hasil
        top_k: Jumlah hasil teratas per query (default: 5)
        rrf_k: Smoothing constant untuk RRF (default: 60)
    """
    print("=" * 60)
    print("STEP 3: MENJALANKAN EVALUASI KETIGA METODE PENCARIAN")
    print("=" * 60)

    # --- Load dataset ---
    print("\n📄 Memuat dataset...")
    df = pd.read_csv(data_path)
    print(f"   {len(df)} artikel dimuat")

    # --- Load ground truth ---
    print("\n📋 Memuat ground truth...")
    gt_data = load_ground_truth(gt_path)
    queries = gt_data["queries"]
    print(f"   {len(queries)} query dimuat")

    # --- Bangun index & model ---
    print("\n🔍 Membangun BM25 index...")
    tokenized_corpus = [str(doc).split() for doc in df['processed_content']]
    bm25_index = build_bm25_index(tokenized_corpus)

    print("🧠 Memuat model Semantic Search...")
    model = load_embedding_model()
    corpus_embeddings = encode_corpus(model, df['content'].tolist())

    # --- Jalankan pencarian ---
    print(f"\n🏃 Menjalankan pencarian (top-{top_k}) untuk {len(queries)} query...\n")

    results_bm25 = {}
    results_semantic = {}
    results_hybrid = {}

    total_start = time.time()

    for i, q in enumerate(queries):
        query_text = q["query_text"]
        query_id = q["query_id"]
        query_processed = preprocess_for_bm25(query_text)

        # --- BM25 ---
        bm25_res = search_bm25(query_processed, bm25_index, top_k=top_k)
        results_bm25[query_id] = {
            "query_text": query_text,
            "category": q["category"],
            "relevant_article_ids": q["relevant_article_ids"],
            "results": [
                {
                    "rank": rank + 1,
                    "article_id": int(doc_idx),
                    "score": float(score),
                    "title": str(df['title'].iloc[doc_idx])
                }
                for rank, (doc_idx, score) in enumerate(bm25_res)
            ]
        }

        # --- Semantic ---
        sem_res = search_semantic(query_text, model, corpus_embeddings, top_k=top_k)
        results_semantic[query_id] = {
            "query_text": query_text,
            "category": q["category"],
            "relevant_article_ids": q["relevant_article_ids"],
            "results": [
                {
                    "rank": rank + 1,
                    "article_id": int(doc_idx),
                    "score": float(score),
                    "title": str(df['title'].iloc[doc_idx])
                }
                for rank, (doc_idx, score) in enumerate(sem_res)
            ]
        }

        # --- Hybrid (ambil top-20 dari masing-masing, lalu fusi ke top-K) ---
        bm25_20 = search_bm25(query_processed, bm25_index, top_k=20)
        sem_20 = search_semantic(query_text, model, corpus_embeddings, top_k=20)
        hybrid_res = reciprocal_rank_fusion(bm25_20, sem_20, k=rrf_k, top_k=top_k)
        results_hybrid[query_id] = {
            "query_text": query_text,
            "category": q["category"],
            "relevant_article_ids": q["relevant_article_ids"],
            "results": [
                {
                    "rank": rank + 1,
                    "article_id": int(doc_idx),
                    "score": float(score),
                    "title": str(df['title'].iloc[doc_idx])
                }
                for rank, (doc_idx, score) in enumerate(hybrid_res)
            ]
        }

        # Progress
        bm25_retrieved = [r["article_id"] for r in results_bm25[query_id]["results"]]
        sem_retrieved = [r["article_id"] for r in results_semantic[query_id]["results"]]
        hyb_retrieved = [r["article_id"] for r in results_hybrid[query_id]["results"]]
        relevant = set(q["relevant_article_ids"])

        bm25_hits = len(set(bm25_retrieved) & relevant)
        sem_hits = len(set(sem_retrieved) & relevant)
        hyb_hits = len(set(hyb_retrieved) & relevant)

        print(f"  [{i+1:2d}/{len(queries)}] {query_id} | \"{query_text[:45]}{'...' if len(query_text) > 45 else ''}\"")
        print(f"           Hits → BM25: {bm25_hits}/{len(relevant)} | Semantic: {sem_hits}/{len(relevant)} | Hybrid: {hyb_hits}/{len(relevant)}")

    total_time = time.time() - total_start

    # --- Simpan hasil ---
    os.makedirs(results_dir, exist_ok=True)

    for filename, data in [
        ("results_bm25.json", results_bm25),
        ("results_semantic.json", results_semantic),
        ("results_hybrid.json", results_hybrid),
    ]:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ Evaluasi selesai dalam {total_time:.2f} detik!")
    print(f"   Hasil disimpan ke:")
    print(f"   📁 {os.path.join(results_dir, 'results_bm25.json')}")
    print(f"   📁 {os.path.join(results_dir, 'results_semantic.json')}")
    print(f"   📁 {os.path.join(results_dir, 'results_hybrid.json')}")
    print(f"\n👉 Langkah selanjutnya: jalankan 'python compute_metrics.py'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_path = os.path.join(project_root, "preprocessed_news_sample.csv")
    gt_path = os.path.join(script_dir, "ground_truth.json")
    results_dir = os.path.join(script_dir, "results")

    if not os.path.exists(gt_path):
        print("❌ File ground_truth.json tidak ditemukan!")
        print("   Jalankan 'python annotate_helper.py' terlebih dahulu.")
        sys.exit(1)

    run_evaluation(data_path, gt_path, results_dir)
