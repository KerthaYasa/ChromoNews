"""
create_pool.py — Step 1: Generate pool anotasi untuk ground truth

Jalankan ketiga metode pencarian pada 25 query uji,
gabungkan (union) top-10 dari masing-masing metode,
dan simpan sebagai annotation_pool.json untuk proses anotasi manual.
"""

import os
import sys
import json
import pandas as pd

# Fix encoding untuk Windows (cp1252 tidak support emoji)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Tambahkan parent directory agar bisa import modul dari root project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import preprocess_for_bm25
from bm25_search import build_bm25_index, search_bm25
from semantic_search import load_embedding_model, encode_corpus, search_semantic
from hybrid_search import reciprocal_rank_fusion

# ============================================================================
# DEFINISI 25 QUERY UJI
# Dikategorikan berdasarkan tipe query untuk analisis per kategori
# ============================================================================
QUERIES = [
    # --- Kategori 1: Keyword Spesifik (7 query) ---
    {"query_id": "q01", "query_text": "kasus korupsi KPK", "category": "keyword_spesifik"},
    {"query_id": "q02", "query_text": "mudik lebaran 2023", "category": "keyword_spesifik"},
    {"query_id": "q03", "query_text": "Rafael Alun pajak", "category": "keyword_spesifik"},
    {"query_id": "q04", "query_text": "harga emas hari ini", "category": "keyword_spesifik"},
    {"query_id": "q05", "query_text": "Piala Dunia U-20 Indonesia", "category": "keyword_spesifik"},
    {"query_id": "q06", "query_text": "Jokowi IKN Nusantara", "category": "keyword_spesifik"},
    {"query_id": "q07", "query_text": "THR pegawai swasta 2023", "category": "keyword_spesifik"},

    # --- Kategori 2: Natural Language (7 query) ---
    {"query_id": "q08", "query_text": "dampak ekonomi dari kebangkrutan Silicon Valley Bank terhadap pasar global", "category": "natural_language"},
    {"query_id": "q09", "query_text": "bagaimana persiapan pemerintah menghadapi arus mudik lebaran tahun ini", "category": "natural_language"},
    {"query_id": "q10", "query_text": "apa penyebab tragedi Kanjuruhan dan siapa yang bertanggung jawab", "category": "natural_language"},
    {"query_id": "q11", "query_text": "mengapa harta kekayaan pejabat pajak menjadi sorotan publik", "category": "natural_language"},
    {"query_id": "q12", "query_text": "perkembangan terbaru kasus penembakan di Papua oleh kelompok bersenjata", "category": "natural_language"},
    {"query_id": "q13", "query_text": "kebijakan pemerintah dalam menangani krisis perbankan global 2023", "category": "natural_language"},
    {"query_id": "q14", "query_text": "perselisihan politik terkait pemilihan ketua Mahkamah Konstitusi", "category": "natural_language"},

    # --- Kategori 3: Sinonim / Istilah Tidak Eksak (6 query) ---
    {"query_id": "q15", "query_text": "pekerja informal susah dapat pinjaman bank", "category": "sinonim"},
    {"query_id": "q16", "query_text": "kecelakaan sepak bola stadion Malang", "category": "sinonim"},
    {"query_id": "q17", "query_text": "koruptor pegawai negeri harta mewah", "category": "sinonim"},
    {"query_id": "q18", "query_text": "bencana banjir Jakarta warga mengungsi", "category": "sinonim"},
    {"query_id": "q19", "query_text": "kompetisi olahraga internasional ditunda", "category": "sinonim"},
    {"query_id": "q20", "query_text": "kenaikan tarif bahan bakar minyak masyarakat protes", "category": "sinonim"},

    # --- Kategori 4: Typo Ringan (5 query) ---
    {"query_id": "q21", "query_text": "korupsi KPK tersangka suab", "category": "typo"},
    {"query_id": "q22", "query_text": "Jokowii resmikan proyek infrastuktur", "category": "typo"},
    {"query_id": "q23", "query_text": "haga emas antam turun", "category": "typo"},
    {"query_id": "q24", "query_text": "pemilu presden 2024 kampanye", "category": "typo"},
    {"query_id": "q25", "query_text": "mudik lebaraan kereta api tiket", "category": "typo"},
]


def create_annotation_pool(data_path, output_path, pool_top_k=10):
    """
    Generate annotation pool untuk setiap query.
    
    Untuk setiap query:
    1. Jalankan BM25 → ambil top-{pool_top_k}
    2. Jalankan Semantic → ambil top-{pool_top_k}
    3. Jalankan Hybrid (RRF dari top-20 masing-masing) → ambil top-{pool_top_k}
    4. Gabungkan (union) semua artikel unik → jadi pool anotasi
    """
    # --- Load dataset ---
    print("=" * 60)
    print("STEP 1: GENERATE ANNOTATION POOL")
    print("=" * 60)
    
    print("\n📄 Memuat dataset...")
    df = pd.read_csv(data_path)
    print(f"   Dataset berisi {len(df)} artikel")

    # --- Bangun index BM25 ---
    print("\n🔍 Membangun BM25 index...")
    tokenized_corpus = [str(doc).split() for doc in df['processed_content']]
    bm25_index = build_bm25_index(tokenized_corpus)
    print("   BM25 index siap!")

    # --- Load model Semantic ---
    print("\n🧠 Memuat model Semantic Search...")
    model = load_embedding_model()
    corpus_embeddings = encode_corpus(model, df['content'].tolist())
    print("   Model semantic siap!")

    # --- Proses setiap query ---
    print(f"\n🏊 Memproses {len(QUERIES)} query...\n")
    
    all_queries = []
    total_pool_size = 0

    for i, q in enumerate(QUERIES):
        query_text = q["query_text"]
        query_processed = preprocess_for_bm25(query_text)
        
        # BM25 top-K
        bm25_results = search_bm25(query_processed, bm25_index, top_k=pool_top_k)
        bm25_ids = {idx for idx, _ in bm25_results}
        
        # Semantic top-K
        sem_results = search_semantic(query_text, model, corpus_embeddings, top_k=pool_top_k)
        sem_ids = {idx for idx, _ in sem_results}
        
        # Hybrid (ambil top-20 lalu fusi, ambil top-K)
        bm25_20 = search_bm25(query_processed, bm25_index, top_k=20)
        sem_20 = search_semantic(query_text, model, corpus_embeddings, top_k=20)
        hybrid_results = reciprocal_rank_fusion(bm25_20, sem_20, top_k=pool_top_k)
        hybrid_ids = {idx for idx, _ in hybrid_results}
        
        # Union semua artikel unik
        all_ids = bm25_ids | sem_ids | hybrid_ids
        
        # Buat pool dengan detail artikel
        pool = []
        for article_id in sorted(all_ids):
            row = df.iloc[article_id]
            content_str = str(row['content'])
            pool.append({
                "article_id": int(article_id),
                "title": str(row['title']),
                "date": str(row.get('date', '')),
                "content": content_str,
                "found_by": {
                    "bm25": article_id in bm25_ids,
                    "semantic": article_id in sem_ids,
                    "hybrid": article_id in hybrid_ids
                },
                "is_relevant": None  # Akan diisi manual oleh user
            })
        
        all_queries.append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "category": q["category"],
            "pool_size": len(pool),
            "pool": pool
        })
        
        total_pool_size += len(pool)
        print(f"  [{i+1:2d}/25] {q['query_id']} | \"{query_text[:40]}{'...' if len(query_text) > 40 else ''}\"")
        print(f"          BM25: {len(bm25_ids)} | Semantic: {len(sem_ids)} | Hybrid: {len(hybrid_ids)} | Pool: {len(pool)} artikel unik")

    # --- Simpan hasil ---
    output_data = {
        "description": "Annotation pool untuk evaluasi metode pencarian ChromoNews",
        "total_queries": len(all_queries),
        "total_articles_to_annotate": total_pool_size,
        "queries": all_queries
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ Annotation pool berhasil dibuat!")
    print(f"   File: {output_path}")
    print(f"   Total query: {len(all_queries)}")
    print(f"   Total artikel yang perlu dianotasi: {total_pool_size}")
    print(f"   Rata-rata pool per query: {total_pool_size / len(all_queries):.1f} artikel")
    print(f"\n👉 Langkah selanjutnya: jalankan 'python annotate_helper.py'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    data_path = os.path.join(project_root, "preprocessed_news_sample.csv")
    output_path = os.path.join(script_dir, "annotation_pool.json")
    
    create_annotation_pool(data_path, output_path)
