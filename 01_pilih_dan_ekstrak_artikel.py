"""
01_pilih_dan_ekstrak_artikel.py
================================
LANGKAH 1 dari pipeline validasi 5W1H.

Tujuan:
- Menjalankan 25 query yang MEWAKILI variasi topik dataset (bukan asal
  tebak) lewat hybrid search (BM25 + semantic) yang sudah ada di project.
- Mengambil TOP-2 artikel per query -> sampel awal max 50 artikel.
- Dedup: kalau ada artikel yang muncul di >1 query, dia cuma dihitung
  sekali (supaya sampel akhir tidak overcount artikel populer).
- Menjalankan extract_5w1h() pada tiap artikel terpilih.
- Menyimpan hasil ke 'artikel_untuk_penilaian.csv' -> jadi input untuk
  Langkah 2 (generate template Excel penilaian).

CATATAN PENTING:
- Jalankan file ini dari root folder project (folder yang berisi app.py),
  supaya semua import "extraction.xxx" konsisten.
- 25 query di bawah ini SAYA SUSUN berdasarkan sebaran topik yang saya
  temukan di dataset kamu (politik/pilpres, hukum & korupsi, ekonomi,
  sosial-agama, internasional, olahraga, teknologi/bisnis). SILAKAN
  SESUAIKAN kalau menurutmu ada topik besar di dataset yang belum
  terwakili -- jangan asal terima daftar ini mentah-mentah.
"""

import pandas as pd
from pathlib import Path

from preprocess import preprocess_for_bm25
from bm25_search import build_bm25_index, search_bm25
from semantic_search import load_embedding_model, encode_corpus, search_semantic
from hybrid_search import reciprocal_rank_fusion
from extraction.rule_based_5w1h import extract_5w1h

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "preprocessed_news_sample.csv"   # sumber yang BERSIH
OUTPUT_FILE = BASE_DIR / "artikel_untuk_penilaian.csv"

TOP_K_PER_QUERY = 2   # sesuai rencana: 2 artikel terbaik per query

# =============================================================================
# 25 QUERY -- disusun manual untuk mewakili sebaran topik dataset.
# Ganti/tambah sesuai kebutuhanmu, tapi JANGAN semuanya dari 1 topik saja.
# =============================================================================
QUERIES = [
    # Politik & Pilpres
    "pilkada indonesia 2024",
    "simulasi pilpres capres cawapres",
    "koalisi partai politik",
    "kampanye pemilu",
    "survei elektabilitas capres",
    # Hukum & Korupsi
    "korupsi KPK",
    "rafael alun pajak",
    "kasus pencucian uang",
    "operasi tangkap tangan",
    "sidang tersangka korupsi",
    # Ekonomi & Bisnis
    "kenaikan harga BBM",
    "kolapsnya bank Silicon Valley Bank",
    "investasi asing Indonesia",
    "pendapatan perusahaan kuartal",
    "inflasi dan harga pangan",
    # Sosial & Agama
    "perayaan hari besar keagamaan",
    "bantuan sosial pemerintah",
    "pendidikan dan kampus",
    # Internasional
    "hubungan Indonesia dengan negara lain",
    "konflik internasional",
    "kunjungan pejabat asing",
    # Olahraga
    "kualifikasi piala dunia sepak bola",
    "prestasi atlet Indonesia",
    # Teknologi & Infrastruktur
    "pembangunan IKN",
    "transportasi dan mudik",
]

assert len(QUERIES) == 25, f"Harus 25 query, sekarang ada {len(QUERIES)}"


def main():
    print("Memuat dataset (preprocessed_news_sample.csv)...")
    df = pd.read_csv(DATASET_FILE)
    print(f"-> {len(df):,} artikel dimuat")

    print("\nMembangun BM25 index...")
    tokenized_corpus = [str(doc).split() for doc in df["processed_content"]]
    bm25_index = build_bm25_index(tokenized_corpus)

    print("Memuat model semantic search (ini bisa makan waktu / butuh internet)...")
    model = load_embedding_model()
    corpus_embeddings = encode_corpus(model, df["content"].tolist())

    selected_rows = {}   # doc_idx -> {"query": ..., "rank": ...}

    for q in QUERIES:
        q_bm25 = preprocess_for_bm25(q)
        bm25_results = search_bm25(q_bm25, bm25_index, top_k=20)
        sem_results = search_semantic(q, model, corpus_embeddings, top_k=20)
        hybrid_results = reciprocal_rank_fusion(bm25_results, sem_results, top_k=TOP_K_PER_QUERY)

        for rank, (doc_idx, score) in enumerate(hybrid_results, start=1):
            if doc_idx not in selected_rows:
                selected_rows[doc_idx] = {"query": q, "rank": rank, "rrf_score": score}
            # kalau doc_idx sudah pernah kepilih dari query lain -> dedup,
            # tetap simpan query PERTAMA yang menemukannya, tidak dobel.

    print(f"\nTotal artikel unik terpilih: {len(selected_rows)} "
          f"(dari maksimum {len(QUERIES) * TOP_K_PER_QUERY} slot, "
          f"selisihnya adalah overlap antar query)")

    if len(selected_rows) < 40:
        print("PERINGATAN: jumlah artikel unik jauh di bawah 50. "
              "Overlap antar query terlalu tinggi -- pertimbangkan query "
              "yang lebih beragam / tambah beberapa query baru.")

    print("\nMenjalankan ekstraksi 5W1H untuk tiap artikel terpilih...")
    out_rows = []
    for doc_idx, info in selected_rows.items():
        row = df.iloc[doc_idx]
        article = {
            "title": row["title"],
            "content": row["content"],
            "date": row["date"],
        }
        result = extract_5w1h(article)

        out_rows.append({
            "artikel_id": int(doc_idx),
            "query_sumber": info["query"],
            "title": row["title"],
            "date": row["date"],
            "content": row["content"],
            "extracted_who": "; ".join(result["who"]) if isinstance(result["who"], list) else result["who"],
            "extracted_what": result["what"],
            "extracted_when": "; ".join(result["when"]) if isinstance(result["when"], list) else result["when"],
            "extracted_where": "; ".join(result["where"]) if isinstance(result["where"], list) else result["where"],
            "extracted_why": result["why"],
            "extracted_how": result["how"],
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSelesai. Disimpan ke: {OUTPUT_FILE}")
    print(f"Jumlah artikel final untuk divalidasi: {len(out_df)}")
    print("\nLangkah selanjutnya: jalankan 02_buat_template_penilaian.py")


if __name__ == "__main__":
    main()
