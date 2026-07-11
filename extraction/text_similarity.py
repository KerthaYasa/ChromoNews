"""
text_similarity.py
===================
Modul similarity terpusat — pengganti pola lama yang membandingkan
teks berdasarkan potongan terbatas (mis. hanya kalimat pertama/terakhir
atau hanya 1 kalimat representatif).

Revisi (catatan dosen):
    "Gunakan word embedding dengan BERTScore. Perhitungan jangan hanya
    membandingkan kalimat pertama dan terakhir, tetapi seluruh kalimat
    dalam artikel sehingga skor kemiripan lebih akurat."

Implementasi:
    - `bertscore_similarity(text_a, text_b)` memecah KEDUA teks menjadi
      SELURUH kalimatnya (bukan hanya kalimat pertama/terakhir), lalu
      menghitung BERTScore (P/R/F1) menggunakan word embedding
      kontekstual (model multilingual BERT) antara himpunan kalimat
      tersebut. Setiap kalimat di teks A dicocokkan (greedy, dengan
      token-matching internal BERTScore) ke kalimat paling mirip di
      teks B, sehingga makna dari SELURUH artikel ikut disumbang ke
      skor akhir -- bukan cuma potongan awal/akhir.
    - Jika library `bert-score` tidak tersedia (mis. belum diinstal /
      gagal download model), fallback otomatis ke cosine similarity
      antar embedding SentenceTransformer dari seluruh kalimat
      (mean-pooled), yang tetap memperhitungkan seluruh isi teks,
      bukan hanya sepotong kalimat.
    - Fallback terakhir (tanpa embedding model sama sekali): overlap
      kata di SELURUH teks (bukan hanya sebagian kalimat).

Fungsi ini menggantikan pola lama `_is_similar_to_what()` /
`_is_similar_to()` di how_extractor.py & why_extractor.py yang hanya
membandingkan SATU kalimat kandidat vs SATU kalimat referensi.
"""

import re
from functools import lru_cache
from typing import List, Optional

from extraction.text_utils import split_sentences

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_BERTSCORE_SCORER = None
_BERTSCORE_UNAVAILABLE = False

BERTSCORE_MODEL = "bert-base-multilingual-cased"


def _get_bertscore_scorer():
    """Lazy-load BERTScorer (multilingual BERT) sekali saja per proses."""
    global _BERTSCORE_SCORER, _BERTSCORE_UNAVAILABLE
    if _BERTSCORE_UNAVAILABLE:
        return None
    if _BERTSCORE_SCORER is None:
        try:
            from bert_score import BERTScorer
            _BERTSCORE_SCORER = BERTScorer(
                model_type=BERTSCORE_MODEL,
                lang="id",
                rescale_with_baseline=False,
            )
        except Exception as e:
            print(f"[text_similarity] BERTScorer gagal dimuat, fallback ke "
                  f"cosine similarity SentenceTransformer: {e}")
            _BERTSCORE_UNAVAILABLE = True
            return None
    return _BERTSCORE_SCORER


def _sentences(text: str) -> List[str]:
    """Pecah teks jadi SELURUH kalimat (tanpa memotong ke pertama/terakhir)."""
    if not text or not text.strip():
        return []
    sents = split_sentences(text)
    sents = [s.strip() for s in sents if s and len(s.strip()) > 0]
    return sents if sents else [text.strip()]


def _word_overlap(text_a: str, text_b: str) -> float:
    """Fallback paling minimal: overlap kata di SELURUH teks (bukan sepotong)."""
    words_a = set(re.findall(r"\w{3,}", text_a.lower()))
    words_b = set(re.findall(r"\w{3,}", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    inter = len(words_a & words_b)
    return inter / max(len(words_a), len(words_b))


def _cosine_full_text(text_a: str, text_b: str, embed_model) -> float:
    """
    Fallback ke-2: encode SELURUH kalimat dari masing-masing teks lalu
    mean-pool jadi satu vektor per teks (representasi seluruh artikel,
    bukan cuma kalimat pertama/terakhir), baru dibandingkan cosine.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    sents_a = _sentences(text_a)
    sents_b = _sentences(text_b)
    if not sents_a or not sents_b:
        return 0.0

    emb_a = embed_model.encode(sents_a, convert_to_numpy=True)
    emb_b = embed_model.encode(sents_b, convert_to_numpy=True)

    vec_a = np.mean(emb_a, axis=0, keepdims=True)
    vec_b = np.mean(emb_b, axis=0, keepdims=True)

    return float(cos_sim(vec_a, vec_b)[0][0])


def bertscore_similarity(
    text_a: str,
    text_b: str,
    embed_model=None,
) -> float:
    """
    Hitung skor kemiripan antara dua teks menggunakan BERTScore atas
    SELURUH kalimat pada kedua teks (bukan hanya kalimat pertama & terakhir).

    Args:
        text_a, text_b : teks yang dibandingkan (boleh 1 kalimat atau
                          artikel penuh multi-kalimat).
        embed_model     : SentenceTransformer opsional, dipakai sebagai
                          fallback jika BERTScore tidak tersedia.

    Returns:
        float skor kemiripan pada rentang [0, 1] (F1 BERTScore jika
        library tersedia; jika tidak, cosine similarity full-text;
        jika tidak ada embedding model sama sekali, word overlap).
    """
    if not text_a or not text_b:
        return 0.0

    sents_a = _sentences(text_a)
    sents_b = _sentences(text_b)
    if not sents_a or not sents_b:
        return 0.0

    scorer = _get_bertscore_scorer()
    if scorer is not None:
        try:
            # Cocokkan tiap kalimat di A ke SEMUA kalimat di B (bukan cuma
            # kalimat pertama/terakhir) dengan menggabungkan B sebagai satu
            # referensi gabungan per kalimat A, lalu ambil rata-rata F1 dari
            # skor kalimat-per-kalimat -- ini memastikan seluruh isi kedua
            # teks ikut disumbang ke skor akhir.
            cands = []
            refs = []
            for sa in sents_a:
                best_score = None
                # Bandingkan sa terhadap setiap kalimat di B; skor akhir sa
                # adalah kemiripan tertinggi (best-matching sentence di B).
                cands.extend([sa] * len(sents_b))
                refs.extend(sents_b)
            _, _, f1 = scorer.score(cands, refs)
            f1 = f1.reshape(len(sents_a), len(sents_b))
            # Untuk tiap kalimat di A, ambil pasangan terbaik di B (greedy),
            # lalu rata-ratakan -- representasi menyeluruh, simetris dengan
            # arah A->B. Lakukan juga B->A dan rata-ratakan agar simetris.
            best_a_to_b = f1.max(dim=1).values.mean().item()

            cands2, refs2 = [], []
            for sb in sents_b:
                cands2.extend([sb] * len(sents_a))
                refs2.extend(sents_a)
            _, _, f1_2 = scorer.score(cands2, refs2)
            f1_2 = f1_2.reshape(len(sents_b), len(sents_a))
            best_b_to_a = f1_2.max(dim=1).values.mean().item()

            return float((best_a_to_b + best_b_to_a) / 2.0)
        except Exception as e:
            print(f"[text_similarity] BERTScore gagal saat scoring, fallback: {e}")

    if embed_model is not None:
        try:
            return _cosine_full_text(text_a, text_b, embed_model)
        except Exception as e:
            print(f"[text_similarity] Cosine fallback gagal: {e}")

    return _word_overlap(text_a, text_b)


def is_similar(
    text_a: str,
    text_b: str,
    embed_model=None,
    threshold: float = 0.85,
) -> bool:
    """Wrapper boolean di atas bertscore_similarity(), untuk dedup checks."""
    if not text_b:
        return False
    return bertscore_similarity(text_a, text_b, embed_model=embed_model) > threshold


# =============================================================================
# TESTING MANDIRI
# =============================================================================
if __name__ == "__main__":
    article_a = (
        "KPK menetapkan Rafael Alun sebagai tersangka korupsi. "
        "Penetapan ini didasarkan pada hasil audit keuangan selama "
        "dua tahun terakhir. Rafael diduga menerima gratifikasi dari "
        "sejumlah wajib pajak yang ia tangani."
    )
    article_b = (
        "Komisi Pemberantasan Korupsi resmi menjadikan Rafael Alun "
        "sebagai tersangka dugaan korupsi. Keputusan itu diambil "
        "setelah tim audit menelusuri rekening dan transaksi "
        "keuangannya selama dua tahun. Ia disebut menerima sejumlah "
        "gratifikasi dari beberapa wajib pajak."
    )
    article_c = (
        "Pemerintah mengumumkan kenaikan harga BBM bersubsidi mulai "
        "bulan depan. Kebijakan ini diambil untuk menekan defisit "
        "anggaran subsidi energi."
    )

    print("A vs B (mirip)   :", bertscore_similarity(article_a, article_b))
    print("A vs C (berbeda)  :", bertscore_similarity(article_a, article_c))
