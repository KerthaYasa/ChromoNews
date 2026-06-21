"""
ner_model.py
============
Wrapper untuk model NER terlatih (cahya/bert-base-indonesian-NER) untuk
deteksi MULTI-ENTITAS WHO (PER + ORG) secara native.

REVISI (v3): loader dipisah dari inference, supaya pipeline bisa di-cache
SEKALI di level Streamlit (`@st.cache_resource` di app.py) dan dipakai
ulang ("di-inject") ke semua pemanggilan, BUKAN lazy-singleton tersembunyi
di modul ini. Alasan migrasi dari pola lama (singleton manual):

  1. PERFORMA: load_ner_pipeline() dipanggil SEKALI per sesi app (bukan
     dicoba ulang diam-diam tiap kali dibutuhkan) -- run berikutnya pakai
     resource yang sudah ada di cache Streamlit, tidak reload dari nol.
  2. RETRY OTOMATIS YANG BENAR: kalau loading gagal (exception), Streamlit
     `@st.cache_resource` TIDAK menyimpan exception ke cache -- jadi kalau
     kamu baru saja install dependency yang kurang, rerun berikutnya akan
     OTOMATIS mencoba lagi tanpa perlu restart total proses Streamlit
     (beda dengan singleton manual versi sebelumnya yang mengunci status
     gagal selama proses hidup).
  3. TESTABLE: extract_who_multi() jadi fungsi murni (pure function) yang
     terima pipeline sebagai parameter -- gampang di-unit-test tanpa perlu
     mocking state global.

Model dilatih di dataset id_nergrit_corpus, label yang relevan: PER
(person), ORG (organisasi). WHERE & WHEN tetap rule-based (lihat
hybrid_5w1h.py) sesuai keputusan desain: model hanya untuk WHO.
"""

TARGET_LABELS = {"PER", "ORG"}


def load_ner_pipeline():
    """
    Loader MENTAH -- TIDAK membungkus exception (sengaja).
    Dipanggil dari app.py lewat @st.cache_resource, supaya Streamlit yang
    mengatur caching & retry behavior (lihat docstring modul di atas).

    Returns:
        transformers.Pipeline

    Raises:
        Exception apapun yang terjadi saat load (network error, dependency
        belum lengkap/sentencepiece hilang, dst) -- caller (app.py)
        WAJIB tangkap dengan try/except dan tampilkan pesannya ke user.
    """
    from transformers import pipeline
    return pipeline(
        "ner",
        model="cahya/bert-base-indonesian-NER",
        aggregation_strategy="simple",
    )


def _is_valid_person_org_span(name, full_text, start, end):
    """
    Validasi TAMBAHAN pasca-NER untuk menyaring span yang JELAS rusak.

    KONTEKS BUG: aggregation_strategy="simple" kadang menghasilkan span
    yang memotong DI TENGAH kata, khususnya di sekitar token yang asing
    buat tokenizer WordPiece seperti handle media sosial ("@username").
    Contoh nyata: dari teks "...akun Instagramnya @prasetyoedimarsudi..."
    model bisa mengembalikan span "yoedimarsudi" -- potongan tengah dari
    username, BUKAN nama valid. Guard `name.startswith("##")` yang sudah
    ada TIDAK menangkap ini, karena aggregation_strategy sudah membuang
    prefix "##" sebelum span dikembalikan -- yang rusak adalah BATAS
    span-nya (start/end), bukan token mentahnya.

    Heuristik deteksi:
    1. Kalau karakter TEPAT SEBELUM span adalah huruf/angka (bukan spasi,
       tanda baca, atau awal teks) -> span ini memotong di tengah sebuah
       "kata" yang lebih panjang (mis. nama domain, username, atau kata
       majemuk tanpa spasi) -> TOLAK.
    2. Kalau ada karakter '@' dalam jarak pendek SEBELUM span (cek 15
       karakter ke belakang) tanpa spasi pemisah -> ini pecahan dari
       handle media sosial, bukan nama orang/organisasi -> TOLAK.
    """
    if start is None or end is None:
        return True  # tidak ada info posisi, tidak bisa divalidasi -- lolos saja

    # Cek 1: karakter sebelum span adalah huruf/angka -> span memotong kata
    if start > 0:
        char_before = full_text[start - 1]
        if char_before.isalnum():
            return False

    # Cek 2: ada '@' tanpa spasi pemisah dalam jarak pendek ke belakang
    lookback_start = max(0, start - 20)
    segment_before = full_text[lookback_start:start]
    if "@" in segment_before and " " not in segment_before.split("@")[-1]:
        return False

    return True


def extract_who_multi(pipe, content, title="", max_entities=3):
    """
    Mengekstrak hingga `max_entities` nama orang/organisasi unik dari teks.

    Args:
        pipe: hasil load_ner_pipeline() yang sudah di-cache (BUKAN None --
              kalau model tidak tersedia, caller (hybrid_5w1h.py) yang
              menentukan untuk tidak memanggil fungsi ini sama sekali dan
              langsung pakai fallback heuristik)

    Returns:
        List[str] kandidat WHO, atau None kalau tidak ada entitas terdeteksi.
    """
    if pipe is None:
        return None

    text_for_ner = content[:1500] if content else ""
    if not text_for_ner.strip():
        return None

    try:
        entities = pipe(text_for_ner)
    except Exception:
        return None

    candidates = []
    seen_lower = set()
    for ent in entities:
        label = ent.get("entity_group", "")
        if label not in TARGET_LABELS:
            continue

        start, end = ent.get("start"), ent.get("end")
        if start is not None and end is not None:
            name = text_for_ner[start:end].strip()
        else:
            name = ent.get("word", "").strip()

        if len(name) < 3 or name.startswith("##"):
            continue

        # Saring span yang jelas rusak (lihat docstring _is_valid_person_org_span)
        if not _is_valid_person_org_span(name, text_for_ner, start, end):
            continue

        # Saring nama 1 kata yang TERLALU pendek untuk nama orang utuh
        # (mis. "Ali" tanpa nama belakang menyusul) -- nama orang Indonesia
        # di berita hampir selalu disebut minimal 2 kata saat pertama kali
        # diperkenalkan. Single-word PER dengan panjang <8 karakter SANGAT
        # mungkin entity merging gagal menyatukan dengan kata berikutnya,
        # bukan benar-benar nama 1 kata (beda dgn ORG yg wajar 1 kata, mis "KPK").
        if label == "PER" and " " not in name and len(name) < 8:
            continue

        key = name.lower()
        if key in seen_lower:
            continue
        if any(key in s or s in key for s in seen_lower):
            continue
        seen_lower.add(key)
        candidates.append(name)

    if not candidates:
        return None

    return candidates[:max_entities]
