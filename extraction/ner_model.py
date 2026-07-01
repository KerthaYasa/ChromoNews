"""
ner_model.py - v4
==================
Upgrade dari cahya/bert-base-indonesian-NER (3 label: PER/ORG/LOC kasar)
ke cahya/xlm-roberta-large-indonesian-NER (18 label fine-grained,
termasuk GPE/FAC/LOC terpisah -- jauh lebih cocok untuk WHO & WHERE).

Kalau resource terbatas, ganti MODEL_NAME ke versi base:
    cahya/xlm-roberta-base-indonesian-NER
(12 layer vs 24 layer di large, lebih cepat & ringan, akurasi sedikit turun)
"""

MODEL_NAME = "cahya/xlm-roberta-large-indonesian-NER"

# Label yang relevan untuk WHO
WHO_LABELS = {"PER"}

# Label yang relevan untuk WHERE, diurutkan dari prioritas tertinggi
# GPE = kota/provinsi/negara (paling reliable utk "lokasi" dalam arti umum)
# FAC = gedung/kantor/jalan/bandara (lokasi spesifik kejadian)
# LOC = kawasan/area non-administratif (gunung, sungai, dsb -- jarang dipakai
#       di berita tapi tetap valid)
WHERE_LABEL_PRIORITY = ["GPE", "FAC", "LOC"]


def load_ner_pipeline():
    """
    Loader mentah, tidak membungkus exception (sengaja, biarkan caller -
    misal @st.cache_resource di app.py - yang menangani retry & pesan error).
    """
    from transformers import pipeline
    return pipeline(
        "ner",
        model=MODEL_NAME,
        aggregation_strategy="simple",
    )


def _is_valid_span(name: str, full_text: str, start, end) -> bool:
    """
    Validasi span pasca-NER untuk menyaring potongan kata yang rusak
    (mis. pecahan handle @username, atau span yang motong di tengah kata).
    Logic dipertahankan dari versi lama -- ini sudah bagus.
    """
    if start is None or end is None:
        return True

    if start > 0:
        char_before = full_text[start - 1]
        if char_before.isalnum():
            return False

    lookback_start = max(0, start - 20)
    segment_before = full_text[lookback_start:start]
    if "@" in segment_before and " " not in segment_before.split("@")[-1]:
        return False

    return True


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 150):
    """
    Pecah teks panjang jadi beberapa window dengan overlap, supaya:
    1. Tidak melanggar limit token model (XLM-R: 512 token, ~1500-2000 char
       Bahasa Indonesia tergantung kepadatan subword).
    2. Entitas yang ada di PARUH KEDUA/AKHIR artikel tetap kebagian dilihat
       model -- ini akar masalah utama versi lama (cuma lihat 1500 char
       pertama, padahal 76% artikel di dataset >1500 char).
    3. Overlap mencegah entity di batas potongan ke-cut jadi 2 fragmen rusak.

    Returns: List[(chunk_text, global_offset)]
    """
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((text[start:end], start))
        if end == n:
            break
        start = end - overlap
    return chunks


def _merge_adjacent_entities(entities: list, original_text: str) -> list:
    """
    PERBAIKAN BUG: aggregation_strategy="simple" kadang gagal menyatukan
    span yang seharusnya 1 entity utuh, akibat tokenizer SentencePiece
    memecah kata di tengah subword. Contoh nyata yang ditemukan saat testing:

        "Istana Negara" -> [LOC "Is" (54-56), LOC "tana Negara" (56-67)]

    Padahal end span pertama PERSIS sama dengan start span kedua (56==56),
    artinya tidak ada spasi/separator di antara keduanya di teks asli --
    ini tanda kuat keduanya adalah SATU entity yang terpecah, bukan dua
    entity berbeda yang kebetulan bersebelahan.

    Strategi: kalau entity[i].end == entity[i+1].start (bersambungan TANPA
    gap), dan tidak ada spasi di antara mereka di teks asli, gabungkan jadi
    1 entity. Label diambil dari yang confidence-nya lebih tinggi, score
    di-rata-rata.

    PENTING: fungsi ini harus dipanggil SEBELUM _is_valid_span, supaya
    span yang sudah digabung tidak keburu ditolak oleh validator (yang
    justru didesain menolak span pecahan).
    """
    if not entities:
        return entities

    entities_sorted = sorted(
        [e for e in entities if e["start"] is not None],
        key=lambda e: e["start"]
    )

    merged = []
    i = 0
    while i < len(entities_sorted):
        current = dict(entities_sorted[i])
        j = i + 1
        while j < len(entities_sorted):
            nxt = entities_sorted[j]
            # Bersambungan persis tanpa gap (tidak ada spasi/separator di antaranya)
            if nxt["start"] == current["end"]:
                current["text"] = current["text"] + nxt["text"]
                current["end"] = nxt["end"]
                # pakai score yang lebih tinggi sebagai representasi confidence gabungan
                current["score"] = max(current["score"], nxt["score"])
                # kalau label beda, pertahankan label dari fragmen confidence tertinggi
                if nxt["score"] > entities_sorted[i]["score"]:
                    current["label"] = nxt["label"]
                j += 1
            else:
                break
        merged.append(current)
        i = j

    return merged


def run_ner_chunked(pipe, content: str, chunk_size: int = 1500, overlap: int = 150):
    """
    Jalankan NER ke SELURUH teks (bukan cuma 1500 char pertama) lewat chunking,
    lalu kembalikan list entity ter-normalisasi dengan offset GLOBAL
    (relatif ke `content` penuh, bukan relatif ke chunk).

    Returns: List[dict] dengan keys: text, label, start, end, score
    """
    if pipe is None or not content:
        return []

    all_entities = []
    for chunk, offset in chunk_text(content, chunk_size, overlap):
        try:
            results = pipe(chunk)
        except Exception:
            continue

        chunk_entities = []
        for ent in results:
            label = ent.get("entity_group", "")
            start, end = ent.get("start"), ent.get("end")

            if start is not None and end is not None:
                name = chunk[start:end].strip()
                global_start = offset + start
                global_end = offset + end
            else:
                name = ent.get("word", "").strip()
                global_start = global_end = None

            if not name or name.startswith("##"):
                continue

            chunk_entities.append({
                "text": name,
                "label": label,
                "start": global_start,
                "end": global_end,
                "score": float(ent.get("score", 0.0)),
            })

        # Gabungkan dulu fragmen yang bersambungan SEBELUM validasi span
        chunk_entities = _merge_adjacent_entities(chunk_entities, chunk)

        for e in chunk_entities:
            if len(e["text"]) < 2:
                continue
            local_start = e["start"] - offset if e["start"] is not None else None
            local_end = e["end"] - offset if e["end"] is not None else None
            if not _is_valid_span(e["text"], chunk, local_start, local_end):
                continue
            all_entities.append(e)

    return all_entities
