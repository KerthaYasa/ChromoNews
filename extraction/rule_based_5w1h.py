"""
rule_based_5w1h.py
===================
Modul UTAMA untuk ekstraksi 5W1H secara ALGORITMIK (manual rule-based).

REVISI v3: Hasil setiap elemen 5W1H dibuat lebih ringkas dan tepat sasaran.
Menghindari mengembalikan kalimat utuh yang terlalu panjang/tidak relevan.
Format hasil mengikuti pola AI summarizer:
  - WHO: "Jabatan Nama, Jabatan Nama, dan Jabatan Nama"
  - WHEN: "Peristiwa terjadi pada <tanggal>"
  - WHERE: "Peristiwa terjadi di <lokasi>"
  - WHY: Kalimat penjelas penyebab (1-2 kalimat)
  - HOW: Kalimat penjelas cara/proses (1-2 kalimat)

Fungsi entry point: extract_5w1h(article) -> dict
"""

import re
from datetime import datetime

from .patterns import (
    DATE_PATTERNS,
    CAUSAL_CONNECTORS,
    PURPOSE_CONNECTORS,
    METHOD_CONNECTORS,
    HOW_FALLBACK_CONNECTORS,
    REPORTING_VERBS,
    DATELINE_PATTERN,
    DATELINE_SIMPLE_PATTERN,
    TITLE_PREFIXES,
    INTER_SENTENCE_CAUSAL,
    METHOD_VERB_PATTERN,
    SCRAPING_ARTIFACTS
)
from .gazetteer import find_location_near_dateline, find_locations_in_text, load_locations
from .ner_helper import find_who, find_who_multi

NOT_FOUND = "Tidak disebutkan dalam artikel"

_BULAN_MAP = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}


# =============================================================================
# UTILITAS
# =============================================================================
def _split_sentences(text):
    """Pecah teks menjadi kalimat-kalimat, termasuk menangani kasus scraping
    di mana kalimat sering menempel tanpa spasi (contoh: 'kata1.Kata2')."""
    # Sisipkan newline sebelum huruf besar yang menempel setelah titik
    text = re.sub(r'([a-z0-9\"\'\)])(\.)([A-Z])', r'\1.\n\3', text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _strip_dateline(content):
    """Membersihkan prefix dateline media dari awal artikel."""
    match = DATELINE_PATTERN.match(content)
    if match:
        cand1 = match.group(1).strip()
        cand2 = match.group(2).strip()
        if ".CO" in cand2.upper() or "COM" in cand2.upper() or "INDONESIA" in cand2.upper():
            location = cand1
        else:
            location = cand2
        clean = content[match.end():].strip()
        return clean, location

    match = DATELINE_SIMPLE_PATTERN.match(content)
    if match:
        clean = content[match.end():].strip()
        return clean, None

    return content, None


def _format_metadata_date(date_str):
    """Format ulang tanggal dari kolom metadata (ISO 8601) -> 'DD Bulan YYYY'."""
    try:
        cleaned = date_str.replace("+00", "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(cleaned[:19], fmt)
                return f"{dt.day} {_BULAN_MAP[dt.month]} {dt.year}"
            except ValueError:
                continue
    except Exception:
        pass
    return date_str


def _extract_clause_after_connector(text, connectors, max_len=250,
                                     skip_after_reporting_verb=False):
    """
    Mencari konektor dalam teks dan mengembalikan KLAUSA setelah konektor
    tersebut hingga tanda titik/akhir kalimat berikutnya.
    Hasilnya adalah potongan teks yang ringkas dan relevan.
    """
    text_lower = text.lower()
    all_occurrences = []
    for conn in connectors:
        start_pos = 0
        while True:
            pos = text_lower.find(conn.lower(), start_pos)
            if pos == -1:
                break
            all_occurrences.append((pos, conn))
            start_pos = pos + len(conn)
    all_occurrences.sort(key=lambda x: x[0])

    for pos, conn in all_occurrences:
        if skip_after_reporting_verb:
            preceding = text_lower[max(0, pos - 30):pos]
            if any(rv in preceding for rv in REPORTING_VERBS):
                continue

        start = pos + len(conn)
        remainder = text[start:start + max_len]
        remainder_stripped = remainder.strip()

        # Skip jika diikuti kata ganti demonstratif (bukan klausa informatif)
        if re.match(r"^\s*(itu|tersebut)\b", remainder_stripped, re.IGNORECASE):
            continue

        # Cari akhir kalimat
        end_match = re.search(r'[.!?]', remainder)
        if end_match:
            clause = remainder[:end_match.end()]  # termasuk tanda titiknya
        else:
            clause = remainder
        clause = clause.strip(" ,:-\"")

        if clause and len(clause) >= 10:
            return clause

    return None


def _extract_sentence_containing_connector(text, connectors, sentences=None,
                                            skip_after_reporting_verb=False,
                                            max_sentence_len=250):
    """
    Mencari konektor dan mengembalikan KALIMAT UTUH yang mengandung konektor
    tersebut, HANYA jika kalimatnya cukup pendek dan informatif.
    Jika kalimat terlalu panjang, fallback ke klausa saja.
    """
    if sentences is None:
        sentences = _split_sentences(text)
    text_lower = text.lower()

    for conn in connectors:
        start_pos = 0
        while True:
            pos = text_lower.find(conn.lower(), start_pos)
            if pos == -1:
                break
            start_pos = pos + len(conn)

            if skip_after_reporting_verb:
                preceding = text_lower[max(0, pos - 30):pos]
                if any(rv in preceding for rv in REPORTING_VERBS):
                    continue

            # Validasi klausa setelah konektor
            remainder = text[pos + len(conn):pos + len(conn) + 200].strip()
            if re.match(r"^\s*(itu|tersebut)\b", remainder, re.IGNORECASE):
                continue

            # Cari kalimat utuh yang mengandung posisi ini
            current_pos = 0
            for sent in sentences:
                idx = text.find(sent, current_pos)
                if idx == -1:
                    continue
                if idx <= pos < idx + len(sent):
                    trimmed = sent.strip()
                    if len(trimmed) >= 15:
                        if len(trimmed) <= max_sentence_len:
                            return trimmed
                        else:
                            # Kalimat terlalu panjang, ambil klausa saja
                            clause = _extract_clause_after_connector(
                                sent, [conn], max_len=250,
                                skip_after_reporting_verb=skip_after_reporting_verb
                            )
                            if clause:
                                return clause
                    break
                current_pos = idx + len(sent)

    return None


# =============================================================================
# FILTER WHO
# =============================================================================
_WHO_EXCLUDE_KEYWORDS = [
    "ktt", "konferensi", "summit", "festival", "rapat", "sidang",
    "pemilu", "pilpres", "piala", "liga", "turnamen", "olimpiade",
    "expo", "forum", "lebaran", "idul", "natal", "ramadan",
    "pelabuhan", "bandara", "terminal", "stasiun", "gedung",
    "jalan", "tol", "survei", "survey", "undang",
]


def _is_valid_who_candidate(name):
    """Cek apakah kandidat WHO valid (bukan nama acara/tempat/undang-undang)."""
    name_lower = name.lower()
    if any(kw in name_lower for kw in _WHO_EXCLUDE_KEYWORDS):
        return False
    # Cek apakah ini lokasi di gazetteer
    loc_set = {l.lower() for l in load_locations()}
    if name_lower in loc_set:
        return False
    # Buang kandidat yang dimulai dengan angka romawi atau kata institusi
    first_word = name.split()[0] if name.split() else ""
    _invalid_first = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                      "PT", "CV", "Tbk", "Pasal", "Undang", "Peraturan", "Bab",
                      "Nomor", "No", "Ayat"}
    if first_word in _invalid_first:
        return False
    return True


def _find_title_for_name(name, text):
    """
    Mencari gelar/jabatan yang mendahului nama dalam teks.
    Mengembalikan "Jabatan Nama" atau hanya "Nama" jika tidak ketemu.
    Contoh: "Menteri Perdagangan Zulkifli Hasan" -> "Menteri Perdagangan Zulkifli Hasan"
    """
    name_pos = text.find(name)
    if name_pos == -1:
        # Coba cari case-insensitive
        text_lower = text.lower()
        name_pos = text_lower.find(name.lower())
        if name_pos == -1:
            return name

    # Ambil 80 karakter sebelum nama
    preceding = text[max(0, name_pos - 80):name_pos].strip()

    # Cari gelar/jabatan resmi dari daftar TITLE_PREFIXES
    best_prefix = ""
    for tp in TITLE_PREFIXES:
        # Cari pola "Jabatan" atau "Jabatan Departemen" di akhir preceding text
        pattern = re.compile(
            r"(?:^|[\s,])(" + re.escape(tp) + r"(?:\s+[A-Z][a-zA-Z]*){0,2})\s*$",
            re.IGNORECASE
        )
        match = pattern.search(preceding)
        if match:
            found = match.group(1).strip(" ,")
            # Pastikan prefix dimulai dengan huruf (bukan angka/romawi)
            if found and found[0].isalpha() and len(found) > len(best_prefix):
                best_prefix = found

    if best_prefix:
        return f"{best_prefix} {name}"

    return name


# =============================================================================
# EKSTRAKSI PER-ELEMEN
# =============================================================================
def extract_what(clean_content, title):
    """
    WHAT: Mengambil inti peristiwa (Judul + Kalimat utama).
    """
    sentences = _split_sentences(clean_content)
    for lead in sentences[:3]:
        lead_clean = lead.strip()
        if len(lead_clean.split()) < 5:
            continue
        if any(art.lower() in lead_clean.lower() for art in SCRAPING_ARTIFACTS):
            continue
        if len(lead_clean) > 200:
            lead_clean = lead_clean[:200].rsplit(" ", 1)[0] + "..."
            
        if title and title.lower() not in lead_clean.lower() and len(title) > 10:
            return f"{title}. {lead_clean}"
        return lead_clean

    return title if title else NOT_FOUND


def extract_who(clean_content, title):
    """
    WHO: Mengekstrak nama-nama aktor utama BESERTA jabatan/gelarnya.
    Format: "Jabatan Nama1, Jabatan Nama2, dan Jabatan Nama3"
    """
    candidates = find_who_multi(clean_content, title, max_who=2, min_score=1.5)
    if not candidates:
        return NOT_FOUND

    # Filter kandidat yang bukan orang/lembaga
    filtered = [c for c in candidates if _is_valid_who_candidate(c)]
    if not filtered:
        filtered = candidates

    # Enrich setiap nama dengan jabatan/gelar
    text_context = " ".join(_split_sentences(clean_content)[:6])
    enriched = []
    for name in filtered:
        enriched_name = _find_title_for_name(name, text_context)
        enriched.append(enriched_name)

    # Format output
    if len(enriched) == 1:
        return enriched[0]
    elif len(enriched) == 2:
        return f"{enriched[0]} dan {enriched[1]}"
    else:
        return ", ".join(enriched[:-1]) + f", dan {enriched[-1]}"


def extract_when(clean_content, metadata_date):
    """
    WHEN: Mengekstrak waktu kejadian dalam format ringkas.
    Contoh: "Peristiwa terjadi pada Selasa, 11 April 2023"
    """
    sentences = _split_sentences(clean_content)
    lead_text = " ".join(sentences[:4])

    # Kumpulkan semua tanggal unik
    found_dates = []
    seen = set()
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(lead_text):
            date_str = match.group(0).strip()
            if date_str not in seen:
                seen.add(date_str)
                found_dates.append(date_str)

    # Jika tidak ada di lead, cari di seluruh konten
    if not found_dates:
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(clean_content):
                date_str = match.group(0).strip()
                if date_str not in seen:
                    seen.add(date_str)
                    found_dates.append(date_str)
                    break
            if found_dates:
                break

    if found_dates:
        if len(found_dates) == 1:
            return f"Peristiwa terjadi pada {found_dates[0]}"
        elif len(found_dates) >= 2:
            # Cek apakah tanggal kedua sudah merupakan bagian dari tanggal pertama
            if found_dates[1] in found_dates[0] or found_dates[0] in found_dates[1]:
                return f"Peristiwa terjadi pada {found_dates[0]}"
            return f"Peristiwa terjadi pada {found_dates[0]} dan {found_dates[1]}"

    if metadata_date:
        formatted = _format_metadata_date(str(metadata_date))
        return f"Berdasarkan waktu publikasi, peristiwa terjadi sekitar {formatted}"

    return NOT_FOUND


def extract_where(clean_content, dateline_location):
    """
    WHERE: Mengekstrak lokasi kejadian dalam format ringkas.
    Contoh: "Peristiwa terjadi di Jakarta"
    """
    # 1. Cari lokasi utama
    primary_location = find_location_near_dateline(clean_content, dateline_location)

    # 2. Cari semua lokasi yang disebutkan
    all_locations = find_locations_in_text(clean_content, max_results=3)

    if primary_location:
        # Deduplikasi lokasi
        other_locations = []
        for loc in all_locations:
            if loc.lower() != primary_location.lower():
                # Hindari substring (mis. "Jakarta" dan "Jakarta Selatan")
                if primary_location.lower() not in loc.lower() and loc.lower() not in primary_location.lower():
                    other_locations.append(loc)

        if other_locations:
            others = ", ".join(other_locations[:2])
            return f"Peristiwa terjadi di {primary_location}, terkait juga dengan lokasi {others}"
        else:
            return f"Peristiwa terjadi di {primary_location}"

    return NOT_FOUND


def extract_why(clean_content):
    """
    WHY: Mengekstrak alasan/penyebab peristiwa.
    Prioritas: (1) kausalitas antar-kalimat, (2) konektor kausal, (3) purposive.
    """
    sentences = _split_sentences(clean_content)

    # 1. Cek kausalitas antar-kalimat (Sliding Window): "Pasalnya...", "Sebab itu..."
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower().strip()
        for conn in INTER_SENTENCE_CAUSAL:
            if sent_lower.startswith(conn):
                prev_sent = sentences[i-1] if i > 0 else ""
                combined = f"{prev_sent} {sent}".strip()
                if len(combined) > 300:
                    combined = combined[:300].rsplit(" ", 1)[0] + "..."
                return combined

    # 2. Konektor kausal ("karena", "akibat", "disebabkan", dll)
    result = _extract_sentence_containing_connector(
        clean_content, CAUSAL_CONNECTORS, sentences, max_sentence_len=250
    )
    if result:
        return result

    # 3. Purposive ("guna", "agar", "dalam rangka", dll)
    result = _extract_sentence_containing_connector(
        clean_content, PURPOSE_CONNECTORS, sentences, max_sentence_len=250
    )
    if result:
        return result

    # 4. Fallback "untuk" hanya di 2 kalimat awal
    lead_text = " ".join(sentences[:2])
    clause = _extract_clause_after_connector(lead_text, ["untuk"], max_len=200)
    if clause:
        return clause

    return NOT_FOUND


def extract_how(clean_content):
    """
    HOW: Mengekstrak cara/proses/metode peristiwa.
    """
    sentences = _split_sentences(clean_content)

    # 1. Deteksi "dengan" diikuti kata kerja berimbuhan (pola linguistik kuat)
    for sent in sentences:
        if METHOD_VERB_PATTERN.search(sent.lower()):
            result = sent.strip()
            if len(result) > 280:
                result = result[:280].rsplit(" ", 1)[0] + "..."
            if len(result) >= 20:
                return result

    # 2. Konektor metode ("dengan cara", "melalui", "menggunakan", dll)
    result = _extract_sentence_containing_connector(
        clean_content, METHOD_CONNECTORS, sentences,
        skip_after_reporting_verb=True, max_sentence_len=250
    )
    if result:
        return result

    # 3. Fallback "usai" / "setelah" di 3 kalimat awal
    lead_text = " ".join(sentences[:3])
    lead_sentences = _split_sentences(lead_text)
    result = _extract_sentence_containing_connector(
        lead_text, HOW_FALLBACK_CONNECTORS, lead_sentences, max_sentence_len=250
    )
    if result:
        return result

    return NOT_FOUND


# =============================================================================
# ENTRY POINT
# =============================================================================
def extract_5w1h(article):
    """
    Mengekstrak 5W1H dari satu artikel secara ALGORITMIK (tanpa AI).

    Args:
        article: dict dengan keys 'title', 'content', 'date'

    Returns:
        dict {
            "what": str, "who": str, "when": str,
            "where": str, "why": str, "how": str
        }
    """
    title = str(article.get("title", "") or "")
    content = str(article.get("content", "") or "")
    metadata_date = article.get("date")

    if not content.strip():
        return {k: NOT_FOUND for k in ["what", "who", "when", "where", "why", "how"]}

    clean_content, dateline_location = _strip_dateline(content)

    return {
        "what": extract_what(clean_content, title),
        "who": extract_who(clean_content, title),
        "when": extract_when(clean_content, metadata_date),
        "where": extract_where(clean_content, dateline_location),
        "why": extract_why(clean_content),
        "how": extract_how(clean_content),
    }


# --- Untuk testing mandiri ---
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../preprocessed_news_sample.csv")
    sample = df[df["title"].str.contains("Rafael", case=False, na=False)].head(3)

    for _, row in sample.iterrows():
        article = {"title": row["title"], "content": row["content"], "date": row["date"]}
        result = extract_5w1h(article)
        print("=" * 70)
        print("JUDUL:", article["title"])
        for k, v in result.items():
            print(f"  {k.upper():6s}: {v}")