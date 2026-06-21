"""
rule_based_5w1h.py
===================
Modul UTAMA untuk ekstraksi 5W1H secara ALGORITMIK (manual rule-based),
menggantikan pendekatan lama yang sepenuhnya menyerahkan ekstraksi ke AI
generatif (Gemini).

Filosofi (lihat juga penjelasan di patterns.py):
- Setiap elemen 5W1H diekstrak dengan TEKNIK YANG BERBEDA sesuai sifat
  informasinya, semua berbasis Information Extraction klasik:
    WHAT  -> Lead sentence extraction (struktur piramida terbalik berita)
    WHO   -> Capitalization heuristic + title/org gazetteer (lihat ner_helper.py)
    WHEN  -> Regex pattern tanggal Bahasa Indonesia + fallback metadata
    WHERE -> Gazetteer lookup (lihat gazetteer.py)
    WHY   -> Pattern matching konjungsi kausal ("karena", "akibat", dst)
    HOW   -> Pattern matching konjungsi cara ("dengan cara", "melalui", dst)
- AI TIDAK dilibatkan sama sekali di modul ini. AI baru dipakai belakangan,
  di paraphraser.py, untuk merangkai dict hasil ekstraksi jadi 1 paragraf
  yang enak dibaca.

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
    INTER_SENTENCE_CAUSAL,    # <--- Tambahan baru
    METHOD_VERB_PATTERN,      # <--- Tambahan baru
    SCRAPING_ARTIFACTS        # <--- Tambahan baru
)
from .gazetteer import find_location_near_dateline
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
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _strip_dateline(content):
    """
    Diperbarui: Mengevaluasi swap grup lokasi dan media.
    Membersihkan prefix dateline media dari awal artikel.
    """
    match = DATELINE_PATTERN.match(content)
    if match:
        cand1 = match.group(1).strip()
        cand2 = match.group(2).strip()
        
        # Heuristik: Media biasanya pakai domain (.co, .com) atau "Indonesia"
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
        # Tangani format dengan/tanpa timezone, mis '2023-04-03 12:17:59+00'
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


def _extract_clause_after_connector(text, connectors, max_len=200, skip_after_reporting_verb=False):
    """Helper generik untuk WHY & HOW."""
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
            preceding = text_lower[max(0, pos - 25):pos]
            if any(rv in preceding for rv in REPORTING_VERBS):
                continue

        start = pos + len(conn)
        remainder = text[start:start + max_len]
        remainder_stripped = remainder.strip()

        if re.match(r"^\s*(itu|tersebut)\b", remainder_stripped, re.IGNORECASE):
            continue

        end_match = re.search(r'[.!?]|(?<=\w)"', remainder)
        if end_match:
            clause = remainder[:end_match.start()]
        else:
            clause = remainder
        clause = clause.strip(" ,:-\"")

        if clause and len(clause) >= 5:
            return clause

    return None


# =============================================================================
# EKSTRAKSI PER-ELEMEN
# =============================================================================
def extract_what(clean_content, title):
    """
    Diperbarui: Difilter dari kalimat promosi/sampah dan batas panjang minimal.
    """
    sentences = _split_sentences(clean_content)
    for lead in sentences:
        lead_clean = lead.strip()
        # Filter kalimat yang cuma 3-4 kata (biasanya noise)
        if len(lead_clean.split()) < 5:
            continue
            
        # Filter kata artefak (seperti "Baca Juga", "Gambas")
        if any(art.lower() in lead_clean.lower() for art in SCRAPING_ARTIFACTS):
            continue

        if len(lead_clean) > 280:
            lead_clean = lead_clean[:280].rsplit(" ", 1)[0] + "..."
        return lead_clean
        
    return title if title else NOT_FOUND


def extract_who(clean_content, title):
    """WHO = aktor utama. Lihat ner_helper.find_who()."""
    result = find_who(clean_content, title)
    return result if result else NOT_FOUND


def extract_when(clean_content, metadata_date):
    """WHEN = waktu kejadian."""
    lead_text = " ".join(_split_sentences(clean_content)[:2])
    for pattern in DATE_PATTERNS:
        match = pattern.search(lead_text)
        if match:
            return match.group(0).strip()

    for pattern in DATE_PATTERNS:
        match = pattern.search(clean_content)
        if match:
            return match.group(0).strip()

    if metadata_date:
        return _format_metadata_date(str(metadata_date))
    return NOT_FOUND


def extract_where(clean_content, dateline_location):
    """WHERE = lokasi kejadian. Lihat gazetteer.find_location_near_dateline()."""
    result = find_location_near_dateline(clean_content, dateline_location)
    return result if result else NOT_FOUND


def extract_why(clean_content):
    """
    Diperbarui: Menggunakan Sliding Window 2 kalimat untuk "Pasalnya", "Sebab itu", dll.
    """
    sentences = _split_sentences(clean_content)
    
    # 1. Cek kausalitas antar-kalimat (Sliding Window)
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        for conn in INTER_SENTENCE_CAUSAL:
            if sent_lower.startswith(conn):
                prev_sent = sentences[i-1] if i > 0 else ""
                return f"{prev_sent} {sent}".strip()

    # 2. Cek konektor kausal dalam 1 kalimat
    result = _extract_clause_after_connector(clean_content, CAUSAL_CONNECTORS)
    if result:
        return result

    # 3. Cek tujuan (Purposive)
    result = _extract_clause_after_connector(clean_content, PURPOSE_CONNECTORS)
    if result:
        return result

    # 4. Fallback "untuk"
    lead_text = " ".join(sentences[:2])
    result = _extract_clause_after_connector(lead_text, ["untuk"])
    if result:
        return result

    return NOT_FOUND


def extract_how(clean_content):
    """
    Diperbarui: Menggunakan pola Linguistik "Dengan + Afiksasi Verba" di awal untuk akurasi tinggi.
    """
    sentences = _split_sentences(clean_content)
    
    # 1. Deteksi "dengan" diikuti kata kerja berimbuhan
    for sent in sentences:
        if METHOD_VERB_PATTERN.search(sent.lower()):
            return sent.strip()

    # 2. Metode konjungsi standar
    result = _extract_clause_after_connector(
        clean_content, METHOD_CONNECTORS, skip_after_reporting_verb=True
    )
    if result:
        return result

    # 3. Fallback usai/setelah
    lead_text = " ".join(sentences[:3])
    result = _extract_clause_after_connector(lead_text, HOW_FALLBACK_CONNECTORS)
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