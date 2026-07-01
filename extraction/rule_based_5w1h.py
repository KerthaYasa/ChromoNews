"""
rule_based_5w1h.py
===================
MODUL FINAL - Ekstraksi 5W1H dengan revisi pipeline
"""

import re
from datetime import datetime
from typing import List, Dict, Any

from extraction.patterns import (
    DATE_PATTERNS,
    REPORTING_VERBS,
    TITLE_PREFIXES,
    SCRAPING_ARTIFACTS,
)
from extraction.gazetteer import load_locations

# Import extractor
from extraction.who_extractor import extract_who
from extraction.where_extractor import extract_where
from extraction.how_extractor import extract_how
from extraction.why_extractor import extract_why


# =============================================================================
# CONSTANTS
# =============================================================================
NOT_FOUND = "Tidak disebutkan dalam artikel"

_BULAN_MAP = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}

_NER_PIPELINE = None


# =============================================================================
# INJECT NER
# =============================================================================
def inject_ner_pipeline(pipeline):
    """Inject NER pipeline"""
    global _NER_PIPELINE
    _NER_PIPELINE = pipeline
    
    from extraction.who_extractor import inject_ner_pipeline as inject_who
    from extraction.where_extractor import inject_ner_pipeline as inject_where
    
    inject_who(pipeline)
    inject_where(pipeline)


# =============================================================================
# UTILITY
# =============================================================================
def _split_sentences(text: str) -> List[str]:
    """Pecah teks menjadi kalimat."""
    text = re.sub(r'(\b[A-Z]{1,4})\.([\s])', r'\1. \2', text)
    text = re.sub(r'([a-z0-9"\')\]])(\.)([A-Z])', r'\1.\n\3', text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences


def _is_scraping_artifact(text: str) -> bool:
    """Cek artefak scraping."""
    for pattern in SCRAPING_ARTIFACTS:
        if pattern.lower() in text.lower():
            return True
    return False


def _strip_dateline(content: str) -> tuple:
    """Hapus prefix dateline media."""
    match = re.match(r"^([A-Za-z\.]{2,25})\s*,\s*([A-Za-z\.\s]{2,25})\s*[-–]\s*", content)
    if match:
        cand1 = match.group(1).strip()
        cand2 = match.group(2).strip()
        location = cand1 if ".CO" in cand2.upper() or "COM" in cand2.upper() else cand2
        clean = content[match.end():].strip()
        return clean, location
    
    match = re.match(r"^([A-Z][\w\.\s]{2,25})\s*[-–]\s*", content)
    if match:
        clean = content[match.end():].strip()
        return clean, None
    
    return content, None


# =============================================================================
# WHAT - Extractive Summarization
# =============================================================================
# extract_what (lebih general)
def extract_what(content: str, title: str) -> str:
    """
    WHAT: Prioritaskan kalimat pertama yang informatif (lead sentence).
    Ini biasanya paling baik untuk berita.
    """
    sentences = _split_sentences(content)
    
    for sent in sentences[:5]:  # cek 5 kalimat pertama
        sent = sent.strip()
        if len(sent) < 30:
            continue
        if _is_scraping_artifact(sent):
            continue
        if re.search(r"(klik|follow|download|subscribe|simak|lihat juga)", sent, re.IGNORECASE):
            continue
        
        # Batasi panjang dengan potong rapi
        if len(sent) > 220:
            truncated = sent[:220]
            last_space = truncated.rfind(' ')
            if last_space > 80:
                sent = truncated[:last_space]
        
        return sent
    
    # Fallback ke judul jika tidak ada kalimat bagus
    return title if title else NOT_FOUND


# =============================================================================
# WHEN - Deduplikasi
# =============================================================================
def _format_metadata_date(date_str: str) -> str:
    try:
        cleaned = str(date_str).replace("+00", "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(cleaned[:19], fmt)
                return f"{dt.day} {_BULAN_MAP[dt.month]} {dt.year}"
            except ValueError:
                continue
    except Exception:
        pass
    return str(date_str)


def extract_when(content: str, metadata_date: str) -> List[str]:
    """WHEN dengan deduplikasi."""
    found = []
    seen = set()
    
    clean_content = re.sub(r'(\d+)[–—](\d+)', r'\1-\2', content)
    
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(clean_content):
            date_str = match.group(0).strip()
            key = date_str.lower()
            if key not in seen:
                seen.add(key)
                found.append(date_str)
    
    if found:
        found.sort(key=len, reverse=True)
        deduped = []
        for d in found:
            if not any(d in existing for existing in deduped):
                deduped.append(d)
        return deduped[:3]
    
    if metadata_date:
        return [_format_metadata_date(str(metadata_date))]
    
    return [NOT_FOUND]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def extract_5w1h(article: Dict[str, Any]) -> Dict[str, Any]:
    title = str(article.get("title", "") or "")
    content = str(article.get("content", "") or "")
    metadata_date = article.get("date")
    
    if not content.strip():
        return {
            "what": NOT_FOUND,
            "who": [NOT_FOUND],
            "when": [NOT_FOUND],
            "where": [NOT_FOUND],
            "why": NOT_FOUND,
            "how": NOT_FOUND,
        }
    
    clean_content, dateline_location = _strip_dateline(content)
    
    where_result = extract_where(clean_content, dateline_location=dateline_location)
    who_result = extract_who(clean_content, title=title)
    
    return {
        "what": extract_what(clean_content, title),
        "who": who_result,
        "when": extract_when(clean_content, metadata_date),
        "where": where_result,
        "why": extract_why(clean_content),
        "how": extract_how(clean_content, title),
    }


# =============================================================================
# TESTING
# =============================================================================
if __name__ == "__main__":
    test_articles = [
        {
            "title": "Rafael Alun Ngaku Ditarget Jadi Tersangka",
            "content": """Komisi Pemberantasan Korupsi (KPK) menegaskan penetapan tersangka mantan pejabat pajak Rafael Alun Trisambodo memiliki landasan hukum. Rafael diduga menerima gratifikasi selama periode 2011-2023. Penggeledahan dilakukan di kantor KPK, Jakarta.""",
            "date": "2023-03-31"
        }
    ]
    
    for article in test_articles:
        result = extract_5w1h(article)
        print("\n" + "="*70)
        print(f"JUDUL: {article['title']}")
        for k, v in result.items():
            print(f"{k.upper():6s}: {v}")