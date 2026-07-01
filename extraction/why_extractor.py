"""
why_extractor.py
================
Ekstraksi WHY (Mengapa) dengan konektor kausal.
Sesuai masukan dosen: gunakan kata konektor "karena", "akibat", "disebabkan".
"""

import re
from typing import List, Optional

# =============================================================================
# CAUSAL CONNECTORS (Sesuai masukan dosen)
# =============================================================================
CAUSAL_CONNECTORS = [
    "karena", "sebab", "lantaran", "akibat", "disebabkan", "diakibatkan",
    "dipicu", "buntut", "imbas", "dampak", "berujung",
    "atas dasar", "dengan tujuan", "bertujuan", "dalam rangka",
]

INTER_SENTENCE_CAUSAL = [
    "pasalnya", "oleh karena itu", "sebab itu", "karenanya", "alhasil",
    "hal itu disebabkan", "hal tersebut disebabkan",
]

PURPOSE_CONNECTORS = [
    "untuk", "agar", "supaya", "guna", "berharap", "diharapkan",
]


def _split_sentences(text: str) -> List[str]:
    """Pecah teks menjadi kalimat."""
    text = re.sub(r'(\b[A-Z]{1,4})\.([\s])', r'\1. \2', text)
    text = re.sub(r'([a-z0-9"\')\]])(\.)([A-Z])', r'\1.\n\3', text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences


def _extract_sentence_with_connector(text: str, connectors: List[str]) -> Optional[str]:
    """Cari kalimat yang mengandung salah satu konektor."""
    text_lower = text.lower()
    sentences = _split_sentences(text)
    
    for sent in sentences:
        sent_lower = sent.lower()
        for conn in connectors:
            if conn in sent_lower:
                # Pastikan kalimat bermakna
                if len(sent) > 20:
                    return sent.strip()
    
    return None


def _extract_clause_after_connector(text: str, connectors: List[str]) -> Optional[str]:
    """Ekstrak klausa setelah konektor."""
    text_lower = text.lower()
    
    for conn in connectors:
        pos = text_lower.find(conn)
        if pos == -1:
            continue
        
        # Ambil teks setelah konektor
        after = text[pos + len(conn):].strip()
        
        # Cari akhir kalimat
        end_match = re.search(r'[.!?]', after)
        if end_match:
            clause = after[:end_match.end()]
        else:
            clause = after[:200]
        
        clause = clause.strip(" ,:-\"")
        if len(clause) > 15:
            return clause
    
    return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def extract_why(content: str) -> str:
    """
    WHY: Deteksi alasan menggunakan konektor kausal.
    Sesuai masukan dosen: gunakan "karena", "akibat", "disebabkan".
    """
    if not content:
        return "Tidak disebutkan dalam artikel"
    
    sentences = _split_sentences(content)
    
    # 1. Cari konektor antar-kalimat (pasalnya, oleh karena itu)
    for sent in sentences:
        sent_lower = sent.lower().strip()
        for conn in INTER_SENTENCE_CAUSAL:
            if sent_lower.startswith(conn):
                if len(sent) > 20:
                    return sent.strip()
    
    # 2. Cari konektor kausal di dalam kalimat
    result = _extract_sentence_with_connector(content, CAUSAL_CONNECTORS)
    if result:
        return result
    
    # 3. Cari tujuan (untuk, agar, guna)
    result = _extract_sentence_with_connector(content, PURPOSE_CONNECTORS)
    if result:
        return result
    
    # 4. Coba ekstrak klausa setelah "karena"
    clause = _extract_clause_after_connector(content, ["karena", "sebab"])
    if clause:
        return clause
    
    return "Tidak disebutkan dalam artikel"