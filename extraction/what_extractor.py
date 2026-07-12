"""
what_extractor.py
==================
WHAT: Multi-event extractive summarization.
Prioritaskan kalimat pertama yang informatif (lead sentence).
Jika artikel multi-event, tambahkan sinyal event kedua sebagai konteks.

Revisi:
1. Perbaikan regex untuk mendeteksi akronim (KPK, DPR, dll).
2. Perbaikan logika pemotongan (truncation) agar tanda baca akhir diganti titik dengan rapi.
3. Memastikan kalimat utama (lead) diakhiri dengan titik sebelum digabung.
"""

import re
from extraction.text_utils import split_sentences, is_scraping_artifact

# =============================================================================
# TOPIC SHIFT MARKERS — satu sumber untuk deteksi & strip
# =============================================================================
_TOPIC_SHIFT_WORDS = [
    "sementara itu", "di sisi lain", "selain itu", "adapun",
    "terkait hal ini", "terkait itu", "dalam kesempatan yang sama",
    "sebelumnya", "lebih lanjut", "di samping itu",
    "tak hanya itu", "tidak hanya itu",
    "di tempat terpisah", "pada kesempatan lain",
]
_TOPIC_SHIFT_ALTERNATION = "|".join(re.escape(w) for w in _TOPIC_SHIFT_WORDS)

_TOPIC_SHIFT_MARKERS = re.compile(
    rf'\b({_TOPIC_SHIFT_ALTERNATION})\b', re.IGNORECASE,
)
_TOPIC_SHIFT_STRIP_PATTERN = re.compile(
    rf'^(?:{_TOPIC_SHIFT_ALTERNATION})[,\s]*', re.IGNORECASE,
)


def _is_new_event_sentence(sent: str, prev_sents: list) -> bool:
    if not _TOPIC_SHIFT_MARKERS.search(sent):
        return False
    
    prev_text = " ".join(prev_sents).lower()
    
    # REVISI 1: Regex sekarang menangkap akronim huruf kapital (mis. KPK, DPR) 
    # maupun entitas dengan awalan kapital (mis. Jokowi, Jakarta)
    new_caps = re.findall(r'\b[A-Z][A-Za-z]*\b', sent)
    for cap in new_caps:
        if cap.lower() not in prev_text:
            return True
            
    return False


def extract_what(content: str, title: str) -> str:
    NOT_FOUND = "Tidak disebutkan dalam artikel"
    sentences = split_sentences(content)

    lead = None
    for sent in sentences[:5]:
        sent = sent.strip()
        if len(sent) < 30:
            continue
        if is_scraping_artifact(sent):
            continue
        if re.search(r"(klik|follow|download|subscribe|simak|lihat juga)", sent, re.IGNORECASE):
            continue

        if len(sent) > 400:
            truncated = sent[:400]
            last_punct = max(truncated.rfind(','), truncated.rfind('.'), truncated.rfind(';'))
            if last_punct > 80:
                # REVISI 2: Potong karakter HINGGA SEBELUM tanda baca, lalu tambahkan titik
                sent = truncated[:last_punct].strip() + '.'

        lead = sent
        break

    if not lead:
        return title if title else NOT_FOUND

    total = len(sentences)
    scan_end = max(15, min(30, int(total * 0.6)))
    second_event = None
    
    for sent in sentences[5:scan_end]:
        sent = sent.strip()
        if len(sent) < 30 or is_scraping_artifact(sent):
            continue
        if _is_new_event_sentence(sent, sentences[:5]):
            if len(sent) > 200:
                truncated = sent[:200]
                last_punct = max(truncated.rfind(','), truncated.rfind('.'), truncated.rfind(';'))
                if last_punct > 60:
                    # REVISI 2: Potong karakter HINGGA SEBELUM tanda baca, lalu tambahkan titik
                    sent = truncated[:last_punct].strip() + '.'
            second_event = sent
            break

    if second_event:
        second_clean = _TOPIC_SHIFT_STRIP_PATTERN.sub('', second_event).strip()
        if second_clean:
            # REVISI 3: Pastikan lead diakhiri titik sebelum disambung menjadi multi-event
            if not lead.endswith('.'):
                lead += '.'
                
            first_char = second_clean[0]
            rest = second_clean[1:]
            first_word = second_clean.split()[0] if second_clean.split() else ""
            
            if first_word.isupper() and len(first_word) > 1:
                return f"{lead} Selain itu, {second_clean}"
            return f"{lead} Selain itu, {first_char.lower()}{rest}"

    return lead