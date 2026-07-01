"""
how_extractor.py - FIXED
================
Fokus pada cara/proses/aksi, bukan pengulangan WHAT.
"""

import re
from typing import List

def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n{2,}", text.strip())
    if len(paras) <= 1:
        paras = re.split(r"\n", text.strip())
    return [p.strip() for p in paras if p.strip() and len(p.strip()) > 30]


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r'(\b[A-Z]{1,4})\.([\s])', r'\1. \2', text)
    text = re.sub(r'([a-z0-9"\')\]])(\.)([A-Z])', r'\1.\n\3', text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences


def _is_scraping_artifact(text: str) -> bool:
    patterns = [
        r"^(baca juga|gambas|scroll|advertisement|lihat juga|simak juga)",
        r"(klik|follow|download|subscribe)",
        r"pilihan editor|trending",
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def extract_how(content: str, title: str = "") -> str:
    """
    HOW: Fokus pada proses/cara + gabungan kalimat yang koheren.
    """
    if not content:
        return "Tidak disebutkan dalam artikel"
    
    paragraphs = _split_paragraphs(content)
    candidates = []
    
    how_keywords = [
        "dengan cara", "melalui", "lewat", "menggunakan", "secara", 
        "persiapan", "antisipasi", "rute", "jalur", "hindari", "menyiapkan",
        "mencegah", "mengurangi", "beristirahat", "mengisi", "memilih"
    ]
    
    for para_idx, para in enumerate(paragraphs):
        sentences = _split_sentences(para)
        for sent in sentences:
            s = sent.strip()
            if len(s) < 25 or _is_scraping_artifact(s):
                continue
            
            score = 1.0
            lower = s.lower()
            
            # Bonus kata kunci HOW
            for kw in how_keywords:
                if kw in lower:
                    score += 2.5
                    break
            
            # Bonus jika di paragraf belakang (lebih mungkin menjelaskan cara)
            if para_idx >= len(paragraphs) * 0.5:
                score += 1.2
            
            candidates.append({'text': s, 'score': score, 'para_idx': para_idx})
    
    if not candidates:
        return "Tidak disebutkan dalam artikel"
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best = candidates[0]['text']
    
    # === GABUNGAN 1-2 KALIMAT (Logic yang lebih baik) ===
    if len(candidates) > 1 and candidates[1]['score'] >= 1.2:
        second = candidates[1]['text']
        
        # Cek apakah kalimat kedua koheren
        best_words = set(re.findall(r'\w{3,}', best.lower()))
        second_words = set(re.findall(r'\w{3,}', second.lower()))
        overlap = len(best_words & second_words)
        
        combined = best + " " + second
        if len(combined) < 340 and overlap >= 1:
            # Tambahkan connector jika perlu
            if overlap < 3:
                combined = best + " Sementara itu, " + second
            return combined.strip()
    
    return best