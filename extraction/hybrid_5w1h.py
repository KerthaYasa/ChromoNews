"""
hybrid_5w1h.py
==============
Versi HYBRID dari ekstraksi 5W1H -- entry point UTAMA yang dipakai app.py
(menggantikan rule_based_5w1h.extract_5w1h sebagai sumber data UI).

REVISI (v3): ner_pipeline & qa_pipeline sekarang DI-INJECT sebagai parameter
(bukan diambil dari singleton internal modul ner_model.py/qa_model.py).
Pipeline di-load SEKALI di app.py lewat @st.cache_resource lalu di-pass
turun ke sini -- lihat docstring ner_model.py untuk alasan lengkap.

Kombinasi teknik per elemen:

  WHO        -> Model NER (cahya/bert-base-indonesian-NER), MULTI-hasil
                (PER + ORG). Fallback ke heuristik kapitalisasi multi-
                kandidat (ner_helper.find_who_multi) kalau pipeline None
                (model gagal dimuat) atau tidak ada kandidat dari model.
  WHEN/WHERE -> TETAP rule-based (regex + gazetteer), MULTI-hasil -- ambil
                SEMUA tanggal/lokasi yang disebut, bukan cuma yang
                pertama/skor tertinggi.
  WHY/HOW    -> rule-based DULU (cepat, explainable, dipanggil untuk
                SEMUA artikel). Kalau hasilnya NOT_FOUND, baru fallback
                ke extractive QA model (kalau pipeline tersedia).

extraction/rule_based_5w1h.py TIDAK DIUBAH dan tetap berfungsi sebagai
baseline murni algoritmik -- berguna untuk ablation study di laporan.
"""

import re

from . import rule_based_5w1h as rb
from . import ner_model
from . import qa_model
from .gazetteer import find_locations_in_text
from .patterns import DATE_PATTERNS

NOT_FOUND = rb.NOT_FOUND


def _is_plausible_date(date_str):
    """
    Filter artefak scraping seperti "1314 Maret 2023" (harusnya "13-14 Maret
    2023" tapi tanda hubung/spasi hilang saat preprocessing sumber data).
    Validasi sederhana: ambil digit pertama di awal string, kalau > 31
    (bukan tanggal valid dalam sebulan) -> tolak match ini.
    """
    leading_digits = re.match(r"^\(?(\d+)", date_str)
    if leading_digits:
        first_num = leading_digits.group(1)
        # Kalau gabungan dua tanggal nyambung tanpa pemisah (mis. "1314"),
        # angkanya akan > 31 dan jelas bukan tanggal valid dalam sebulan.
        if len(first_num) > 2 and int(first_num) > 31:
            return False
    return True


def _add_unique_no_substring(found_list, candidate):
    """
    Tambahkan `candidate` ke `found_list` HANYA jika tidak redundan:
    - skip kalau candidate adalah substring dari item yang sudah ada
      (mis. "10/4/2023" vs "Senin (10/4/2023)" yang sudah masuk duluan)
    - kalau candidate JUSTRU lebih panjang/lengkap dan item lama adalah
      substring-nya, ganti item lama dengan candidate yang lebih lengkap
    """
    for i, existing in enumerate(found_list):
        if candidate in existing:
            return False  # sudah ter-cover oleh entry yang lebih lengkap
        if existing in candidate:
            found_list[i] = candidate  # candidate lebih lengkap, ganti
            return True
    found_list.append(candidate)
    return True


# =============================================================================
# WHEN -- rule-based MULTI-hasil
# =============================================================================
def extract_when_multi(clean_content, metadata_date, max_dates=3):
    """Kumpulkan SEMUA tanggal unik (non-redundan) yang disebut."""
    found = []
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(clean_content):
            val = m.group(0).strip()
            if not _is_plausible_date(val):
                continue
            _add_unique_no_substring(found, val)

    found = found[:max_dates]

    if not found and metadata_date:
        found = [rb._format_metadata_date(str(metadata_date))]

    return found if found else [NOT_FOUND]


# =============================================================================
# WHERE -- rule-based MULTI-hasil (gazetteer)
# =============================================================================
def extract_where_multi(clean_content, dateline_location, max_locations=3):
    """Kumpulkan SEMUA lokasi unik (non-redundan) yang disebut, prioritas dateline duluan."""
    found = []
    if dateline_location:
        _add_unique_no_substring(found, dateline_location)

    more = find_locations_in_text(clean_content, max_results=max_locations + 2)
    for loc in more:
        _add_unique_no_substring(found, loc)
        if len(found) >= max_locations:
            break

    return found[:max_locations] if found else [NOT_FOUND]


# =============================================================================
# WHO -- model NER, fallback ke heuristik
# =============================================================================
def extract_who_multi(clean_content, title, ner_pipeline, max_who=3):
    ner_result = ner_model.extract_who_multi(ner_pipeline, clean_content, title, max_entities=max_who)
    if ner_result:
        return ner_result, "ner-model"

    # Fallback: heuristik kapitalisasi, SEKARANG JUGA multi-kandidat
    # (tidak perlu model NER untuk dapat lebih dari 1 WHO -- lihat
    # ner_helper.find_who_multi)
    fallback = rb.find_who_multi(clean_content, title, max_who=max_who)
    return (fallback if fallback else [NOT_FOUND]), "heuristic"


# =============================================================================
# WHY / HOW -- rule-based dulu, fallback ke QA model kalau gagal
# =============================================================================
def extract_why_hybrid(clean_content, qa_pipeline):
    result = rb.extract_why(clean_content)
    if result != NOT_FOUND:
        return result, "rule-based"

    qa_result = qa_model.answer_question(qa_pipeline, "why", clean_content)
    if qa_result:
        return qa_result, "qa-model"

    return NOT_FOUND, "none"


def extract_how_hybrid(clean_content, qa_pipeline):
    result = rb.extract_how(clean_content)
    if result != NOT_FOUND:
        return result, "rule-based"

    qa_result = qa_model.answer_question(qa_pipeline, "how", clean_content)
    if qa_result:
        return qa_result, "qa-model"

    return NOT_FOUND, "none"


# =============================================================================
# ENTRY POINT
# =============================================================================
def extract_5w1h_hybrid(article, ner_pipeline=None, qa_pipeline=None):
    """
    Args:
        article: dict dengan keys 'title', 'content', 'date'
        ner_pipeline: hasil ner_model.load_ner_pipeline() yang SUDAH di-cache
                      di app.py (lewat @st.cache_resource), atau None kalau
                      model gagal dimuat (otomatis fallback ke heuristik)
        qa_pipeline: hasil qa_model.load_qa_pipeline() yang sudah di-cache,
                     atau None kalau model gagal dimuat (otomatis fallback
                     ke "Tidak disebutkan dalam artikel" untuk kasus yang
                     rule-based juga gagal)

    Returns:
        dict {
            "what": str,
            "who": List[str], "when": List[str], "where": List[str],
            "why": str, "how": str,
            "who_source": str,   # "ner-model" | "heuristic"
            "why_source": str,   # "rule-based" | "qa-model" | "none"
            "how_source": str,   # "rule-based" | "qa-model" | "none"
        }
    """
    title = str(article.get("title", "") or "")
    content = str(article.get("content", "") or "")
    metadata_date = article.get("date")

    if not content.strip():
        return {
            "what": NOT_FOUND, "who": [NOT_FOUND], "when": [NOT_FOUND],
            "where": [NOT_FOUND], "why": NOT_FOUND, "how": NOT_FOUND,
            "who_source": "none", "why_source": "none", "how_source": "none",
        }

    clean_content, dateline_location = rb._strip_dateline(content)

    who_val, who_src = extract_who_multi(clean_content, title, ner_pipeline)
    why_val, why_src = extract_why_hybrid(clean_content, qa_pipeline)
    how_val, how_src = extract_how_hybrid(clean_content, qa_pipeline)

    return {
        "what": rb.extract_what(clean_content, title),
        "who": who_val,
        "when": extract_when_multi(clean_content, metadata_date),
        "where": extract_where_multi(clean_content, dateline_location),
        "why": why_val,
        "how": how_val,
        "who_source": who_src,
        "why_source": why_src,
        "how_source": how_src,
    }

