import re
from typing import List, Optional, Set

from extraction.text_utils import split_sentences, is_scraping_artifact
from extraction.patterns import (
    METHOD_CONNECTORS,
    METHOD_VERB_PATTERN,
    HOW_FALLBACK_CONNECTORS,
    is_valid_how_secara,
    is_valid_how_dengan,
)
from extraction.text_similarity import bertscore_similarity
from extraction.pos_tagger import sentence_has_verb

# Opsional: jika modul pos_tagger menyediakan penghitung verb yang lebih
# akurat (hasil POS tagging asli), pakai itu. Kalau tidak ada, jatuh ke
# heuristik morfologi sebagai pendekatan kasar (lihat _heuristic_verb_count).
try:
    from extraction.pos_tagger import count_verbs as _pos_count_verbs
except ImportError:
    _pos_count_verbs = None

# ---------------------------------------------------------------------------
# Lazy-loaded model & Prototypes
# ---------------------------------------------------------------------------
_EMBED_MODEL = None

HOW_PROTOTYPES = [
    "Peristiwa ini terjadi melalui serangkaian proses yang sistematis.",
    "Hal tersebut dilakukan dengan cara melakukan pendekatan bertahap.",
    "Pelaku menggunakan metode tertentu untuk mencapai tujuannya.",
    "Proses ini dilaksanakan secara bertahap dan terorganisir.",
    "Caranya adalah dengan memanfaatkan sarana yang tersedia.",
    "Tindakan dilakukan lewat mekanisme yang telah ditetapkan.",
    "Langkah-langkah yang diambil meliputi berbagai tahapan.",
    "Modus operandi yang digunakan cukup terstruktur.",
]

# Bobot per jenis pola instrumental
_HIGH_WEIGHT_CONNECTORS = {"dengan cara", "dengan modus", "modus operandi", "modusnya"}
_WEIGHT_HIGH = 3.0
_WEIGHT_METHOD_CONNECTOR = 2.0
_WEIGHT_DENGAN_VERB = 2.5
_WEIGHT_SECARA = 1.5
_WEIGHT_AWAL = 1.5
_WEIGHT_CONCRETE_ACTION = 0.5

# --- BARU (v10): bobot untuk sinyal struktural ---
_WEIGHT_VERB_DENSITY_MAX = 0.6     # bonus maksimum dari densitas verb
_WEIGHT_TITLE_ANCHOR_MAX = 0.8     # bonus maksimum dari overlap ke judul/WHAT
_PENALTY_QUOTED = 1.2              # penalti besar untuk kalimat kutipan/laporan
_PENALTY_NEGATION_HARD = 999.0     # efektif hard-exclude (dipakai sebagai gate, bukan skor)

MIN_SENT_LEN = 25
MIN_WORD_COUNT = 5
WHAT_DEDUP_THRESHOLD = 0.85
ULTIMATE_FALLBACK_THRESHOLD = 0.55  # dinaikkan dari 0.40 (v9) -> lebih konservatif,
                                     # menerima lebih banyak NOT_FOUND demi presisi

MAX_ADJACENT_GAP = 2
MIN_SECOND_SCORE_RATIO = 0.5

NEGATION_HEAVY_THRESHOLD = 2  # >=2 penanda negasi -> kalimat defensif/klarifikasi

NOT_FOUND = "Tidak disebutkan dalam artikel"

_STATUS_FACT_PATTERN = re.compile(
    r"\b(?:telah\s+)?(?:ditetapkan|dinyatakan|ditunjuk|dinobatkan|ditasbihkan)\s+"
    r"(?:sebagai|menjadi)\s+\w+",
    re.IGNORECASE,
)

# =============================================================================
# 1) QUOTE / NARRATIVE SPLIT
# =============================================================================
_QUOTE_CHARS_PATTERN = re.compile(r'["“”\'‘’]')
_REPORTING_VERBS = re.compile(
    r"\b(kata|ujar|tutur|ungkap|sebut|tandas|menurut|ucap|tegas|jelas|"
    r"terang|imbuh|tambah|papar|beber|kata dia|katanya|ucapnya|tuturnya)\b",
    re.IGNORECASE,
)


def _is_quoted_or_reported(sentence: str) -> bool:
    """
    True jika kalimat adalah kutipan langsung narasumber atau kalimat
    pelaporan ucapan (mengandung tanda kutip signifikan ATAU verba
    pelaporan seperti 'kata', 'ujar', dst).

    Rasional: kalimat semacam ini secara struktural cenderung berisi
    opini, motif, bantahan, atau penekanan retoris narasumber --
    bukan deskripsi objektif reporter tentang proses/cara suatu
    tindakan dieksekusi. Ini BUKAN larangan mutlak (narasumber kadang
    memang menjelaskan metode teknis), tapi sinyal untuk DEPRIORITASKAN
    dibanding kalimat narasi murni, bukan di-hard-exclude.
    """
    quote_char_count = len(_QUOTE_CHARS_PATTERN.findall(sentence))
    has_substantial_quote = quote_char_count >= 2  # minimal sepasang tanda kutip
    has_reporting_verb = bool(_REPORTING_VERBS.search(sentence.lower()))
    return has_substantial_quote or has_reporting_verb


# =============================================================================
# 2) NEGATION-AWARE GATE
# =============================================================================
_NEGATION_MARKERS = re.compile(
    r"\b(tidak|todak|bukan|tanpa|jangan|tak|nggak|gak|belum)\b",
    re.IGNORECASE,
)


def _is_negation_heavy(sentence: str, threshold: int = NEGATION_HEAVY_THRESHOLD) -> bool:
    """
    True jika kalimat didominasi penanda negasi -- biasanya daftar "apa
    yang TIDAK dilakukan" (klarifikasi, bantahan, pembelaan diri), bukan
    penjelasan cara suatu tindakan benar-benar dilakukan.

    Ini gate WAJIB dijalankan SEBELUM verb density dihitung, karena verb
    yang dinegasikan ("tidak melakukan kriminal") tidak boleh menyumbang
    skor positif -- verb counting murni buta terhadap polaritas.
    """
    matches = _NEGATION_MARKERS.findall(sentence.lower())
    return len(matches) >= threshold


# =============================================================================
# 3) VERB DENSITY (sinyal tambahan)
# =============================================================================
_VERB_MORPHOLOGY_HINTS = re.compile(
    r"\b(me\w{3,}|di\w{3,}|ber\w{3,}|ter\w{3,}|\w+kan|\w+kannya)\b",
    re.IGNORECASE,
)


def _heuristic_verb_count(sentence: str) -> int:
    """
    Pendekatan kasar jumlah verb berdasarkan morfologi awalan/akhiran
    umum Bahasa Indonesia (me-, di-, ber-, ter-, -kan). INI HEURISTIK,
    bukan POS tagging asli -- akan salah hitung nomina berimbuhan
    (mis. 'pemerintah', 'kementerian'). Dipakai hanya sebagai fallback
    kalau modul pos_tagger tidak menyediakan penghitung verb eksplisit,
    dan bobotnya sengaja dibuat kecil (lihat _WEIGHT_VERB_DENSITY_MAX)
    supaya kesalahan heuristik ini tidak mendominasi skor akhir.
    """
    words = sentence.split()
    count = 0
    for w in words:
        w_clean = w.strip(".,\"'“”‘’!?;:")
        if _VERB_MORPHOLOGY_HINTS.fullmatch(w_clean):
            count += 1
    return count


def _verb_density(sentence: str) -> float:
    """
    Rasio jumlah verb terhadap total kata dalam kalimat. Dipakai sebagai
    BONUS tambahan (bukan gate), dan HANYA berkontribusi positif jika
    kalimat sudah lolos gate negasi -- lihat pemanggilannya di scoring.
    """
    words = sentence.split()
    if not words:
        return 0.0

    if _pos_count_verbs is not None:
        try:
            n_verbs = _pos_count_verbs(sentence)
        except Exception:
            n_verbs = _heuristic_verb_count(sentence)
    else:
        n_verbs = _heuristic_verb_count(sentence)

    return min(n_verbs / len(words), 1.0)


# =============================================================================
# 4) TITLE / WHAT ANCHORING
# =============================================================================
_STOPWORDS_ID = {
    "yang", "di", "ke", "dari", "dan", "atau", "dengan", "untuk", "pada",
    "dalam", "adalah", "akan", "ini", "itu", "juga", "sudah", "telah",
    "para", "sebagai", "oleh", "karena", "jika", "maka", "namun", "tetapi",
    "bahwa", "agar", "supaya", "hingga", "sampai", "antara", "saat", "ketika",
}


def _extract_anchor_tokens(text: str) -> Set[str]:
    """
    Ekstrak token signifikan (bukan stopword, panjang > 3 huruf) dari
    judul atau kalimat WHAT, untuk dipakai sebagai jangkar topik/aktor.
    Ini pendekatan lexical overlap sederhana -- bukan NER/dependency
    parsing penuh -- tapi cukup untuk menyingkirkan kalimat yang topiknya
    melenceng total dari peristiwa utama (mis. kalimat statistik yang
    tidak menyebut aktor/aksi apa pun dari judul/WHAT).
    """
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in tokens if len(t) > 3 and t not in _STOPWORDS_ID}


def _title_anchor_bonus(sentence: str, anchor_tokens: Set[str], max_bonus: float = _WEIGHT_TITLE_ANCHOR_MAX) -> float:
    """
    Bonus proporsional terhadap jumlah token kalimat yang overlap dengan
    anchor_tokens (dari judul + WHAT). Kalimat yang sama sekali tidak
    menyinggung aktor/topik utama peristiwa akan mendapat bonus ~0,
    sehingga turun peringkat secara alami dibanding kalimat yang relevan
    -- tanpa perlu tahu topik spesifik artikelnya apa.
    """
    if not anchor_tokens:
        return 0.0
    sent_tokens = _extract_anchor_tokens(sentence)
    if not sent_tokens:
        return 0.0
    overlap = sent_tokens & anchor_tokens
    ratio = len(overlap) / max(len(anchor_tokens), 1)
    return min(ratio, 1.0) * max_bonus


# =============================================================================
# 5) STATIVE / INTERPRETIVE FILTER (kategori umum, bukan pola per-kasus)
# =============================================================================
_INTERPRETIVE_MARKERS = re.compile(
    r"^\s*(artinya|ini berarti|dengan kata lain|maksudnya|singkatnya|"
    r"dapat disimpulkan|hal ini menunjukkan|sederhananya)\b",
    re.IGNORECASE,
)

_COPULA_STATE_PATTERN = re.compile(
    r"\b(?:akan|telah|sudah|adalah|merupakan)\s+(?:menjadi\s+)?\w+",
    re.IGNORECASE,
)

# Verb aksi konkret -- verb generik/dual-use ("menggunakan", "melalui",
# "memanfaatkan") SENGAJA tidak dimasukkan karena sama seringnya dipakai
# di kalimat motif/status maupun kalimat cara sungguhan, sehingga tidak
# reliable sebagai sinyal HOW yang berdiri sendiri.
_CONCRETE_ACTION_VERBS = re.compile(
    r"\b(memasang|dipasang|menempel|ditempel|menyebar(?:kan)?|disebar(?:kan)?|"
    r"menggelar|digelar|mengunggah|diunggah|menyampaikan|disampaikan|"
    r"mengirim(?:kan)?|dikirim(?:kan)?|membentangkan|dibentangkan|"
    r"menuliskan|dituliskan|mencoret|dicoret|memampangkan|dipampangkan|"
    r"melakukan|dilakukan|lakukan|menyelenggarakan|diselenggarakan|"
    r"mengadakan|diadakan|menyusun|disusun|merancang|dirancang)\b",
    re.IGNORECASE,
)


def _is_interpretive_sentence(sentence: str) -> bool:
    """Kalimat yang menafsirkan ulang fakta sebelumnya, bukan mendeskripsikan proses."""
    return bool(_INTERPRETIVE_MARKERS.match(sentence.strip()))


def _is_copula_state_sentence(sentence: str) -> bool:
    """
    Kalimat pernyataan status/klasifikasi/hitungan murni (kopula),
    KECUALI juga mengandung verb aksi konkret lain (kalimat campuran
    tetap boleh lolos).
    """
    sent_lower = sentence.lower()
    if not _COPULA_STATE_PATTERN.search(sent_lower):
        return False
    if _CONCRETE_ACTION_VERBS.search(sent_lower):
        return False
    return True


_FIXED_TERM_SECARA = re.compile(
    r"\b(pemilu|pemilihan|dipilih|terpilih)\w*\s+(presiden|kepala daerah|"
    r"wakil rakyat|umum)?\s*secara langsung\b",
    re.IGNORECASE,
)


def _is_fixed_term_secara_langsung(sentence: str) -> bool:
    """
    'secara langsung' pada frasa 'pemilihan ... secara langsung' adalah
    istilah baku (jenis/sistem pemilu), bukan adverbia cara yang
    menjelaskan bagaimana suatu tindakan dalam artikel ini dieksekusi.
    """
    return bool(_FIXED_TERM_SECARA.search(sentence.lower()))


# =============================================================================
# WHY vs HOW disambiguation (tiered, dari v9 -- tetap dipakai)
# =============================================================================
_STRONG_WHY_PHRASES = re.compile(
    r"\b(untuk pembenaran|sebagai pembenaran|bertujuan agar|dengan alasan|"
    r"untuk syahwat|demi ambisi|tujuannya adalah|dilandasi|alasannya|"
    r"motifnya|dipicu oleh|dilatarbelakangi)\b",
    re.IGNORECASE,
)

_WEAK_WHY_MARKERS = re.compile(
    r"\b(karena|sebab|akibat|demi|supaya|agar)\b",
    re.IGNORECASE,
)


def _is_why_not_how(sentence: str) -> bool:
    """Kalimat lebih menjelaskan MOTIF/ALASAN (why) ketimbang CARA (how)."""
    sent_lower = sentence.lower()

    if _STRONG_WHY_PHRASES.search(sent_lower):
        return True

    if _WEAK_WHY_MARKERS.search(sent_lower) and not _CONCRETE_ACTION_VERBS.search(sent_lower):
        return True

    return False


def _is_status_fact_sentence(sentence: str) -> bool:
    if not _STATUS_FACT_PATTERN.search(sentence):
        return False
    sent_lower = sentence.lower()
    has_other_process_marker = any(conn in sent_lower for conn in METHOD_CONNECTORS)
    if has_other_process_marker:
        return False
    return True


# =============================================================================
# Master gate: satu fungsi yang merangkum SEMUA filter struktural di atas,
# dipanggil konsisten di setiap layer supaya tidak ada celah kebocoran.
# =============================================================================
def _fails_structural_gate(sentence: str) -> bool:
    """
    True jika kalimat harus DIBUANG sepenuhnya dari kandidat HOW karena
    salah satu kondisi struktural berikut:
    - Motif/rasionalisasi (WHY), bukan cara (HOW)
    - Negation-heavy (daftar bantahan/klarifikasi)
    - Kalimat interpretif ("Artinya, ...")
    - Kalimat status/kopula murni ("akan menjadi ...")
    - 'secara langsung' sebagai istilah baku pemilu, bukan adverbia cara
    - Pernyataan status dengan pola "ditetapkan sebagai/menjadi"

    Fungsi ini TIDAK memeriksa kutipan/reporting -- itu ditangani
    terpisah lewat _is_quoted_or_reported() sebagai PENALTI, bukan gate,
    karena kutipan kadang tetap satu-satunya sumber detail yang tersedia.
    """
    if _is_why_not_how(sentence):
        return True
    if _is_negation_heavy(sentence):
        return True
    if _is_interpretive_sentence(sentence):
        return True
    if _is_copula_state_sentence(sentence):
        return True
    if _is_status_fact_sentence(sentence):
        return True
    return False


def inject_embed_model(model):
    global _EMBED_MODEL
    _EMBED_MODEL = model


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"[how_extractor] SentenceTransformer load gagal: {e}")
            return None
    return _EMBED_MODEL


HOW_TOP_N = 3


def _score_instrumental_patterns(sentence: str) -> float:
    total_score = 0.0
    sent_lower = sentence.lower()

    for conn in METHOD_CONNECTORS:
        if conn in sent_lower:
            if conn in _HIGH_WEIGHT_CONNECTORS:
                total_score += _WEIGHT_HIGH
            elif conn in ("berawal dari", "bermula dari", "diawali dari", "diawali"):
                total_score += _WEIGHT_AWAL
            else:
                total_score += _WEIGHT_METHOD_CONNECTOR

    if METHOD_VERB_PATTERN.search(sentence) and is_valid_how_dengan(sentence):
        total_score += _WEIGHT_DENGAN_VERB

    if is_valid_how_secara(sentence) and not _is_fixed_term_secara_langsung(sentence):
        total_score += _WEIGHT_SECARA

    return total_score


_PROCESS_VERBS = re.compile(
    r"\b(menggelar|memeriksa|menggeledah|menyita|"
    r"memproses|mengumpulkan|menyelidiki|mengusut|menindaklanjuti|"
    r"menelusuri|memverifikasi|mengecek|menginvestigasi)\b",
    re.IGNORECASE,
)


def _has_process_indicator(sentence: str) -> bool:
    if _PROCESS_VERBS.search(sentence):
        return True
    sent_lower = sentence.lower()
    if any(conn in sent_lower for conn in METHOD_CONNECTORS):
        return True
    return False


def _combine_top_how_candidates(
    candidates: List[dict],
    score_key: str = "final_score",
    max_gap: int = MAX_ADJACENT_GAP,
    min_score_ratio: float = MIN_SECOND_SCORE_RATIO,
) -> str:
    """
    Gabungkan kandidat HOW #1 dan #2 jika saling melengkapi (posisi
    berdekatan, bukan duplikat, skor #2 tidak jauh di bawah #1) supaya
    jawaban HOW tidak kehilangan detail yang tersebar di kalimat
    berikutnya (mis. kalimat cara umum + kalimat detail eksekusi).
    """
    if not candidates:
        return NOT_FOUND
    if len(candidates) == 1:
        return candidates[0]["text"]

    top, second = candidates[0], candidates[1]
    top_score = top.get(score_key, 0.0)
    second_score = second.get(score_key, 0.0)

    if top_score <= 0:
        return top["text"]

    score_ratio = second_score / top_score
    is_adjacent = abs(top["index"] - second["index"]) <= max_gap
    is_duplicate = top["text"].strip() == second["text"].strip()

    if is_adjacent and not is_duplicate and score_ratio >= min_score_ratio:
        ordered = sorted([top, second], key=lambda c: c["index"])
        return " ".join(c["text"].strip() for c in ordered)

    return top["text"]


# =============================================================================
# Layer 0: POS + BERTScore + Pattern, dengan quote-penalty & verb-density bonus
# =============================================================================
def _find_verb_bertscore_candidates(
    sentences: List[str],
    embed_model,
    anchor_tokens: Set[str],
    top_n: int = HOW_TOP_N,
) -> List[dict]:

    scored = []
    for idx, sent in enumerate(sentences):
        if len(sent) < MIN_SENT_LEN or len(sent.split()) < MIN_WORD_COUNT:
            continue
        if is_scraping_artifact(sent):
            continue
        if not sentence_has_verb(sent):
            continue
        if not _has_process_indicator(sent):
            continue
        if _fails_structural_gate(sent):
            continue

        proto_scores = [
            bertscore_similarity(sent, proto, embed_model=embed_model)
            for proto in HOW_PROTOTYPES
        ]
        best_bert_score = max(proto_scores) if proto_scores else 0.0

        pattern_score = _score_instrumental_patterns(sent)

        concrete_bonus = (
            _WEIGHT_CONCRETE_ACTION if _CONCRETE_ACTION_VERBS.search(sent.lower()) else 0.0
        )

        # Verb density -- HANYA berkontribusi karena kalimat sudah lolos
        # gate negasi di _fails_structural_gate() di atas. Ini urutan
        # yang disengaja: densitas verb tanpa gate negasi bisa salah
        # menaikkan kalimat "tidak...tidak...tidak..." yang verb-nya
        # banyak tapi semuanya dinegasikan.
        density_bonus = _verb_density(sent) * _WEIGHT_VERB_DENSITY_MAX

        anchor_bonus = _title_anchor_bonus(sent, anchor_tokens)

        penalty = 0.0
        if _is_quoted_or_reported(sent):
            penalty += _PENALTY_QUOTED

        if pattern_score == 0 and len(sent.split()) < 12:
            penalty += 0.20

        final_score = (
            best_bert_score
            + (pattern_score * 0.15)
            + concrete_bonus
            + density_bonus
            + anchor_bonus
            - penalty
        )

        if sent.strip().lower().startswith(("dan itu", "itu ", "hal itu", "karena itu")):
            if pattern_score == 0:
                continue

        if pattern_score == 0 and final_score < 0.65:
            continue

        scored.append({
            "text": sent,
            "index": idx,
            "bertscore": best_bert_score,
            "pattern_score": pattern_score,
            "is_quoted": _is_quoted_or_reported(sent),
            "final_score": final_score,
            "source": "pos_bertscore_hybrid",
        })

    scored.sort(key=lambda c: (-c["final_score"], c["index"]))
    return scored[:top_n]


def _find_pattern_candidates(sentences: List[str], anchor_tokens: Set[str]) -> List[dict]:
    candidates = []
    for idx, sent in enumerate(sentences):
        if len(sent) < MIN_SENT_LEN:
            continue
        if is_scraping_artifact(sent):
            continue
        if _fails_structural_gate(sent):
            continue

        score = _score_instrumental_patterns(sent)
        if _CONCRETE_ACTION_VERBS.search(sent.lower()):
            score += _WEIGHT_CONCRETE_ACTION

        score += _verb_density(sent) * _WEIGHT_VERB_DENSITY_MAX
        score += _title_anchor_bonus(sent, anchor_tokens)

        if _is_quoted_or_reported(sent):
            score -= _PENALTY_QUOTED

        if score > 0:
            candidates.append({
                "text": sent,
                "index": idx,
                "pattern_score": score,
                "is_quoted": _is_quoted_or_reported(sent),
                "source": "pattern",
            })

    candidates.sort(key=lambda x: (-x["pattern_score"], x["index"]))
    return candidates


def _rank_by_similarity_to_prototypes(candidates, embed_model, prototypes=HOW_PROTOTYPES):
    if not candidates or embed_model is None:
        return candidates

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    import numpy as np

    cand_texts = [c["text"] for c in candidates]
    cand_embeddings = embed_model.encode(cand_texts, convert_to_numpy=True)
    proto_embeddings = embed_model.encode(prototypes, convert_to_numpy=True)
    sim_matrix = cos_sim(cand_embeddings, proto_embeddings)

    for i, cand in enumerate(candidates):
        cand["proto_score"] = float(np.max(sim_matrix[i]))
    for cand in candidates:
        cand["combined_score"] = cand["pattern_score"] + (cand["proto_score"] * 0.5)

    candidates.sort(key=lambda x: -x["combined_score"])
    return candidates


def _is_similar_to_what(candidate_text, what_sentence, embed_model, threshold=WHAT_DEDUP_THRESHOLD):
    if not what_sentence:
        return False
    sim = bertscore_similarity(candidate_text, what_sentence, embed_model=embed_model)
    return sim > threshold


def _filter_what_duplicates(candidates, what_sentence, embed_model, threshold=WHAT_DEDUP_THRESHOLD):
    if not what_sentence:
        return candidates
    return [
        c for c in candidates
        if not _is_similar_to_what(c["text"], what_sentence, embed_model, threshold)
    ]


def _filter_narrative_first(candidates: List[dict]) -> List[dict]:
    """
    Jika ada minimal satu kandidat narasi (bukan kutipan/laporan),
    buang semua kandidat kutipan dari daftar -- narasi selalu diprioritaskan.
    Kutipan hanya dipertahankan sebagai kandidat kalau TIDAK ADA satu pun
    kandidat narasi yang tersedia (upaya terakhir, bukan pilihan utama).
    """
    narrative = [c for c in candidates if not c.get("is_quoted", False)]
    if narrative:
        return narrative
    return candidates


def _fallback_positional(sentences: List[str], what_sentence: str, embed_model, anchor_tokens: Set[str]) -> Optional[str]:
    window = sentences[1:4]
    candidates = []
    seen_idx = set()

    for offset, sent in enumerate(window):
        idx = offset + 1
        if idx in seen_idx:
            continue
        if len(sent.split()) < MIN_WORD_COUNT or is_scraping_artifact(sent):
            continue
        if _fails_structural_gate(sent):
            continue
        if _is_similar_to_what(sent, what_sentence, embed_model, WHAT_DEDUP_THRESHOLD):
            continue

        sent_lower = sent.lower().strip()
        starts_with_connector = any(sent_lower.startswith(conn) for conn in HOW_FALLBACK_CONNECTORS)
        has_process = _has_process_indicator(sent)

        if not (starts_with_connector or has_process):
            continue

        score = 2.0 if starts_with_connector else 1.0
        score += _verb_density(sent) * _WEIGHT_VERB_DENSITY_MAX
        score += _title_anchor_bonus(sent, anchor_tokens)
        if _is_quoted_or_reported(sent):
            score -= _PENALTY_QUOTED

        candidates.append({
            "text": sent,
            "index": idx,
            "pattern_score": score,
            "is_quoted": _is_quoted_or_reported(sent),
        })
        seen_idx.add(idx)

    if not candidates:
        return None

    candidates = _filter_narrative_first(candidates)
    candidates.sort(key=lambda c: (-c["pattern_score"], c["index"]))
    return _combine_top_how_candidates(candidates, score_key="pattern_score")


def _ultimate_semantic_fallback(
    sentences: List[str],
    what_sentence: str,
    embed_model,
    anchor_tokens: Set[str],
) -> Optional[str]:
    """
    Jaring terakhir: cari kalimat dengan skor BERTScore tertinggi
    terhadap HOW_PROTOTYPES, tapi tetap lolos gate struktural. Kutipan
    diizinkan di sini SEBAGAI UPAYA TERAKHIR (lewat _filter_narrative_first
    yang otomatis fallback ke kutipan kalau narasi kosong), karena pada
    titik ini lebih baik memberi jawaban dari kutipan yang relevan
    daripada NOT_FOUND -- asalkan tetap lolos gate negasi/motif/status.

    Threshold dinaikkan (lihat ULTIMATE_FALLBACK_THRESHOLD) supaya layer
    ini tidak lagi memaksakan kalimat yang sekadar "mirip gaya bahasa"
    ke prototype tapi topiknya melenceng -- anchor_bonus membantu
    memisahkan mana yang benar-benar relevan ke peristiwa utama.
    """
    if not embed_model or not sentences:
        return None

    scored = []
    for idx, sent in enumerate(sentences[1:], start=1):
        if len(sent.split()) < MIN_WORD_COUNT or is_scraping_artifact(sent):
            continue
        if _fails_structural_gate(sent):
            continue
        if _is_similar_to_what(sent, what_sentence, embed_model, WHAT_DEDUP_THRESHOLD):
            continue

        proto_scores = [bertscore_similarity(sent, proto, embed_model) for proto in HOW_PROTOTYPES]
        max_proto_score = max(proto_scores) if proto_scores else 0.0

        anchor_bonus = _title_anchor_bonus(sent, anchor_tokens)
        quote_penalty = _PENALTY_QUOTED if _is_quoted_or_reported(sent) else 0.0

        final_score = max_proto_score + anchor_bonus - quote_penalty

        if final_score > 0:
            scored.append({
                "text": sent,
                "index": idx,
                "final_score": final_score,
                "raw_proto_score": max_proto_score,
                "is_quoted": _is_quoted_or_reported(sent),
            })

    if not scored:
        return None

    scored = _filter_narrative_first(scored)
    scored.sort(key=lambda c: (-c["final_score"], c["index"]))

    # Cek threshold terhadap skor proto MENTAH (sebelum bonus/penalti)
    # supaya anchor_bonus tidak dipakai untuk "meloloskan paksa" kalimat
    # yang sebenarnya tidak relevan sama sekali dengan HOW_PROTOTYPES.
    if scored[0]["raw_proto_score"] <= ULTIMATE_FALLBACK_THRESHOLD:
        return None

    return _combine_top_how_candidates(scored, score_key="final_score")


def extract_how(content: str, title: str = "", what_sentence: str = "") -> str:
    if not content:
        return NOT_FOUND

    sentences = split_sentences(content)
    if not sentences:
        return NOT_FOUND

    embed_model = _get_embed_model()

    # Anchor tokens dari judul + kalimat WHAT -- dipakai di semua layer
    # sebagai bonus relevansi topik, bukan gate keras (lihat rasional di
    # _title_anchor_bonus).
    anchor_tokens = _extract_anchor_tokens(title) | _extract_anchor_tokens(what_sentence)

    # --- Layer 0 ---
    verb_candidates = _find_verb_bertscore_candidates(sentences, embed_model, anchor_tokens, top_n=HOW_TOP_N)
    verb_candidates = _filter_what_duplicates(verb_candidates, what_sentence, embed_model)
    verb_candidates = _filter_narrative_first(verb_candidates)

    if verb_candidates:
        return _combine_top_how_candidates(verb_candidates, score_key="final_score")

    # --- Layer 1 ---
    candidates = _find_pattern_candidates(sentences, anchor_tokens)
    candidates = _filter_what_duplicates(candidates, what_sentence, embed_model)
    candidates = _filter_narrative_first(candidates)

    if candidates:
        score_key = "pattern_score"
        if len(candidates) > 1 and embed_model is not None:
            candidates = _rank_by_similarity_to_prototypes(candidates, embed_model)
            score_key = "combined_score"
        return _combine_top_how_candidates(candidates, score_key=score_key)

    # --- Layer 3 ---
    fallback = _fallback_positional(sentences, what_sentence, embed_model, anchor_tokens)
    if fallback:
        return fallback

    # --- Layer 4 ---
    ultimate = _ultimate_semantic_fallback(sentences, what_sentence, embed_model, anchor_tokens)
    if ultimate:
        return ultimate

    return NOT_FOUND