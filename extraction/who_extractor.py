"""
who_extractor.py - v7 (Alias Resolution + Truncation Fix + Coverage)
======================================================================
WHO: NER primary + role classification + alias/coreference resolution
     + truncated name filtering.

Perubahan dari v6:
1. Alias resolution — dictionary nama panggilan tokoh publik Indonesia
   + fuzzy substring merge untuk tangani "Jokowi" vs "Joko Widodo".
2. Truncated name filter — buang kandidat yang terlihat terpotong
   (nama berakhir dengan suku kata terbuka seperti "Sub", "Wid", dll).
3. max_results dinaikkan ke 5 default agar pasangan calon/pihak terlibat
   ikut tertangkap, bukan hanya aktor utama.
"""

import re
from collections import defaultdict

from extraction.ner_model import WHO_LABELS, run_ner_chunked

_NER_PIPELINE = None


def inject_ner_pipeline(pipeline):
    global _NER_PIPELINE
    _NER_PIPELINE = pipeline


# =============================================================================
# BLACKLIST RINGAN
# =============================================================================

BLACKLIST_GENERIC = {
    "media sosial", "televisi", "internet", "online", "video", "foto",
    "tiktok", "instagram", "twitter", "facebook", "youtube", "whatsapp",
    "konferensi", "sidang", "rapat", "wawancara",
    "pihak", "pihak terkait", "oknum", "pejabat", "mantan pejabat",
    "aparat", "tokoh", "saksi", "tersangka", "terdakwa",
}

BLACKLIST_MEDIA_DOMAINS = (".com", ".co.id", ".org", ".net", ".id")

DAY_MONTH_WORDS = {
    "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
    "januari", "februari", "maret", "april", "mei", "juni", "juli",
    "agustus", "september", "oktober", "november", "desember",
}


def _basic_filter(name: str) -> bool:
    """Filter ringan."""
    if not name or len(name) < 2:
        return False
    name_lower = name.lower().strip()
    if name_lower in BLACKLIST_GENERIC:
        return False
    if name_lower in DAY_MONTH_WORDS:
        return False
    if any(d in name_lower for d in BLACKLIST_MEDIA_DOMAINS):
        return False
    if re.search(r'\d', name):
        return False
    if len(name.split()) > 6:
        return False
    return True


def _looks_like_region(name: str) -> bool:
    """Filter nama yang terlihat seperti nama daerah."""
    name_lower = name.lower()
    region_words = {
        "kabupaten", "provinsi", "kota", "kecamatan", "desa", "kelurahan",
        "jawa", "sumatera", "kalimantan", "sulawesi", "papua", "bali", 
        "jakarta", "bandung", "surabaya", "semarang", "yogyakarta", "medan", "makassar"
    }
    parts = name_lower.split()
    if any(p in region_words for p in parts):
        return True
    return False


# =============================================================================
# MASALAH #2 — TRUNCATED NAME FILTER
# =============================================================================

# Nama yang berakhir dengan 1-2 karakter setelah spasi terakhir kemungkinan
# terpotong (NER span boundary salah). Contoh: "Prabowo Sub", "Joko Wi".
# Threshold: kata terakhir <= 3 karakter DAN bukan inisial (tidak diawali huruf besar).
def _looks_truncated(name: str) -> bool:
    """
    Deteksi nama yang kemungkinan terpotong di tengah.
    Contoh: "Prabowo Sub" → True, "Joko Widodo" → False, "Ahmad S." → False.
    Inisial (1 huruf besar + titik) dikecualikan agar tidak drop nama berinisial.
    """
    parts = name.strip().split()
    if len(parts) < 2:
        return False
    last = parts[-1].rstrip(".")
    # Kata terakhir <= 3 karakter dan bukan inisial (huruf besar tunggal)
    if len(last) <= 3 and not (len(last) == 1 and last.isupper()):
        return True
    return False


# =============================================================================
# MASALAH #1 — ALIAS / COREFERENCE RESOLUTION
# =============================================================================

# Dictionary alias tokoh publik Indonesia yang sering muncul di berita.
# Format: alias_lowercase → nama_canonical_lowercase
# Nama canonical dipilih yang paling formal/lengkap.
# Tambahkan entri baru sesuai kebutuhan dataset.
ALIAS_DICT: dict[str, str] = {
    # Presiden / Wapres
    "jokowi":           "joko widodo",
    "pak jokowi":       "joko widodo",
    "presiden jokowi":  "joko widodo",
    "prabowo":          "prabowo subianto",
    "pak prabowo":      "prabowo subianto",
    "gibran":           "gibran rakabuming raka",
    "ma'ruf":           "ma'ruf amin",
    "maruf amin":       "ma'ruf amin",
    # Tokoh politik lain yang sering muncul
    "anies":            "anies baswedan",
    "sby":              "susilo bambang yudhoyono",
    "megawati":         "megawati soekarnoputri",
    "ahok":             "basuki tjahaja purnama",
    "ridwan kamil":     "ridwan kamil",       # canonical = dirinya sendiri
    "ganjar":           "ganjar pranowo",
    "mahfud":           "mahfud md",
    "mahfud md":        "mahfud md",
    "cak imin":         "muhaimin iskandar",
    "muhaimin":         "muhaimin iskandar",
    "sandiaga":         "sandiaga uno",
    "rocky gerung":     "rocky gerung",
    "surya paloh":      "surya paloh",
    "hasto":            "hasto kristiyanto",
    "zulhas":           "zulkifli hasan",
    # Tokoh hukum/KPK yang sering muncul
    "firli":            "firli bahuri",
    "novel":            "novel baswedan",
}


def _resolve_alias(name_lower: str) -> str:
    """
    Lookup alias dictionary. Return nama canonical jika ada, else return input.
    """
    return ALIAS_DICT.get(name_lower, name_lower)


def _is_substring_alias(short: str, long: str) -> bool:
    """
    Cek apakah `short` adalah nama panggilan/singkatan wajar dari `long`.
    Kondisi: semua token di `short` muncul di `long` secara berurutan.
    Contoh: "Prabowo" (1 token) ada dalam "Prabowo Subianto" → True.
            "Jokowi" tidak ada token yang cocok di "Joko Widodo" → False
            (kasus ini ditangani alias dict).
    """
    short_tokens = short.strip().lower().split()
    long_tokens = long.strip().lower().split()
    if not short_tokens or not long_tokens:
        return False
    # short harus subset berurutan dari long
    if len(short_tokens) >= len(long_tokens):
        return False
    # Cari apakah semua token short cocok secara berurutan di long
    it = iter(long_tokens)
    return all(tok in it for tok in short_tokens)


# =============================================================================
# ROLE CLASSIFICATION
# =============================================================================

_REPORTING_VERBS = re.compile(
    r"\b(kata|ujar|ucap|tutur|ungkap|sebut|terang|jelas|sampaik|menurut|"
    r"membenarkan|menyorot|menilai|menegaskan|mengaku|mengakui|"
    r"memastikan|mengatakan|menyampaikan|menyebut|berkata|berujar)\b",
    re.IGNORECASE,
)

_ACTION_VERBS = re.compile(
    r"\b(mendukung|mengumpulkan|mengunggah|mendeklarasikan|melakukan|"
    r"memimpin|menetapkan|mengeluarkan|menandatangani|memutuskan|"
    r"mengumumkan|memerintahkan|meluncurkan|meresmikan|menangkap|"
    r"menggeledah|mengajukan|meminta|mendorong|membuat|"
    r"membawa|mengambil|memberikan|menemukan|menyita|menuntut|"
    r"menghentikan|membubarkan|mengusulkan|menyetujui|menolak|"
    r"mencalonkan|mendaftarkan|berkampanye|memenangkan|mengalahkan)\b",
    re.IGNORECASE,
)

_REPORTING_WINDOW = 8


def _split_sentences_simple(text: str):
    """Pecah teks jadi kalimat."""
    text = re.sub(r'(\b[A-Z]{1,4})\.([\s])', r'\1. \2', text)
    text = re.sub(r'([a-z0-9"\'\)])(\.)\s*([A-Z])', r'\1.\n\3', text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def classify_entity_role(
    entity_name: str,
    content: str,
    first_sentence: str = "",
) -> str:
    """
    Klasifikasi peran entitas:
      - "main_actor"   : subjek aksi aktif atau muncul di kalimat pertama
                         tanpa konteks kutipan.
      - "cited_source" : muncul dalam window kata kerja pelapor.
      - "mentioned"    : default.
    """
    name_lower = entity_name.lower()
    sentences = _split_sentences_simple(content)
    entity_sentences = [s for s in sentences if name_lower in s.lower()]

    if not entity_sentences:
        return "mentioned"

    is_cited = False
    is_actor = False

    for sent in entity_sentences:
        tokens = sent.split()
        name_tokens = name_lower.split()
        entity_positions = []
        for i in range(len(tokens) - len(name_tokens) + 1):
            window = [t.lower().strip(".,;:\"'()") for t in tokens[i:i + len(name_tokens)]]
            if window == name_tokens:
                entity_positions.append(i)

        for ent_pos in entity_positions:
            window_start = max(0, ent_pos - _REPORTING_WINDOW)
            window_end = min(len(tokens), ent_pos + len(name_tokens) + _REPORTING_WINDOW)
            window_text = " ".join(tokens[window_start:window_end])

            if _REPORTING_VERBS.search(window_text):
                is_cited = True

            after_entity = " ".join(
                tokens[ent_pos + len(name_tokens): ent_pos + len(name_tokens) + 6]
            )
            if _ACTION_VERBS.search(after_entity):
                is_actor = True

    if first_sentence and name_lower in first_sentence.lower():
        if not _REPORTING_VERBS.search(first_sentence):
            is_actor = True

    if is_actor:
        return "main_actor"
    if is_cited:
        return "cited_source"
    return "mentioned"


# =============================================================================
# SCORING & RANKING (Role-Aware + Alias Resolution)
# =============================================================================

def _normalize_key(name: str) -> str:
    return re.sub(r'\s+', ' ', name.lower().strip())


def _build_alias_groups(raw_keys: list[str]) -> dict[str, list[str]]:
    """
    Bangun grup alias dari raw_keys:
    1. Resolve via ALIAS_DICT dulu (nama panggilan → canonical).
    2. Lakukan substring merge (nama pendek → nama panjang yang mengandungnya).
    3. Return dict: canonical_key → [list of original keys yang di-merge].
    """
    # Step 1: resolve alias dict
    resolved: dict[str, str] = {}  # original_key → canonical_key
    for k in raw_keys:
        resolved[k] = _resolve_alias(k)

    # Step 2: substring merge
    # Semua canonical key yang sudah di-resolve
    canonical_keys = list(set(resolved.values()))
    canonical_keys_sorted = sorted(canonical_keys, key=len, reverse=True)

    canonical_absorbed: dict[str, str] = {}  # short_canonical → long_canonical
    for long_k in canonical_keys_sorted:
        for short_k in canonical_keys_sorted:
            if short_k == long_k or short_k in canonical_absorbed:
                continue
            if _is_substring_alias(short_k, long_k):
                canonical_absorbed[short_k] = long_k

    # Step 3: build final groups (canonical_long → [original_keys])
    groups: dict[str, list[str]] = defaultdict(list)
    for orig_k in raw_keys:
        canon = resolved[orig_k]
        # Jika canonical ini terserap ke canonical yang lebih panjang, pakai yang panjang
        final_canon = canonical_absorbed.get(canon, canon)
        groups[final_canon].append(orig_k)

    return dict(groups)


def _score_and_rank(
    candidates: list,
    content_len: int,
    first_sentence: str,
    content: str,
    max_results: int = 5,
) -> list:
    """
    Scoring role-aware dengan alias resolution:
        score = (freq_count * 1.0)
              + (is_main_actor * 3.0)
              + (is_in_first_sentence * 1.5)
              - (is_cited_source_only * 0.5)

    Alias & substring merge dilakukan sebelum scoring sehingga
    "Jokowi" dan "Joko Widodo" dihitung sebagai satu entitas.
    """
    # Group mentah berdasarkan normalized text
    raw_groups: dict[str, list] = defaultdict(list)
    for c in candidates:
        raw_groups[_normalize_key(c["text"])].append(c)

    # Alias + substring resolution → canonical groups
    alias_groups = _build_alias_groups(list(raw_groups.keys()))

    # Gabungkan items berdasarkan canonical group
    merged: dict[str, list] = defaultdict(list)
    for canon_key, orig_keys in alias_groups.items():
        for ok in orig_keys:
            merged[canon_key].extend(raw_groups[ok])

    scored = []
    for canon_key, items in merged.items():
        freq = len(items)

        # Display name: pilih nama terpanjang yang TIDAK terpotong,
        # lalu coba restore jika masih terpotong
        name_candidates = sorted(
            {it["text"] for it in items}, key=len, reverse=True
        )
        display_name = next(
            (n for n in name_candidates if not _looks_truncated(n)),
            name_candidates[0],
        )

        # Jika masih terpotong, coba restore dari teks asli
        if _looks_truncated(display_name):
            restored = _restore_full_name(display_name, content)
            if not _looks_truncated(restored):
                display_name = restored
            else:
                display_name = canon_key.title()

        role = classify_entity_role(display_name, content, first_sentence)
        is_main_actor = role == "main_actor"
        is_cited_source_only = role == "cited_source"
        is_in_first_sentence = (
            canon_key in first_sentence.lower() if first_sentence else False
        )

        total_score = (
            (freq * 1.0)
            + (is_main_actor * 3.0)
            + (is_in_first_sentence * 1.5)
            - (is_cited_source_only * 0.5)
        )

        scored.append((total_score, freq, display_name, role))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, _, name, _ in scored[:max_results]]


# =============================================================================
# MAIN
# =============================================================================

def extract_who(content: str, title: str = "", max_results: int = 5) -> list:
    """
    Ekstraksi WHO dengan alias resolution dan truncation fix.
    max_results dinaikkan ke 5 (dari 4) agar pasangan calon/pihak terlibat
    ikut tertangkap.
    """
    global _NER_PIPELINE

    sentences = _split_sentences_simple(content)
    first_sentence = sentences[0] if sentences else ""

    if _NER_PIPELINE is not None:
        entities = run_ner_chunked(_NER_PIPELINE, content)

        candidates = []
        for e in entities:
            if e["label"] != "PER":
                continue

            name = e["text"].strip()

            # Masalah #2: buang nama yang kelihatan terpotong dari NER span boundary
            # Tapi coba restore dulu dari teks asli sebelum dibuang
            if _looks_truncated(name):
                name = _restore_full_name(name, content)
                if _looks_truncated(name):
                    continue  # masih terpotong setelah restore, skip

            if not _basic_filter(name):
                continue

            if _looks_like_region(name):
                continue

            # Bersihkan gelar jabatan di depan nama
            clean_name = re.sub(
                r'^(Menteri|Direktur|Gubernur|Presiden|Ketua|Wakil|Founder)\s+',
                '', name, flags=re.IGNORECASE,
            ).strip()
            if clean_name:
                e["text"] = clean_name

            candidates.append(e)

        if candidates:
            # Validasi substring pendek
            cand_texts = [c["text"] for c in candidates]
            valid_candidates = []
            for c in candidates:
                text = c["text"]
                if len(text) < 6:
                    is_substr = any(
                        text.lower() in other.lower() and len(text) < len(other)
                        for other in cand_texts
                    )
                    if is_substr:
                        continue
                valid_candidates.append(c)
            candidates = valid_candidates

        if candidates:
            ranked = _score_and_rank(
                candidates, len(content), first_sentence, content, max_results,
            )
            if ranked:
                return ranked

    return extract_who_fallback(content, max_results)


def _restore_full_name(name: str, content: str) -> str:
    """
    Post-processing: jika `name` adalah prefix dari nama yang muncul
    lebih lengkap di teks asli, kembalikan versi lengkapnya.

    Contoh: NER return "Prabow" tapi di teks ada "Prabowo" → return "Prabowo".
    Contoh: NER return "Prabowo Sub" tapi di teks ada "Prabowo Subianto" →
            return "Prabowo Subianto".

    Algoritma:
    1. Cari semua kemunculan `name` sebagai prefix di teks (case-insensitive).
    2. Untuk setiap posisi, ambil token lengkap berikutnya di teks asli.
    3. Jika token itu memperpanjang `name` secara wajar (bukan kata baru),
       ambil sebagai versi lengkap.
    """
    if not content or not name:
        return name

    # Cari nama sebagai substring murni di dalam kata (bisa di awal, tengah, akhir)
    escaped = re.escape(name)
    for match in re.finditer(rf'\b\S*{escaped}\S*\b', content, re.IGNORECASE):
        full_word = match.group(0).strip('.,;:"\')(')
        if len(full_word) > len(name):
            # Pastikan tambahannya wajar (hanya alfabet) agar tidak memasukkan simbol
            if re.match(r'^[a-zA-Z\-]+$', full_word):
                return full_word

    # Cari nama + kata berikutnya jika nama terpotong sebelum kata terakhir
    # Contoh: "Prabowo Sub" → "Prabowo Subianto"
    name_words = name.split()
    last_word = name_words[-1] if name_words else ""
    if len(name_words) >= 2 and len(last_word) <= 4:
        # Kata terakhir pendek — kemungkinan terpotong
        prefix = " ".join(name_words[:-1])
        escaped_prefix = re.escape(prefix)
        m2 = re.search(
            rf'\b{escaped_prefix}\s+({re.escape(last_word)}\w*)',
            content, re.IGNORECASE,
        )
        if m2:
            full_last = m2.group(1).strip('.,;:"\')(')
            if full_last.lower() != last_word.lower():
                return f"{prefix} {full_last}"

    return name
    """Deteksi nama wilayah secara sederhana."""
    lower = name.lower()
    if re.search(
        r'\b(jawa|sumatera|kalimantan|sulawesi|bali|ntt|ntb|maluku|papua|'
        r'yogyakarta|jakarta|bandung|surabaya|semarang)\b',
        lower,
    ):
        return True
    return False


# =============================================================================
# FALLBACK REGEX
# =============================================================================

def extract_who_fallback(content: str, max_results: int = 5) -> list:
    gelar_list = (
        "Menteri|Kepala|Gubernur|Presiden|Wakil Presiden|Komisaris|"
        "Direktur|Direktur Utama|Jaksa|Hakim|Kombes|Brigjen|Irjen|"
        "AKBP|Kompol|Sekretaris|Ketua|Wakil|Staf|Asisten|"
        "Inspektur|Kepala Badan|Deputi"
    )
    candidates = []

    pattern1 = rf'\b({gelar_list})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{0,3}})\b'
    for m in re.finditer(pattern1, content):
        name = f"{m.group(1)} {m.group(2)}"
        if _basic_filter(name) and not _looks_truncated(name):
            candidates.append({
                "text": name, "label": "PER",
                "start": m.start(), "end": m.end(), "score": 0.6,
            })

    pattern2 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    for m in re.finditer(pattern2, content):
        name = m.group(1)
        if _basic_filter(name) and not _looks_truncated(name):
            candidates.append({
                "text": name, "label": "PER",
                "start": m.start(), "end": m.end(), "score": 0.4,
            })

    if not candidates:
        return ["Tidak disebutkan dalam artikel"]

    sentences = _split_sentences_simple(content)
    first_sentence = sentences[0] if sentences else ""

    return _score_and_rank(
        candidates, len(content), first_sentence, content, max_results,
    ) or ["Tidak disebutkan dalam artikel"]
