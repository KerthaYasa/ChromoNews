"""
pos_tagger.py
=============
POS (Part-of-Speech) tagging Bahasa Indonesia, dipakai extraction/how_extractor.py
untuk mengidentifikasi kalimat yang mengandung kata kerja (verb) sebagai syarat
kandidat jawaban HOW (sesuai catatan dosen: "Lakukan POS Tagging untuk
mengidentifikasi kata kerja").

Strategi:
    - PRIMARY: model POS tagger Indonesia berbasis transformers
      ("w11wo/indonesian-roberta-base-posp-tagger", dilatih di dataset POSP
      dari IndoNLU). Tag verba pada tagset POSP semuanya diawali "VB"
      (VBI = verba intransitif, VBT = verba transitif, VBP = verba pasif,
      VBL = verba kopula/linking). Kita anggap verba jika label memuat "VB".
    - FALLBACK (jika model gagal dimuat / tidak ada koneksi internet):
      heuristik morfologi Indonesia -- kata dianggap verba jika memiliki
      afiks verba baku: awalan me-/mem-/men-/meng-/meny-/ber-/di-/ter-/per-,
      atau akhiran -kan/-i pada kata yang juga berawalan salah satu di atas.
      Heuristik ini konservatif (dipakai di banyak tool NLP Indonesia ringan
      ketika tidak ada tagger statistik), bukan pengganti model, tapi tetap
      berupa identifikasi morfologis kata kerja -- bukan sekadar tebak posisi.
"""

import re
from typing import List, Tuple

MODEL_NAME = "w11wo/indonesian-roberta-base-posp-tagger"

_POS_PIPELINE = None
_POS_UNAVAILABLE = False

# ---------------------------------------------------------------------------
# Fallback morfologis (dipakai kalau model transformers tidak bisa dimuat)
# ---------------------------------------------------------------------------
_VERB_PREFIX_PATTERN = re.compile(
    r"^(memper|diper|meng|meny|men|mem|me|ber|ter|per|di)[a-z]{3,}",
    re.IGNORECASE,
)
_VERB_SUFFIX_PATTERN = re.compile(r"[a-z]{4,}(kan|nya)$", re.IGNORECASE)

# Kata kerja umum tanpa afiks yang sering muncul di berita
_BARE_VERBS = {
    "ada", "jadi", "kena", "buat", "tahu", "mau", "bisa", "pergi", "datang",
    "beri", "ambil", "lihat", "kata", "ujar", "sebut", "tegas", "jelas",
    "tolak", "terima", "minta", "duga",
}

# Kata fungsi (preposisi/konjungsi/pronomina) yang KEBETULAN cocok pola
# afiks di atas tapi BUKAN verba -- wajib dikecualikan agar heuristik tidak
# terlalu longgar (tanpa daftar ini, "melalui", "sebagai", "dengan" dsb.
# ikut lolos sebagai "verba" hanya karena berakhiran/berawalan mirip).
_NON_VERB_EXCEPTIONS = {
    "melalui", "sebagai", "dengan", "tentang", "terhadap", "menurut",
    "hingga", "sedang", "dari", "demi", "bagi", "kepada", "seperti",
    "berikut", "berupa", "diantara", "berbagai", "beberapa",
    "berikutnya", "terkait", "tersebut",
}


def _load_pos_pipeline():
    """Lazy-load pipeline token-classification POS Indonesia."""
    global _POS_PIPELINE, _POS_UNAVAILABLE
    if _POS_UNAVAILABLE:
        return None
    if _POS_PIPELINE is None:
        try:
            from transformers import pipeline
            _POS_PIPELINE = pipeline(
                "token-classification",
                model=MODEL_NAME,
                aggregation_strategy="simple",
            )
        except Exception as e:
            print(f"[pos_tagger] Model POS gagal dimuat, fallback ke heuristik "
                  f"morfologi verba Indonesia: {e}")
            _POS_UNAVAILABLE = True
            return None
    return _POS_PIPELINE


def inject_pos_pipeline(pipe):
    """Untuk reuse pipeline yang sudah dimuat di app.py (@st.cache_resource)."""
    global _POS_PIPELINE
    _POS_PIPELINE = pipe


def _is_verb_word_heuristic(word: str) -> bool:
    w = word.lower().strip(".,!?\"'()")
    if not w or len(w) < 4:
        return False
    if w in _NON_VERB_EXCEPTIONS:
        return False
    if w in _BARE_VERBS:
        return True
    # Nominalisasi "pe-...-an" (mis. "penyelidikan", "pemeriksaan") BUKAN
    # verba meski mengandung akhiran "an" -- kecualikan pola ini secara
    # umum, bukan cuma via daftar statis.
    if re.match(r"^pe[a-z]+an$", w):
        return False
    if _VERB_PREFIX_PATTERN.match(w):
        return True
    if _VERB_SUFFIX_PATTERN.search(w):
        return True
    return False


def sentence_has_verb(sentence: str) -> bool:
    """
    True jika kalimat mengandung minimal 1 kata kerja (verb), berdasarkan
    POS tagging model (jika tersedia) atau heuristik morfologi fallback.
    """
    pipe = _load_pos_pipeline()

    if pipe is not None:
        try:
            tags = pipe(sentence)
            for t in tags:
                label = str(t.get("entity_group", t.get("entity", ""))).upper()
                if "VB" in label:
                    return True
            return False
        except Exception as e:
            print(f"[pos_tagger] Gagal tagging kalimat, fallback heuristik: {e}")

    # Fallback heuristik morfologi
    words = re.findall(r"[A-Za-z]+", sentence)
    return any(_is_verb_word_heuristic(w) for w in words)


def get_verbs(sentence: str) -> List[str]:
    """Ambil daftar kata (token) yang teridentifikasi sebagai verba di kalimat."""
    pipe = _load_pos_pipeline()

    if pipe is not None:
        try:
            tags = pipe(sentence)
            verbs = []
            for t in tags:
                label = str(t.get("entity_group", t.get("entity", ""))).upper()
                if "VB" in label:
                    verbs.append(t.get("word", "").strip())
            return [v for v in verbs if v]
        except Exception as e:
            print(f"[pos_tagger] Gagal ambil verba, fallback heuristik: {e}")

    words = re.findall(r"[A-Za-z]+", sentence)
    return [w for w in words if _is_verb_word_heuristic(w)]


# =============================================================================
# TESTING MANDIRI
# =============================================================================
if __name__ == "__main__":
    samples = [
        "Polisi menangkap pelaku melalui rekaman CCTV di lokasi kejadian.",
        "Indonesia telah ditetapkan sebagai tuan rumah Piala Dunia U-20.",
        "KPK melakukan penyelidikan melalui audit keuangan dan pemeriksaan rekening.",
    ]
    for s in samples:
        print(f"{s}\n  -> ada verba? {sentence_has_verb(s)} | verba: {get_verbs(s)}\n")
