"""
paraphraser.py
==============
Pengganti summarizer.py (DEPRECATED).

Perbedaan kunci dengan arsitektur lama:
- Lama: AI menerima FULL CONTENT artikel dan diminta MENGEKSTRAK + meringkas
  sekaligus (1 panggilan AI/artikel, AI menentukan sendiri apa itu 5W1H).
- Baru: AI hanya menerima dict 5W1H yang SUDAH diekstrak secara algoritmik
  oleh extraction/rule_based_5w1h.py, dan tugasnya HANYA merangkai 6 fakta
  pendek itu menjadi satu paragraf yang mengalir & enak dibaca dalam Bahasa
  Indonesia (paraphrasing, bukan ekstraksi/analisis).

Manfaat:
- Prompt jauh lebih pendek (6 fakta singkat vs ribuan karakter artikel)
  -> lebih cepat & murah secara token.
- AI tidak punya ruang untuk "mengarang" fakta baru di luar yang sudah
  diekstrak algoritma -> mengurangi risiko halusinasi.
- Kalau elemen 5W1H "Tidak disebutkan dalam artikel", AI diinstruksikan
  untuk tidak memaksakan menyebutnya di paragraf.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def configure_gemini(api_key=None):
    """
    Configure Gemini API dengan API key.
    Jika api_key tidak diberikan, akan mencoba mengambil dari environment
    variable GEMINI_API_KEY.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API Key tidak ditemukan. Set environment variable "
            "GEMINI_API_KEY atau masukkan melalui UI."
        )

    genai.configure(api_key=api_key)


def _format_value(value):
    """
    Helper: dict 5W1H hasil hybrid extractor punya nilai List[str] untuk
    who/when/where (multi-entitas) dan str biasa untuk what/why/how.
    Fungsi ini menyeragamkan jadi string yang siap ditampilkan/diprompt.
    """
    if isinstance(value, list):
        cleaned = [v for v in value if v and "Tidak disebutkan" not in v]
        return ", ".join(cleaned) if cleaned else "Tidak disebutkan dalam artikel"
    return value


def _extract_key_sentences(content: str, w5h1: dict) -> list:
    """
    Pilih 3-4 kalimat asli dari artikel yang paling representatif
    untuk Opsi A extractive summarization:
      1. Kalimat WHAT (lead sentence)
      2. Kalimat WHY jika berbeda dari WHAT
      3. Kalimat kutipan langsung paling awal jika ada
      4. Kalimat HOW jika berbeda dari WHAT dan WHY
    Kalimat dipilih dari teks ASLI, bukan dari field terstruktur.
    """
    if not content:
        return []

    import re

    what = w5h1.get("what", "")
    why  = w5h1.get("why", "")
    how  = w5h1.get("how", "")

    NOT_FOUND_STR = "Tidak disebutkan dalam artikel"

    # Pecah kalimat
    text = re.sub(r'(\b[A-Z]{1,4})\.([\s])', r'\1. \2', content)
    text = re.sub(r'([a-z0-9"\')\]])(\.)\s*([A-Z])', r'\1.\n\3', text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]

    selected = []
    used_norm = set()

    def norm(s):
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        return s

    def add(sent):
        n = norm(sent)
        if n not in used_norm and sent not in selected:
            used_norm.add(n)
            selected.append(sent)

    # 1. Lead sentence (dari WHAT atau kalimat pertama valid)
    if what and NOT_FOUND_STR not in what:
        add(what)
    elif sentences:
        add(sentences[0])

    # 2. Kalimat WHY jika ada dan berbeda dari WHAT
    if why and NOT_FOUND_STR not in why:
        if norm(why) not in used_norm:
            add(why)

    # 3. Kalimat kutipan langsung (mengandung tanda kutip atau kata kerja pelapor)
    quote_re = re.compile(r'["\u201c\u201d]|(\bkata\b|\bujar\b|\bmenurut\b)', re.I)
    for sent in sentences[:15]:
        if len(selected) >= 3:
            break
        if quote_re.search(sent) and norm(sent) not in used_norm:
            add(sent)
            break

    # 4. Kalimat HOW jika ada dan berbeda
    if how and NOT_FOUND_STR not in how:
        if norm(how) not in used_norm:
            add(how)

    return selected[:4]


def _fallback_paragraph(w5h1, title, content: str = "") -> str:
    """
    Fallback TANPA AI: Opsi A extractive summarization.
    Ambil 3-4 kalimat asli artikel yang paling representatif,
    gabungkan apa adanya — kalimat asli sudah gramatikal, tidak perlu template.
    Jika content tidak tersedia, fallback ke template minimal berbasis WHAT saja.
    """
    # Opsi A: gunakan kalimat asli artikel
    if content:
        sentences = _extract_key_sentences(content, w5h1)
        if sentences:
            return " ".join(sentences)

    # Fallback minimal jika content tidak ada: hanya WHAT + WHY + HOW
    # tanpa label field / daftar nama mentah
    NOT_FOUND_STR = "Tidak disebutkan dalam artikel"
    what = _format_value(w5h1.get("what", ""))
    why  = _format_value(w5h1.get("why", ""))
    how  = _format_value(w5h1.get("how", ""))

    parts = []
    if what and NOT_FOUND_STR not in what:
        parts.append(what if what.endswith('.') else what + '.')
    if why and NOT_FOUND_STR not in why and why.lower() not in what.lower():
        parts.append(why if why.endswith('.') else why + '.')
    if how and NOT_FOUND_STR not in how and how.lower() not in what.lower():
        parts.append(how if how.endswith('.') else how + '.')

    if not parts:
        return title or "Ringkasan tidak tersedia."
    return " ".join(parts)


def paraphrase_5w1h(w5h1, title="", query="", content=""):
    """
    Merangkai hasil ekstraksi 5W1H (algoritmik) menjadi 1 paragraf natural
    menggunakan Gemini sebagai PARAPHRASER murni (bukan extractor).

    Args:
        w5h1:    dict {"what","who","when","where","why","how"} hasil dari
                 extraction.rule_based_5w1h.extract_5w1h()
        title:   judul artikel (konteks tambahan)
        query:   query pencarian user (opsional, konteks tambahan)
        content: teks artikel asli (untuk extractive fallback Opsi A)

    Returns:
        str -- satu paragraf Bahasa Indonesia natural
    """
    facts_lines = "\n".join(f"- {k.upper()}: {_format_value(v)}" for k, v in w5h1.items())

    prompt = f"""Kamu adalah AI penyunting bahasa. Kamu TIDAK boleh menambahkan
fakta baru di luar yang diberikan. Tugasmu HANYA merangkai daftar fakta
5W1H berikut menjadi SATU paragraf Bahasa Indonesia yang natural, mengalir,
dan enak dibaca (3-5 kalimat), seolah-olah ditulis ulang oleh jurnalis.

Judul artikel: "{title}"

Fakta-fakta (hasil ekstraksi otomatis):
{facts_lines}

Instruksi:
- Jika suatu fakta bernilai "Tidak disebutkan dalam artikel", JANGAN
  disebutkan sama sekali di paragraf (lewati saja secara halus, jangan
  bilang "tidak diketahui" dsb).
- Jangan menambahkan opini, asumsi, atau fakta yang tidak ada di daftar.
- Tulis HANYA paragraf akhirnya, tanpa judul, tanpa markdown, tanpa
  bullet point, tanpa tanda kutip pembuka/penutup.
"""

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Bersihkan jika model tetap membungkus dengan kutip/markdown
        text = text.strip('"').strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
        return text if text else _fallback_paragraph(w5h1, title, content)
    except Exception as e:
        print(f"Paraphraser error: {e}")
        return _fallback_paragraph(w5h1, title, content)


# --- Untuk testing mandiri ---
if __name__ == "__main__":
    dummy = {
        "what": "Menteri Perdagangan berangkat menuju India untuk kunjungan kerja",
        "who": "Zulkifli Hasan",
        "when": "13 Maret 2023",
        "where": "India",
        "why": "Tidak disebutkan dalam artikel",
        "how": "Tidak disebutkan dalam artikel",
    }
    try:
        configure_gemini()
        print(paraphrase_5w1h(dummy, title="Mendag Terbang ke India"))
    except Exception as e:
        print(f"(Tanpa API key, pakai fallback) {e}")
        print(_fallback_paragraph(dummy, "Mendag Terbang ke India"))
