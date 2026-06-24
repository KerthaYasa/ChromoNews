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


def _fallback_paragraph(w5h1, title):
    """
    Fallback TANPA AI: gabungkan hasil ekstraksi algoritmik jadi paragraf
    sederhana pakai template (dipakai kalau API key tidak tersedia/gagal),
    supaya aplikasi tetap berfungsi penuh tanpa AI sama sekali.
    """
    what = _format_value(w5h1.get("what", ""))
    who = _format_value(w5h1.get("who", ""))
    when = _format_value(w5h1.get("when", ""))
    where = _format_value(w5h1.get("where", ""))
    why = _format_value(w5h1.get("why", ""))
    how = _format_value(w5h1.get("how", ""))

    parts = []
    if what and "Tidak disebutkan" not in what:
        parts.append(what)
    if who and "Tidak disebutkan" not in who:
        parts.append(f"Pihak yang terlibat: {who}.")
    if when and "Tidak disebutkan" not in when:
        parts.append(when + ".")
    if where and "Tidak disebutkan" not in where:
        parts.append(where + ".")
    if why and "Tidak disebutkan" not in why:
        parts.append(why)
    if how and "Tidak disebutkan" not in how:
        parts.append(how)

    if not parts:
        return title or "Ringkasan tidak tersedia."
    return " ".join(parts)


def paraphrase_5w1h(w5h1, title="", query=""):
    """
    Merangkai hasil ekstraksi 5W1H (algoritmik) menjadi 1 paragraf natural
    menggunakan Gemini sebagai PARAPHRASER murni (bukan extractor).

    Args:
        w5h1: dict {"what","who","when","where","why","how"} hasil dari
              extraction.rule_based_5w1h.extract_5w1h()
        title: judul artikel (konteks tambahan)
        query: query pencarian user (opsional, konteks tambahan)

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
        return text if text else _fallback_paragraph(w5h1, title)
    except Exception as e:
        print(f"Paraphraser error: {e}")
        return _fallback_paragraph(w5h1, title)


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
