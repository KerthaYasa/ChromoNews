"""
generate_ai_5w1h.py — Validasi AI, Step A: AI mengekstrak 5W1H
====================================================================
Sesuai catatan dosen (Validasi dengan AI):
    "Gunakan 50 berita sampel yang sama. Minta AI menghasilkan
    ekstraksi 5W1H."

Skrip ini memakai SAMPEL 50 BERITA YANG SAMA seperti
evaluation/create_5w1h_eval_pool.py (seed sampling identik) supaya hasil
sistem, ground truth manusia, dan hasil AI bisa dibandingkan head-to-head
pada baris yang sama persis.

AI (Gemini) diminta melakukan ekstraksi 5W1H MURNI dari teks artikel
(BUKAN dibantu hasil ekstraksi sistem seperti di paraphraser.py -- di sini
AI harus membaca ulang seluruh artikel dan menyimpulkan sendiri 6
komponennya, supaya jadi pembanding independen yang adil).

Cara pakai:
    python evaluation/generate_ai_5w1h.py
    (butuh GEMINI_API_KEY di environment variable atau file .env)

Output:
    5w1h_eval_ai.csv -- kolom sama seperti 5w1h_eval_template.csv, TAPI
    kolom "*_truth" diisi hasil AI (bukan manusia), supaya bisa langsung
    dibandingkan dengan compare_system_vs_ai.py.
"""

import os
import sys
import json
import random
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.rule_based_5w1h import extract_5w1h
from paraphraser import configure_gemini

SEED = 42  # HARUS SAMA dengan create_5w1h_eval_pool.py agar sampel identik
N_SAMPLE = 50
DEFAULT_DATA_PATH = "preprocessed_news_sample.csv"
OUTPUT_PATH = "5w1h_eval_ai.csv"
NOT_FOUND = "Tidak disebutkan dalam artikel"

_AI_PROMPT_TEMPLATE = """Kamu adalah asisten analisis berita. Baca artikel berikut,
lalu ekstrak jawaban untuk 6 pertanyaan (5W+1H) HANYA berdasarkan isi
artikel (jangan mengarang/menambah fakta di luar artikel).

Judul: "{title}"

Isi artikel:
\"\"\"{content}\"\"\"

Jawab dalam format JSON PERSIS seperti ini, tanpa markdown/backtick, tanpa
komentar tambahan:
{{
  "what": "<kejadian utama, 1 kalimat>",
  "who": "<siapa yang terlibat, pisahkan beberapa nama dengan titik koma ;>",
  "when": "<kapan terjadi, pisahkan beberapa tanggal dengan titik koma ;>",
  "where": "<lokasi kejadian, pisahkan beberapa lokasi dengan titik koma ;>",
  "why": "<alasan/penyebab, 1 kalimat>",
  "how": "<cara/proses terjadinya, 1 kalimat>"
}}

Jika suatu informasi memang TIDAK disebutkan dalam artikel, isi dengan
tepat string: "{not_found}"
"""


def _call_gemini_5w1h(title: str, content: str, model, retries: int = 2) -> dict:
    prompt = _AI_PROMPT_TEMPLATE.format(title=title, content=content[:6000], not_found=NOT_FOUND)

    for attempt in range(retries + 1):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
            data = json.loads(text)
            return {
                "what": data.get("what", NOT_FOUND),
                "who": data.get("who", NOT_FOUND),
                "when": data.get("when", NOT_FOUND),
                "where": data.get("where", NOT_FOUND),
                "why": data.get("why", NOT_FOUND),
                "how": data.get("how", NOT_FOUND),
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"   ⚠ Gagal parsing respons AI setelah {retries+1} percobaan: {e}")
            return {k: NOT_FOUND for k in ("what", "who", "when", "where", "why", "how")}


def _join_list(value):
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return value


def generate_ai_extraction(data_path=DEFAULT_DATA_PATH, n_sample=N_SAMPLE, seed=SEED, output_path=OUTPUT_PATH):
    print("=" * 70)
    print(f"VALIDASI AI - STEP A: EKSTRAKSI 5W1H OLEH AI ({n_sample} berita)")
    print("=" * 70)

    configure_gemini()
    import google.generativeai as genai
    model = genai.GenerativeModel("gemini-flash-latest")

    print(f"\n📄 Memuat dataset dari '{data_path}'...")
    df = pd.read_csv(data_path)

    random.seed(seed)
    sampled_indices = random.sample(range(len(df)), min(n_sample, len(df)))
    df_sample = df.iloc[sampled_indices].reset_index(drop=True)
    print(f"   Sampel {len(df_sample)} berita (seed={seed} -- SAMA dengan "
          f"create_5w1h_eval_pool.py, jadi baris identik).")

    rows = []
    print(f"\n🤖 Meminta AI mengekstrak 5W1H per berita (bisa beberapa menit)...")
    for i, row in df_sample.iterrows():
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))

        sys_result = extract_5w1h({"title": title, "content": content, "date": row.get("date", "")})
        ai_result = _call_gemini_5w1h(title, content, model)

        rows.append({
            "id": row.get("id", i),
            "title": title,
            "content": content,

            "what_pred": sys_result["what"],
            "what_truth": ai_result["what"],

            "who_pred": _join_list(sys_result["who"]),
            "who_truth": ai_result["who"],

            "when_pred": _join_list(sys_result["when"]),
            "when_truth": ai_result["when"],

            "where_pred": _join_list(sys_result["where"]),
            "where_truth": ai_result["where"],

            "why_pred": sys_result["why"],
            "why_truth": ai_result["why"],

            "how_pred": sys_result["how"],
            "how_truth": ai_result["how"],
        })

        if (i + 1) % 5 == 0:
            print(f"   ...{i + 1}/{len(df_sample)} selesai")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Hasil ekstraksi AI disimpan ke '{output_path}'")
    print(f"  Kolom '*_pred' = hasil sistem, '*_truth' = hasil AI.")
    print(f"  Jalankan evaluation/compare_system_vs_ai.py untuk bandingkan.")

    return out_df


if __name__ == "__main__":
    generate_ai_extraction()
