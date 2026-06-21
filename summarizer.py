"""
=============================================================================
‼️ DEPRECATED — SUDAH TIDAK DIPAKAI DI app.py ‼️
=============================================================================
File ini adalah versi LAMA (full-AI summarization) sebelum migrasi arsitektur.
Disimpan hanya sebagai referensi/perbandingan "before vs after" untuk laporan
STKI, BUKAN bagian dari pipeline aktif.

Pipeline AKTIF sekarang:
  - Ekstraksi 5W1H: extraction/rule_based_5w1h.py  (algoritmik, tanpa AI)
  - Parafrase akhir: paraphraser.py                (AI, hanya merangkai)

Alasan migrasi: lihat docstring di extraction/patterns.py dan paraphraser.py.
=============================================================================
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables dari .env file
load_dotenv()

def configure_gemini(api_key=None):
    """
    Configure Gemini API dengan API key.
    Jika api_key tidak diberikan, akan mencoba mengambil dari environment variable GEMINI_API_KEY.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Gemini API Key tidak ditemukan. Set environment variable GEMINI_API_KEY atau masukkan melalui UI.")
        
    genai.configure(api_key=api_key)

def summarize_articles(articles, query):
    """
    Meringkas kumpulan artikel berita menggunakan Gemini.
    
    Args:
        articles: List of dicts dengan keys: 'title', 'content', 'date'
        query: Query pencarian user (untuk konteks)
    Returns:
        {
            "abstract_summary": "...",      # Ringkasan abstraktif
            "chronological_summary": "..."  # Ringkasan kronologis
        }
    """
    if not articles:
        return {
            "abstract_summary": "Tidak ada artikel relevan untuk diringkas.",
            "chronological_summary": "Tidak ada peristiwa yang dapat disusun secara kronologis."
        }

    # Format artikel menjadi string untuk prompt
    formatted_articles = ""
    for i, article in enumerate(articles, 1):
        formatted_articles += f"\n--- ARTIKEL {i} ---\n"
        formatted_articles += f"Tanggal: {article['date']}\n"
        formatted_articles += f"Judul: {article['title']}\n"
        # Ambil max 1500 karakter per artikel agar tidak melebihi token limit
        content = str(article['content'])[:1500] 
        formatted_articles += f"Isi: {content}...\n"

    system_prompt = f"""Kamu adalah AI News Analyst ahli.
Diberikan kumpulan artikel berita yang terkait dengan topik pencarian pengguna: "{query}".

Tugasmu adalah menghasilkan output dalam format JSON yang valid, yang berisi dua kunci utama:
1. "abstract_summary": Rangkuman abstraktif inti dari seluruh artikel terkait topik "{query}" dalam 1-2 paragraf padat. Fokus pada fakta utama, tokoh penting, dan dampak.
2. "chronological_summary": Susunan perkembangan peristiwa berdasarkan urutan waktu (dari tanggal paling awal ke terbaru) yang ada di dalam artikel. Format nilainya adalah string yang berisi teks dengan bullet points atau list berurutan per tanggal.

Gunakan bahasa Indonesia yang baku dan mudah dipahami.
HANYA KEMBALIKAN JSON VALID TANPA MARKDOWN APAPUN (jangan pakai blok kode ```json).

Berikut adalah artikel-artikelnya:
{formatted_articles}
"""

    try:
        # Menggunakan gemini-flash-latest yang tersedia di API key saat ini
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(system_prompt)
        
        # Parse JSON dari respons
        response_text = response.text.strip()
        
        # Bersihkan dari blok markdown JSON jika Gemini masih ngeyel
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        result = json.loads(response_text)
        
        return {
            "abstract_summary": result.get("abstract_summary", "Gagal mengekstrak ringkasan abstraktif."),
            "chronological_summary": result.get("chronological_summary", "Gagal mengekstrak ringkasan kronologis.")
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Raw Response: {response.text}")
        return {
            "abstract_summary": "Terjadi kesalahan saat memformat respons dari AI (Bukan JSON yang valid).",
            "chronological_summary": response.text # Tampilkan raw response sebagai fallback
        }
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {
            "abstract_summary": f"Terjadi kesalahan saat memanggil API Gemini: {str(e)}",
            "chronological_summary": "Pastikan API Key valid dan kuota mencukupi."
        }


def extract_5w1h(article, query):
    """
    Mengekstrak elemen 5W1H dari satu artikel berita menggunakan Gemini Agent.
    
    Args:
        article: Dict dengan keys: 'title', 'content', 'date'
        query: Query pencarian user (untuk konteks)
    Returns:
        {
            "what": "...",    # Apa yang terjadi
            "who": "...",     # Siapa yang terlibat
            "when": "...",    # Kapan terjadi
            "where": "...",   # Di mana terjadi
            "why": "...",     # Mengapa terjadi
            "how": "..."      # Bagaimana bisa terjadi
        }
    """
    default_result = {
        "what": "Tidak terdeteksi",
        "who": "Tidak terdeteksi",
        "when": "Tidak terdeteksi",
        "where": "Tidak terdeteksi",
        "why": "Tidak terdeteksi",
        "how": "Tidak terdeteksi"
    }
    
    if not article:
        return default_result

    # Ambil konten artikel (max 2000 karakter untuk efisiensi token)
    content = str(article.get('content', ''))[:2000]
    
    prompt = f"""Kamu adalah AI News Analyst. Analisis artikel berita berikut dan ekstrak elemen 5W1H (What, Who, When, Where, Why, How).

Konteks pencarian: "{query}"

Judul: {article.get('title', 'Tidak ada judul')}
Tanggal Terbit: {article.get('date', 'Tidak diketahui')}
Isi Artikel:
{content}

Instruksi:
- Ekstrak setiap elemen 5W1H secara ringkas (1-2 kalimat per elemen)
- Jika suatu elemen tidak ditemukan dalam artikel, isi dengan "Tidak disebutkan dalam artikel"
- Gunakan bahasa Indonesia yang baku
- HANYA KEMBALIKAN JSON VALID tanpa markdown apapun

Format output JSON:
{{
    "what": "Peristiwa atau kejadian utama yang diberitakan",
    "who": "Pihak-pihak yang terlibat (pelaku, korban, saksi, dsb)",
    "when": "Waktu terjadinya peristiwa",
    "where": "Lokasi terjadinya peristiwa",
    "why": "Penyebab atau latar belakang peristiwa",
    "how": "Kronologi singkat atau cara terjadinya peristiwa"
}}
"""

    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        
        # Bersihkan dari blok markdown JSON
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
            
        result = json.loads(response_text)
        
        # Pastikan semua key ada
        for key in ["what", "who", "when", "where", "why", "how"]:
            if key not in result:
                result[key] = "Tidak terdeteksi"
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"5W1H JSON Decode Error: {e}")
        print(f"Raw Response: {response.text}")
        return default_result
    except Exception as e:
        print(f"5W1H Error: {e}")
        return default_result

# --- Untuk testing mandiri ---
if __name__ == "__main__":
    import pandas as pd
    
    # Load env (pastikan ada .env dengan GEMINI_API_KEY)
    try:
        configure_gemini()
        print("Berhasil konfigurasi Gemini API.")
    except Exception as e:
        print(f"Peringatan: {e}")
        print("Memasukkan dummy response...")
        # Lanjut dengan simulasi tanpa exit untuk kemudahan testing jika belum ada key
        
    print("\nMemuat sampel data...")
    df = pd.read_csv('preprocessed_news_sample.csv')
    
    # Ambil 3 artikel teratas terkait "Rafael Alun" sebagai simulasi hasil Hybrid Search
    sample_docs = df[df['title'].str.contains('Rafael', case=False, na=False)].head(3)
    
    if len(sample_docs) > 0:
        articles_to_summarize = []
        for _, row in sample_docs.iterrows():
            articles_to_summarize.append({
                'title': row['title'],
                'date': row['date'],
                'content': row['content']
            })
            
        print(f"Mengirim {len(articles_to_summarize)} artikel ke Gemini...\n")
        
        try:
            summary = summarize_articles(articles_to_summarize, "Rafael Alun")
            print("=== RINGKASAN ABSTRAKTIF ===")
            print(summary["abstract_summary"])
            print("\n=== RINGKASAN KRONOLOGIS ===")
            print(summary["chronological_summary"])
        except Exception as e:
            print(f"Test gagal: {e}")
    else:
        print("Tidak ada data sampel untuk dites.")
