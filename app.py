import streamlit as st
import pandas as pd
import time

from preprocess import preprocess_for_bm25
from bm25_search import build_bm25_index, search_bm25
from semantic_search import load_embedding_model, encode_corpus, search_semantic
from hybrid_search import reciprocal_rank_fusion
from extraction import extract_5w1h_hybrid, load_ner_pipeline, load_qa_pipeline
from paraphraser import configure_gemini, paraphrase_5w1h, _fallback_paragraph

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="ChromoNews | Hybrid Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FLAT MODERN (TANPA CARD / GLASSMORPHISM) ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;800&display=swap');

    /* Global Typography Override */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Gradient Line Accent for Main Header */
    .gradient-text {
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* --- Flat Article Section --- */
    .article-section {
        border-left: 3px solid #7b2ff7;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 0 8px 8px 0;
        transition: background 0.2s ease;
    }
    .article-section:hover {
        background: rgba(255, 255, 255, 0.04);
    }
    .article-title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.15rem;
        margin-bottom: 0.4rem;
        font-family: 'Outfit', sans-serif;
    }
    .article-meta {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.8rem;
    }
    .article-date {
        color: #00d2ff;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .article-score {
        color: #94a3b8;
        font-size: 0.78rem;
    }
    .article-snippet {
        color: #cbd5e1;
        font-size: 0.93rem;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }

    /* --- Flat Metric Cards (Dataset Tab) --- */
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border-bottom: 3px solid #7b2ff7;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* --- Flat Ringkasan Topik Section --- */
    .summary-section {
        border-left: 3px solid #00d2ff;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        color: #e2e8f0;
        line-height: 1.7;
        font-size: 0.95rem;
        background: rgba(0, 210, 255, 0.03);
        border-radius: 0 8px 8px 0;
    }

    /* --- 5W1H Inline Grid --- */
    .w5h1-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.6rem;
    }
    .w5h1-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.45rem 0.7rem;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.02);
    }
    .w5h1-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 40px;
        height: 22px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.6rem;
        font-family: 'Outfit', sans-serif;
        flex-shrink: 0;
        letter-spacing: 0.5px;
    }
    .badge-what  { background: rgba(0, 210, 255, 0.12); color: #00d2ff; }
    .badge-who   { background: rgba(123, 47, 247, 0.12); color: #a78bfa; }
    .badge-when  { background: rgba(16, 185, 129, 0.12); color: #10b981; }
    .badge-where { background: rgba(245, 158, 11, 0.12); color: #f59e0b; }
    .badge-why   { background: rgba(239, 68, 68, 0.12);  color: #ef4444; }
    .badge-how   { background: rgba(236, 72, 153, 0.12); color: #ec4899; }
    .w5h1-text {
        color: #cbd5e1;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    /* --- Kolom Parafrase (Kanan) --- */
    .paraphrase-box {
        background: rgba(0, 210, 255, 0.03);
        border-left: 3px solid #00d2ff;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        color: #e2e8f0;
        font-size: 0.92rem;
        line-height: 1.65;
        height: 100%;
    }
    .paraphrase-label {
        color: #00d2ff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        display: block;
    }
    .w5h1-label {
        color: #a78bfa;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
        display: block;
    }

    /* --- Streamlit Component Overrides --- */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(123, 47, 247, 0.4);
    }
    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
        border-radius: 8px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00d2ff;
        box-shadow: 0 0 0 1px #00d2ff;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI CACHING UNTUK LOAD DATA & MODEL (DIOPTIMASI) ---
@st.cache_resource(show_spinner="📄 Memuat Dataset...")
def load_data():
    try:
        df = pd.read_csv('preprocessed_news_sample.csv')
        return df
    except FileNotFoundError:
        st.error("File 'preprocessed_news_sample.csv' tidak ditemukan. Pastikan modul 1 & 2 sudah dijalankan.")
        return None

@st.cache_resource(show_spinner="🔍 Membangun BM25 Index...")
def load_bm25(_df):
    tokenized_corpus = [str(doc).split() for doc in _df['processed_content']]
    bm25_index = build_bm25_index(tokenized_corpus)
    return bm25_index

@st.cache_resource(show_spinner="🧠 Memuat Model Semantic Search...")
def load_semantic(_df):
    import gc
    model = load_embedding_model()
    corpus_embeddings = encode_corpus(model, _df['content'].tolist())
    gc.collect()  # Bersihkan memori yang tidak terpakai
    return model, corpus_embeddings

@st.cache_resource(show_spinner="🤖 Memuat Model NER (deteksi WHO)...")
def load_ner_model():
    """
    Di-cache PERMANEN oleh Streamlit selama proses app hidup -- hanya
    dieksekusi SEKALI (saat pertama kali dipanggil), bukan diulang tiap
    user melakukan search. Run berikutnya (rerun Streamlit karena
    interaksi user) langsung pakai resource dari cache ini, TIDAK reload.

    Kalau loading gagal (exception), Streamlit TIDAK menyimpan exception
    ke cache -- jadi rerun berikutnya akan otomatis dicoba lagi (berguna
    kalau kamu baru saja `pip install` dependency yang sebelumnya hilang).
    """
    return load_ner_pipeline()

@st.cache_resource(show_spinner="🤖 Memuat Model QA (fallback WHY/HOW)...")
def load_qa_model():
    """Sama seperti load_ner_model() di atas -- cache permanen, retry otomatis kalau gagal."""
    return load_qa_pipeline()

def get_hybrid_models():
    """
    Load KEDUA model hybrid (NER + QA) dalam SATU pemanggilan, dipakai SEKALI
    di awal alur search -- bukan dicoba satu-satu tersebar di tengah proses
    ekstraksi per-artikel. Masing-masing model di-try/except TERPISAH supaya
    kalau salah satu gagal (mis. NER ok tapi QA gagal), yang lain tetap bisa
    dipakai (graceful degradation, bukan all-or-nothing).
    """
    ner_pipeline, ner_error = None, None
    try:
        ner_pipeline = load_ner_model()
    except Exception as e:
        ner_error = str(e)

    qa_pipeline, qa_error = None, None
    try:
        qa_pipeline = load_qa_model()
    except Exception as e:
        qa_error = str(e)

    return ner_pipeline, ner_error, qa_pipeline, qa_error

# --- INISIALISASI (BERTAHAP) ---
df = load_data()
if df is not None:
    bm25_index = load_bm25(df)
    semantic_model, corpus_embeddings = load_semantic(df)
else:
    bm25_index, semantic_model, corpus_embeddings = None, None, None

# Eager-load model hybrid (NER + QA) di awal app, SEJAJAR dengan BM25/semantic
# di atas -- bukan ditunda sampai user pertama kali search. Berkat
# @st.cache_resource, ini hanya benar-benar dieksekusi SEKALI selama proses
# Streamlit hidup; rerun berikutnya (tiap interaksi user) langsung pakai
# hasil cache, prosesnya jadi ringan/instan.
ner_pipeline, ner_load_error, qa_pipeline, qa_load_error = get_hybrid_models()

# --- SIDEBAR: KONFIGURASI & INFO ---
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    api_key_input = st.text_input("Gemini API Key", type="password", placeholder="Masukkan API Key Anda...", help="Diperlukan untuk fitur Summarization")
    
    st.markdown("---")
    st.markdown("### 📊 Parameter Search")
    top_k = st.slider("Jumlah Artikel (Top-K)", min_value=3, max_value=20, value=5, help="Jumlah artikel teratas yang akan di-retrieve dan dikirim ke AI Summarizer")
    rrf_k = st.number_input("RRF K Constant", min_value=1, max_value=100, value=60, help="Smoothing constant untuk Reciprocal Rank Fusion")
    
    st.markdown("---")
    st.markdown("### 💡 Contoh Query")
    st.markdown("""
    - `kasus korupsi KPK 2023`
    - `perkembangan kasus Rafael Alun pajak`
    - `dampak ekonomi Silicon Valley Bank`
    - `mudik lebaran 2023`
    """)

# --- MAIN APP HEADER ---
st.markdown('<h1 class="gradient-text">🧬 ChromoNews</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">News Retrieval using Hybrid BM25 & Semantic Search with Rule-Based 5W1H Extraction + AI Paraphrasing</div>', unsafe_allow_html=True)

if df is None:
    st.stop() # Stop execution jika data gagal dimuat

# --- SEARCH BAR ---
search_col1, search_col2 = st.columns([5, 1])
with search_col1:
    query = st.text_input("🔍 Cari berita...", placeholder="Ketik topik berita, tokoh, atau peristiwa...", label_visibility="collapsed")
with search_col2:
    search_button = st.button("Search", use_container_width=True)

# --- TABS ---
tab_dataset, tab_ringkasan = st.tabs(["📂 Dataset", "📝 Hasil & Ringkasan"])

# --- TAB 1: DATASET ---
with tab_dataset:
    st.markdown("### Informasi Dataset")
    
    # Metric Cards
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Berita</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">Bulan (Mar-Apr 2023)</div></div>', unsafe_allow_html=True)
    with m_col3:
        avg_len = int(df['content'].str.len().mean())
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_len}</div><div class="metric-label">Rata-rata Karakter</div></div>', unsafe_allow_html=True)
    
    # Dataframe Display
    display_df = df.copy()
    st.dataframe(
        display_df[['date', 'title', 'content']],
        use_container_width=True,
        hide_index=True,
        height=400
    )

# --- TAB 2: HASIL & RINGKASAN ---
with tab_ringkasan:
    if search_button and query:

        # Catatan: validasi & konfigurasi Gemini API key dilakukan di tahap
        # AI Paraphraser (setelah ekstraksi 5W1H algoritmik selesai),
        # karena AI di arsitektur baru ini HANYA dipakai untuk merangkai
        # hasil ekstraksi jadi paragraf natural -- bukan untuk retrieval
        # ataupun ekstraksi 5W1H itu sendiri.

        # --- PROSES SEARCH (RETRIEVAL) ---
        with st.spinner("🔍 Mencari dokumen yang relevan (Hybrid Search)..."):
            start_time = time.time()
            
            # Preprocess query untuk BM25 (case folding + stopword + stemming)
            # Note: Query untuk Semantic Search menggunakan teks ASLI (tanpa preprocessing)
            query_processed = preprocess_for_bm25(query)
            
            # 1. BM25 Search (menggunakan query yang sudah di-preprocess)
            bm25_results = search_bm25(query_processed, bm25_index, top_k=20)
            
            # 2. Semantic Search (menggunakan query ASLI tanpa preprocessing)
            semantic_results = search_semantic(query, semantic_model, corpus_embeddings, top_k=20)
            
            # 3. Hybrid Search (RRF)
            hybrid_results = reciprocal_rank_fusion(bm25_results, semantic_results, k=rrf_k, top_k=top_k)
            
            retrieval_time = time.time() - start_time
            
        st.success(f"Ditemukan {len(hybrid_results)} artikel yang relevan dalam {retrieval_time:.2f} detik.")
        
        # --- MENYIAPKAN DATA UNTUK EKSTRAKSI & UI ---
        retrieved_articles = []
        for doc_idx, rrf_score in hybrid_results:
            row = df.iloc[doc_idx]
            retrieved_articles.append({
                'title': row['title'],
                'date': row['date'],
                'content': row['content'],
                'score': rrf_score
            })

        # --- KONFIGURASI AI (untuk parafrase) ---
        ai_ready = False
        if api_key_input:
            try:
                configure_gemini(api_key=api_key_input)
                ai_ready = True
            except Exception as e:
                st.error(f"Gagal mengkonfigurasi Gemini API: {e}")

        # --- TAHAP 1: EKSTRAKSI 5W1H HYBRID (NER untuk WHO, rule-based multi untuk WHEN/WHERE, rule-based+QA fallback untuk WHY/HOW) ---
        # ner_pipeline & qa_pipeline sudah di-load SEKALI di awal app (lihat
        # get_hybrid_models() di bagian INISIALISASI) -- di sini TIDAK ada
        # loading ulang, cuma dipakai (pipeline bisa None kalau gagal load).
        status_msgs = []
        if ner_pipeline is None:
            status_msgs.append(f"NER (WHO): {ner_load_error or 'gagal dimuat'}")
        if qa_pipeline is None:
            status_msgs.append(f"QA (WHY/HOW fallback): {qa_load_error or 'gagal dimuat'}")
        if status_msgs:
            st.caption("⚠️ Model hybrid tidak aktif, pakai fallback rule-based/heuristik. Detail error: " + " | ".join(status_msgs))

        with st.spinner("🔬 Mengekstrak 5W1H (hybrid: NER + rule-based + QA fallback)..."):
            start_extract_time = time.time()
            all_5w1h = []
            for art in retrieved_articles:
                article_data = {'title': art['title'], 'date': art['date'], 'content': art['content']}
                all_5w1h.append(extract_5w1h_hybrid(article_data, ner_pipeline=ner_pipeline, qa_pipeline=qa_pipeline))
            extract_time = time.time() - start_extract_time
        st.caption(f"⚙️ Ekstraksi 5W1H (hybrid): {extract_time:.3f}s untuk {len(all_5w1h)} artikel")

        # --- TAHAP 2: AI PARAPHRASER (merangkai hasil ekstraksi jadi paragraf natural) ---
        all_paraphrases = []
        if ai_ready:
            with st.spinner("🤖 AI merangkai hasil ekstraksi menjadi paragraf natural..."):
                start_ai_time = time.time()
                for art, w5h1 in zip(retrieved_articles, all_5w1h):
                    paragraph = paraphrase_5w1h(w5h1, title=art['title'], query=query)
                    all_paraphrases.append(paragraph)
                ai_time = time.time() - start_ai_time
            st.caption(f"🤖 AI Paraphrasing: {ai_time:.2f}s untuk {len(all_paraphrases)} artikel")
        else:
            st.info("ℹ️ Gemini API Key belum dimasukkan — paragraf di kolom kanan ditampilkan dari penggabungan otomatis hasil ekstraksi (tanpa AI). Masukkan API Key di sidebar untuk paragraf yang lebih natural.")
            for art, w5h1 in zip(retrieved_articles, all_5w1h):
                all_paraphrases.append(_fallback_paragraph(w5h1, art['title']))

        # --- MENAMPILKAN ARTIKEL: KIRI 5W1H TERSTRUKTUR | KANAN PARAGRAF NATURAL ---
        st.markdown(f"### 📑 Top {len(retrieved_articles)} Artikel Relevan")

        for i, art in enumerate(retrieved_articles):
            item = all_5w1h[i]
            paragraph = all_paraphrases[i] if i < len(all_paraphrases) else ""

            # Header artikel (judul + meta)
            header_html = (
f'<div class="article-section" style="margin-bottom:0.6rem;">'
f'<div class="article-title">{art["title"]}</div>'
f'<div class="article-meta">'
f'<span class="article-date">🕒 {art["date"]}</span>'
f'<span class="article-score">RRF: {art["score"]:.4f}</span>'
f'</div>'
f'</div>'
            )
            st.markdown(header_html, unsafe_allow_html=True)

            col_left, col_right = st.columns([1, 1])

            with col_left:
                def _join(val):
                    if isinstance(val, list):
                        vals = [v for v in val if v and "Tidak disebutkan" not in v]
                        return ", ".join(vals) if vals else "Tidak terdeteksi"
                    return val if val else "Tidak terdeteksi"

                why_src = item.get("why_source", "")
                how_src = item.get("how_source", "")
                why_tag = " <em style='opacity:0.6;font-size:0.7em;'>(via QA model)</em>" if why_src == "qa-model" else ""
                how_tag = " <em style='opacity:0.6;font-size:0.7em;'>(via QA model)</em>" if how_src == "qa-model" else ""

                w5h1_html = (
'<div class="w5h1-grid" style="grid-template-columns:1fr;">'
'<span class="w5h1-label">🔍 5W+1H (Hybrid: NER + Rule-based + QA fallback)</span>'
'<div class="w5h1-item"><span class="w5h1-badge badge-what">WHAT</span>'
f'<span class="w5h1-text">{item.get("what", "Tidak terdeteksi")}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-who">WHO</span>'
f'<span class="w5h1-text">{_join(item.get("who"))}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-when">WHEN</span>'
f'<span class="w5h1-text">{_join(item.get("when"))}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-where">WHERE</span>'
f'<span class="w5h1-text">{_join(item.get("where"))}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-why">WHY</span>'
f'<span class="w5h1-text">{item.get("why", "Tidak terdeteksi")}{why_tag}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-how">HOW</span>'
f'<span class="w5h1-text">{item.get("how", "Tidak terdeteksi")}{how_tag}</span></div>'
'</div>'
                )
                st.markdown(w5h1_html, unsafe_allow_html=True)

            with col_right:
                paraphrase_html = (
'<div class="paraphrase-box">'
'<span class="paraphrase-label">📝 Ringkasan Natural (AI Paraphrase)</span>'
f'{paragraph}'
'</div>'
                )
                st.markdown(paraphrase_html, unsafe_allow_html=True)

            # Tombol "Baca Selengkapnya"
            with st.expander("📖 Baca Selengkapnya"):
                st.write(art['content'])

            st.markdown("---")

    elif search_button and not query:
        st.warning("Silakan masukkan kata kunci pencarian terlebih dahulu.")
    else:
        # Tampilan kosong saat belum search
        st.info("👈 Masukkan kata kunci pencarian dan klik 'Search' untuk melihat hasil.")
        
        # Placeholder Illustration
        st.markdown("""
        <div style="text-align: center; opacity: 0.5; padding: 4rem;">
            <div style="font-size: 5rem; margin-bottom: 1rem;">🔍</div>
            <h3 style="font-family: 'Outfit', sans-serif;">Menunggu Kueri Pencarian</h3>
            <p>Sistem siap mencari dan meringkas ratusan berita dalam hitungan detik.</p>
        </div>
        """, unsafe_allow_html=True)
