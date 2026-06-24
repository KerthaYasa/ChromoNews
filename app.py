import streamlit as st
import pandas as pd
import time

from preprocess import preprocess_for_bm25
from bm25_search import build_bm25_index, search_bm25
from semantic_search import load_embedding_model, encode_corpus, search_semantic
from hybrid_search import reciprocal_rank_fusion
from extraction.rule_based_5w1h import extract_5w1h
from paraphraser import configure_gemini, paraphrase_5w1h, _fallback_paragraph

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="ChromoNews | Hybrid Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False

# --- THEME CONFIGURATION ---
is_dark = st.session_state.theme == 'dark'

if is_dark:
    # Dark mode colors
    app_bg = "#0a0e17"
    sidebar_bg = "#111827"
    text_color = "#e2e8f0"
    card_bg = "rgba(255, 255, 255, 0.04)"
    card_hover = "rgba(255, 255, 255, 0.07)"
    sub_color = "#94a3b8"
    border_color = "rgba(255, 255, 255, 0.08)"
    input_bg = "rgba(255, 255, 255, 0.06)"
    input_border = "rgba(255, 255, 255, 0.12)"
    tab_bg = "rgba(255, 255, 255, 0.03)"
    tab_hover = "rgba(255, 255, 255, 0.06)"
    expander_bg = "rgba(255, 255, 255, 0.03)"
    header_bg = "#0a0e17"
    dataframe_header_bg = "rgba(255,255,255,0.05)"
    dataframe_cell_bg = "rgba(255,255,255,0.02)"
    divider_color = "rgba(255, 255, 255, 0.06)"
    scrollbar_thumb = "rgba(255, 255, 255, 0.15)"
else:
    # Light mode colors
    app_bg = "#f8f9fc"
    sidebar_bg = "#ffffff"
    text_color = "#1a1a2e"
    card_bg = "rgba(0, 0, 0, 0.02)"
    card_hover = "rgba(0, 0, 0, 0.04)"
    sub_color = "#64748b"
    border_color = "rgba(0, 0, 0, 0.08)"
    input_bg = "#ffffff"
    input_border = "rgba(0, 0, 0, 0.15)"
    tab_bg = "rgba(0, 0, 0, 0.02)"
    tab_hover = "rgba(0, 0, 0, 0.04)"
    expander_bg = "rgba(0, 0, 0, 0.02)"
    header_bg = "#f8f9fc"
    dataframe_header_bg = "rgba(0,0,0,0.04)"
    dataframe_cell_bg = "#ffffff"
    divider_color = "rgba(0, 0, 0, 0.08)"
    scrollbar_thumb = "rgba(0, 0, 0, 0.15)"

# --- MEGA CSS: Override EVERYTHING ---
st.markdown(f"""
<style>
    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;800&display=swap');

    /* ===== CSS CUSTOM PROPERTIES (Streamlit reads these) ===== */
    :root {{
        --background-color: {app_bg} !important;
        --secondary-background-color: {sidebar_bg} !important;
        --text-color: {text_color} !important;
        --font: 'Inter', sans-serif !important;
    }}

    /* ===== GLOBAL RESET ===== */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        color: {text_color} !important;
    }}

    /* ===== MAIN APP BACKGROUND ===== */
    .stApp,
    .stApp > header,
    .main .block-container {{
        background-color: {app_bg} !important;
        color: {text_color} !important;
    }}

    /* ===== TOP HEADER BAR ===== */
    [data-testid="stHeader"],
    header[data-testid="stHeader"] {{
        background-color: {app_bg} !important;
        backdrop-filter: none !important;
    }}

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {divider_color} !important;
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown li {{
        color: {text_color} !important;
    }}

    /* ===== ALL TEXT ELEMENTS ===== */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        font-family: 'Outfit', sans-serif !important;
    }}
    p, span, div, label, li {{
        color: {text_color};
    }}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {{
        color: {text_color} !important;
    }}
    .stMarkdown {{
        color: {text_color} !important;
    }}

    /* ===== MAIN HEADER GRADIENT ===== */
    .gradient-text {{
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0px;
    }}
    .sub-header {{
        color: {sub_color} !important;
        -webkit-text-fill-color: {sub_color} !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }}

    /* ===== THEME TOGGLE STYLING ===== */
    .theme-switch-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 16px;
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        margin-bottom: 0.3rem;
    }}
    .theme-switch-label {{
        font-size: 0.9rem;
        font-weight: 500;
        color: {text_color} !important;
    }}
    .theme-switch-icon {{
        font-size: 1.3rem;
    }}

    /* Style the st.toggle switch */
    [data-testid="stSidebar"] .stToggle > label {{
        color: {text_color} !important;
    }}
    [data-testid="stSidebar"] .stToggle > div > div {{
        color: {text_color} !important;
    }}

    /* ===== INPUT ELEMENTS ===== */
    .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stTextInput > div > div > input {{
        background-color: {input_bg} !important;
        border: 1px solid {input_border} !important;
        color: {text_color} !important;
        border-radius: 8px !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: #00d2ff !important;
        box-shadow: 0 0 0 1px #00d2ff !important;
    }}
    .stTextInput > div > div > input::placeholder {{
        color: {sub_color} !important;
        opacity: 0.7 !important;
    }}

    /* Number input */
    .stNumberInput > div > div > input,
    [data-testid="stSidebar"] .stNumberInput > div > div > input {{
        background-color: {input_bg} !important;
        border: 1px solid {input_border} !important;
        color: {text_color} !important;
    }}
    .stNumberInput button {{
        color: {text_color} !important;
        border-color: {input_border} !important;
        background-color: {card_bg} !important;
    }}

    /* Slider */
    [data-testid="stSidebar"] .stSlider > div {{
        color: {text_color} !important;
    }}
    [data-testid="stSidebar"] .stSlider p {{
        color: {text_color} !important;
    }}

    /* ===== SEARCH BUTTON ===== */
    .stButton > button {{
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        transform: scale(1.02) !important;
        box-shadow: 0 5px 15px rgba(123, 47, 247, 0.4) !important;
        color: white !important;
    }}
    .stButton > button:active,
    .stButton > button:focus {{
        color: white !important;
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 100%) !important;
    }}

    /* ===== TAB STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0 !important;
        background: {tab_bg} !important;
        border-radius: 10px !important;
        padding: 4px !important;
        border: 1px solid {border_color} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {sub_color} !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: {tab_hover} !important;
        color: {text_color} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.15) 0%, rgba(123, 47, 247, 0.15) 100%) !important;
        color: {text_color} !important;
        font-weight: 600 !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background: linear-gradient(90deg, #00d2ff, #7b2ff7) !important;
        height: 3px !important;
        border-radius: 2px !important;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* ===== METRIC CARDS ===== */
    .metric-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border-bottom: 3px solid #7b2ff7;
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(123, 47, 247, 0.15);
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {text_color} !important;
        font-family: 'Outfit', sans-serif;
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: {sub_color} !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }}

    /* ===== ARTICLE SECTIONS ===== */
    .article-section {{
        border-left: 3px solid #7b2ff7;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        background: {card_bg};
        border-radius: 0 8px 8px 0;
        transition: background 0.2s ease;
    }}
    .article-section:hover {{
        background: {card_hover};
    }}
    .article-title {{
        color: {text_color} !important;
        font-weight: 600;
        font-size: 1.15rem;
        margin-bottom: 0.4rem;
        font-family: 'Outfit', sans-serif;
    }}
    .article-meta {{
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.8rem;
    }}
    .article-date {{
        color: #00d2ff !important;
        font-size: 0.82rem;
        font-weight: 500;
    }}
    .article-score {{
        color: {sub_color} !important;
        font-size: 0.78rem;
    }}

    /* ===== 5W1H GRID ===== */
    .w5h1-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.6rem;
    }}
    .w5h1-item {{
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.45rem 0.7rem;
        border-radius: 6px;
        background: {card_bg};
    }}
    .w5h1-badge {{
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
    }}
    .badge-what  {{ background: rgba(0, 210, 255, 0.12); color: #00d2ff !important; }}
    .badge-who   {{ background: rgba(123, 47, 247, 0.12); color: #a78bfa !important; }}
    .badge-when  {{ background: rgba(16, 185, 129, 0.12); color: #10b981 !important; }}
    .badge-where {{ background: rgba(245, 158, 11, 0.12); color: #f59e0b !important; }}
    .badge-why   {{ background: rgba(239, 68, 68, 0.12);  color: #ef4444 !important; }}
    .badge-how   {{ background: rgba(236, 72, 153, 0.12); color: #ec4899 !important; }}
    .w5h1-text {{
        color: {text_color} !important;
        font-size: 0.85rem;
        line-height: 1.4;
    }}
    .w5h1-label {{
        color: #a78bfa !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
        display: block;
    }}

    /* ===== PARAPHRASE BOX ===== */
    .paraphrase-box {{
        background: rgba(0, 210, 255, 0.03);
        border-left: 3px solid #00d2ff;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        color: {text_color} !important;
        font-size: 0.92rem;
        line-height: 1.65;
        height: 100%;
    }}
    .paraphrase-label {{
        color: #00d2ff !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        display: block;
    }}

    /* ===== DATAFRAME / TABLE ===== */
    [data-testid="stDataFrame"] {{
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    [data-testid="stDataFrame"] > div {{
        background: {card_bg} !important;
    }}
    /* Override Glide Data Grid (Streamlit's dataframe renderer) */
    [data-testid="stDataFrame"] canvas {{
        border-radius: 8px !important;
    }}

    /* ===== DIVIDERS / HORIZONTAL RULES ===== */
    hr, .stDivider {{
        border-color: {divider_color} !important;
    }}
    .main hr {{
        border-color: {divider_color} !important;
    }}

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {{
        background: {expander_bg} !important;
        border-radius: 8px !important;
        color: {text_color} !important;
    }}
    details {{
        border-color: {border_color} !important;
    }}
    [data-testid="stExpander"] {{
        border-color: {border_color} !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {text_color} !important;
    }}
    [data-testid="stExpander"] div[role="button"] p {{
        color: {text_color} !important;
    }}

    /* ===== ALERTS (Success, Info, Warning) ===== */
    [data-testid="stAlert"] {{
        border-radius: 8px !important;
    }}
    .stSuccess, .stAlert {{
        color: {text_color} !important;
    }}

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {scrollbar_thumb};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {sub_color};
    }}

    /* ===== TOOLTIP / HELP ICONS ===== */
    [data-testid="stTooltipIcon"] {{
        color: {sub_color} !important;
    }}

    /* ===== BOTTOM TOOLBAR / FOOTER ===== */
    footer {{
        display: none !important;
    }}
    .viewerBadge_container__r5tak {{
        display: none !important;
    }}
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

# --- INISIALISASI (BERTAHAP) ---
df = load_data()
if df is not None:
    bm25_index = load_bm25(df)
    semantic_model, corpus_embeddings = load_semantic(df)
else:
    bm25_index, semantic_model, corpus_embeddings = None, None, None

# --- SIDEBAR: KONFIGURASI & INFO ---
with st.sidebar:
    # --- THEME TOGGLE (Clean single switch) ---
    st.markdown("### 🌗 Tema Tampilan")
    
    # Render visual indicator
    current_icon = "🌙" if is_dark else "☀️"
    current_label = "Mode Gelap" if is_dark else "Mode Terang"
    st.markdown(f"""
    <div class="theme-switch-container">
        <span class="theme-switch-label">{current_label}</span>
        <span class="theme-switch-icon">{current_icon}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Use st.toggle for the actual switch
    use_dark = st.toggle(
        "Mode Gelap",
        value=is_dark,
        label_visibility="collapsed"
    )
    if use_dark != is_dark:
        st.session_state.theme = 'dark' if use_dark else 'light'
        st.rerun()

    st.markdown("---")
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
st.markdown('<div class="sub-header">Platform Pencarian Berita Hybrid dengan Ekstraksi Terstruktur & Ringkasan AI</div>', unsafe_allow_html=True)

if df is None:
    st.stop()

# --- SEARCH BAR ---
search_col1, search_col2 = st.columns([5, 1])
with search_col1:
    query = st.text_input("🔍 Cari berita...", placeholder="Ketik topik berita, tokoh, atau peristiwa...", label_visibility="collapsed")
with search_col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# --- AUTO SWITCH TAB ON SEARCH ---
if search_button and query:
    st.session_state.active_tab = 1
    st.session_state.search_triggered = True

# Determine active tab index
active_tab_index = st.session_state.active_tab

# --- TABS ---
if active_tab_index == 1:
    tab_ringkasan, tab_dataset = st.tabs(["📝 Hasil & Ringkasan", "📂 Dataset"])
else:
    tab_dataset, tab_ringkasan = st.tabs(["📂 Dataset", "📝 Hasil & Ringkasan"])

# --- TAB: DATASET ---
with tab_dataset:
    st.markdown("### Informasi Dataset")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Berita</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">Bulan (Mar-Apr 2023)</div></div>', unsafe_allow_html=True)
    with m_col3:
        avg_len = int(df['content'].str.len().mean())
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_len}</div><div class="metric-label">Rata-rata Karakter</div></div>', unsafe_allow_html=True)
    
    display_df = df.copy()
    st.dataframe(
        display_df[['date', 'title', 'content']],
        use_container_width=True,
        hide_index=True,
        height=400
    )

# --- TAB: HASIL & RINGKASAN ---
with tab_ringkasan:
    if (search_button or st.session_state.search_triggered) and query:
        st.session_state.search_triggered = False

        with st.spinner("🔍 Mencari dokumen yang relevan (Hybrid Search)..."):
            start_time = time.time()
            query_processed = preprocess_for_bm25(query)
            bm25_results = search_bm25(query_processed, bm25_index, top_k=20)
            semantic_results = search_semantic(query, semantic_model, corpus_embeddings, top_k=20)
            hybrid_results = reciprocal_rank_fusion(bm25_results, semantic_results, k=rrf_k, top_k=top_k)
            retrieval_time = time.time() - start_time
            
        st.success(f"Ditemukan {len(hybrid_results)} artikel yang relevan dalam {retrieval_time:.2f} detik.")
        
        retrieved_articles = []
        for doc_idx, rrf_score in hybrid_results:
            row = df.iloc[doc_idx]
            retrieved_articles.append({
                'title': row['title'],
                'date': row['date'],
                'content': row['content'],
                'score': rrf_score
            })

        ai_ready = False
        if api_key_input:
            try:
                configure_gemini(api_key=api_key_input)
                ai_ready = True
            except Exception as e:
                st.error(f"Gagal mengkonfigurasi Gemini API: {e}")

        with st.spinner("🔬 Mengekstrak informasi terstruktur (5W1H)..."):
            start_extract_time = time.time()
            all_5w1h = []
            for art in retrieved_articles:
                article_data = {'title': art['title'], 'date': art['date'], 'content': art['content']}
                all_5w1h.append(extract_5w1h(article_data))
            extract_time = time.time() - start_extract_time
        st.caption(f"⚙️ Waktu ekstraksi: {extract_time:.3f}s untuk {len(all_5w1h)} artikel")

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
            st.info("ℹ️ Kunci API Gemini belum dimasukkan. Ringkasan ditampilkan dalam format standar. Masukkan Kunci API di sidebar untuk ringkasan yang lebih optimal dan natural.")
            for art, w5h1 in zip(retrieved_articles, all_5w1h):
                all_paraphrases.append(_fallback_paragraph(w5h1, art['title']))

        st.markdown(f"### 📑 Top {len(retrieved_articles)} Artikel Relevan")

        for i, art in enumerate(retrieved_articles):
            item = all_5w1h[i]
            paragraph = all_paraphrases[i] if i < len(all_paraphrases) else ""

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
                def _fmt(val):
                    return val if val and "Tidak disebutkan" not in val else "Tidak terdeteksi"

                w5h1_html = (
'<div class="w5h1-grid" style="grid-template-columns:1fr;">'
'<span class="w5h1-label">🔍 Informasi Terstruktur (5W+1H)</span>'
'<div class="w5h1-item"><span class="w5h1-badge badge-what">WHAT</span>'
f'<span class="w5h1-text">{_fmt(item.get("what"))}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-who">WHO</span>'
f'<span class="w5h1-text">{_fmt(item.get("who"))}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-when">WHEN</span>'
f'<span class="w5h1-text">{_fmt(item.get("when"))}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-where">WHERE</span>'
f'<span class="w5h1-text">{_fmt(item.get("where"))}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-why">WHY</span>'
f'<span class="w5h1-text">{_fmt(item.get("why"))}</span></div>'
'<div class="w5h1-item"><span class="w5h1-badge badge-how">HOW</span>'
f'<span class="w5h1-text">{_fmt(item.get("how"))}</span></div>'
'</div>'
                )
                st.markdown(w5h1_html, unsafe_allow_html=True)

            with col_right:
                paraphrase_html = (
'<div class="paraphrase-box">'
'<span class="paraphrase-label">📝 Ringkasan Artikel</span>'
f'{paragraph}'
'</div>'
                )
                st.markdown(paraphrase_html, unsafe_allow_html=True)

            with st.expander("📖 Baca Selengkapnya"):
                st.write(art['content'])

            st.markdown("---")

    elif search_button and not query:
        st.warning("Silakan masukkan kata kunci pencarian terlebih dahulu.")
    else:
        st.info("👈 Masukkan kata kunci pencarian dan klik 'Search' untuk melihat hasil.")
        
        st.markdown(f"""
        <div style="text-align: center; opacity: 0.5; padding: 4rem;">
            <div style="font-size: 5rem; margin-bottom: 1rem;">🔍</div>
            <h3 style="font-family: 'Outfit', sans-serif; color: {text_color} !important;">Menunggu Kueri Pencarian</h3>
            <p style="color: {sub_color} !important;">Sistem siap mencari dan meringkas ratusan berita dalam hitungan detik.</p>
        </div>
        """, unsafe_allow_html=True)

# Reset active tab to dataset when no search is active
if not query:
    st.session_state.active_tab = 0
