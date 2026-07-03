import streamlit as st
import pandas as pd
import time

from preprocess import preprocess_for_bm25
from bm25_search import build_bm25_index, search_bm25
from semantic_search import load_embedding_model, encode_corpus, search_semantic
from hybrid_search import reciprocal_rank_fusion
from extraction.rule_based_5w1h import extract_5w1h, inject_ner_pipeline
from paraphraser import configure_gemini, paraphrase_5w1h, _fallback_paragraph

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="ChromoNews | Hybrid Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE
# =============================================================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'top_k' not in st.session_state:
    st.session_state.top_k = 5
if 'rrf_k' not in st.session_state:
    st.session_state.rrf_k = 60

# =============================================================================
# THEME COLORS
# =============================================================================
is_dark = st.session_state.theme == 'dark'

if is_dark:
    # Dark Mode
    bg_primary = "#0a0e17"
    bg_secondary = "#111827"
    bg_card = "rgba(255, 255, 255, 0.04)"
    bg_card_hover = "rgba(255, 255, 255, 0.08)"
    bg_input = "rgba(255, 255, 255, 0.06)"
    text_primary = "#e2e8f0"
    text_secondary = "#94a3b8"
    text_muted = "#64748b"
    border_color = "rgba(255, 255, 255, 0.08)"
    gradient_start = "#00d2ff"
    gradient_end = "#7b2ff7"
    shadow = "0 4px 20px rgba(123, 47, 247, 0.15)"
else:
    # Light Mode
    bg_primary = "#f0f4f8"
    bg_secondary = "#ffffff"
    bg_card = "rgba(0, 0, 0, 0.02)"
    bg_card_hover = "rgba(0, 0, 0, 0.05)"
    bg_input = "#ffffff"
    text_primary = "#1a1a2e"
    text_secondary = "#64748b"
    text_muted = "#94a3b8"
    border_color = "rgba(0, 0, 0, 0.08)"
    gradient_start = "#00d2ff"
    gradient_end = "#7b2ff7"
    shadow = "0 4px 20px rgba(0, 0, 0, 0.08)"

# =============================================================================
# CSS - FRESH & MODERN
# =============================================================================
st.markdown(f"""
<style>
    /* ===== IMPORTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    /* ===== RESET ===== */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        color: {text_primary} !important;
    }}

    .stApp, .stApp > header, .main .block-container {{
        background-color: {bg_primary} !important;
        color: {text_primary} !important;
    }}

    /* ===== HEADER ===== */
    .app-header {{
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }}
    .app-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 0.25rem;
    }}
    .app-subtitle {{
        font-size: 1.1rem;
        color: {text_secondary} !important;
        font-weight: 300;
        letter-spacing: 0.5px;
    }}

    /* ===== SEARCH BAR ===== */
    .search-container {{
        background: {bg_secondary};
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid {border_color};
        box-shadow: {shadow};
        transition: all 0.3s ease;
        margin: 1rem 0;
    }}
    .search-container:focus-within {{
        border-color: {gradient_start};
        box-shadow: 0 0 0 3px rgba(0, 210, 255, 0.15);
    }}
    .search-input {{
        border: none !important;
        background: transparent !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
        color: {text_primary} !important;
    }}
    .search-input:focus {{
        box-shadow: none !important;
    }}
    .search-btn {{
        background: linear-gradient(135deg, {gradient_start}, {gradient_end}) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        cursor: pointer;
    }}
    .search-btn:hover {{
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(123, 47, 247, 0.4);
    }}

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {{
        background-color: {bg_secondary} !important;
        border-right: 1px solid {border_color} !important;
        padding: 1.5rem 0.5rem;
    }}
    [data-testid="stSidebar"] * {{
        color: {text_primary} !important;
    }}
    .sidebar-section {{
        background: {bg_card};
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid {border_color};
    }}
    .sidebar-section:hover {{
        background: {bg_card_hover};
    }}

    /* ===== THEME TOGGLE ===== */
    .theme-toggle {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 1rem;
        background: {bg_card};
        border: 1px solid {border_color};
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }}
    .theme-toggle-label {{
        font-weight: 500;
        font-size: 0.9rem;
    }}
    .theme-toggle-icon {{
        font-size: 1.3rem;
    }}

    /* ===== METRIC CARDS ===== */
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }}
    .metric-card {{
        background: {bg_secondary};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: {shadow};
    }}
    .metric-value {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: {text_primary} !important;
    }}
    .metric-label {{
        font-size: 0.8rem;
        color: {text_secondary} !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }}

    /* ===== ARTICLE CARD ===== */
    .article-card {{
        background: {bg_secondary};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .article-card:hover {{
        border-color: {gradient_start};
        box-shadow: {shadow};
    }}
    .article-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: {text_primary} !important;
        margin-bottom: 0.3rem;
    }}
    .article-meta {{
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }}
    .article-date {{
        color: {gradient_start} !important;
        font-size: 0.82rem;
        font-weight: 500;
    }}
    .article-score {{
        color: {text_secondary} !important;
        font-size: 0.78rem;
        background: {bg_card};
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
    }}

    /* ===== 5W1H GRID ===== */
    .w5h1-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.4rem;
    }}
    .w5h1-item {{
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.4rem 0.6rem;
        border-radius: 8px;
        background: {bg_card};
        transition: background 0.2s;
    }}
    .w5h1-item:hover {{
        background: {bg_card_hover};
    }}
    .w5h1-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 38px;
        height: 20px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.55rem;
        font-family: 'Outfit', sans-serif;
        flex-shrink: 0;
        letter-spacing: 0.5px;
    }}
    .badge-what  {{ background: rgba(0, 210, 255, 0.15); color: #00d2ff !important; }}
    .badge-who   {{ background: rgba(123, 47, 247, 0.15); color: #a78bfa !important; }}
    .badge-when  {{ background: rgba(16, 185, 129, 0.15); color: #10b981 !important; }}
    .badge-where {{ background: rgba(245, 158, 11, 0.15); color: #f59e0b !important; }}
    .badge-why   {{ background: rgba(239, 68, 68, 0.15); color: #ef4444 !important; }}
    .badge-how   {{ background: rgba(236, 72, 153, 0.15); color: #ec4899 !important; }}
    .w5h1-text {{
        font-size: 0.82rem;
        line-height: 1.4;
        color: {text_primary} !important;
    }}
    .w5h1-label {{
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: {text_secondary} !important;
        margin-bottom: 0.3rem;
        display: block;
    }}

    /* ===== PARAPHRASE BOX ===== */
    .paraphrase-box {{
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.04), rgba(123, 47, 247, 0.04));
        border-left: 3px solid {gradient_start};
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        font-size: 0.9rem;
        line-height: 1.7;
        height: 100%;
        min-height: 100px;
    }}
    .paraphrase-label {{
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: {gradient_start} !important;
        margin-bottom: 0.3rem;
        display: block;
    }}

    /* ===== EMPTY STATE ===== */
    .empty-state {{
        text-align: center;
        padding: 4rem 2rem;
        opacity: 0.6;
    }}
    .empty-state-icon {{
        font-size: 4rem;
        margin-bottom: 1rem;
    }}
    .empty-state-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: {text_primary} !important;
    }}
    .empty-state-desc {{
        color: {text_secondary} !important;
        max-width: 400px;
        margin: 0.5rem auto;
    }}

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {{
        background: {bg_card} !important;
        border-radius: 8px !important;
        color: {text_primary} !important;
        font-weight: 500 !important;
    }}
    .streamlit-expanderHeader:hover {{
        background: {bg_card_hover} !important;
    }}

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {text_muted};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {text_secondary};
    }}

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {{
        .app-title {{ font-size: 2.5rem; }}
        .metric-grid {{ grid-template-columns: 1fr; }}
        .w5h1-grid {{ grid-template-columns: 1fr; }}
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNGSI CACHING
# =============================================================================
@st.cache_resource(show_spinner="📄 Memuat Dataset...")
def load_data():
    try:
        df = pd.read_csv('preprocessed_news_sample.csv')
        return df
    except FileNotFoundError:
        st.error("❌ File 'preprocessed_news_sample.csv' tidak ditemukan. Jalankan preprocess.py terlebih dahulu.")
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
    gc.collect()
    return model, corpus_embeddings

@st.cache_resource(show_spinner="🏷️ Memuat Model NER Indonesia...")
def load_ner_model():
    try:
        from extraction.ner_model import load_ner_pipeline
        pipe = load_ner_pipeline()
        return pipe
    except Exception as e:
        st.session_state['ner_load_error'] = str(e)
        return None

# =============================================================================
# LOAD DATA
# =============================================================================
df = load_data()
if df is None:
    st.stop()

bm25_index = load_bm25(df)
semantic_model, corpus_embeddings = load_semantic(df)

ner_pipeline = load_ner_model()
if ner_pipeline is not None:
    try:
        inject_ner_pipeline(ner_pipeline)
    except Exception:
        pass

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🎨 Tampilan")
    
    # Theme Toggle
    current_icon = "🌙" if is_dark else "☀️"
    current_label = "Mode Gelap" if is_dark else "Mode Terang"
    st.markdown(f"""
    <div class="theme-toggle">
        <span class="theme-toggle-label">{current_label}</span>
        <span class="theme-toggle-icon">{current_icon}</span>
    </div>
    """, unsafe_allow_html=True)
    
    use_dark = st.toggle("Mode Gelap", value=is_dark, label_visibility="collapsed")
    if use_dark != is_dark:
        st.session_state.theme = 'dark' if use_dark else 'light'
        st.rerun()
    
    st.markdown("---")
    
    # API Key
    st.markdown("### ⚙️ Konfigurasi")
    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="Masukkan API Key...",
        help="Diperlukan untuk ringkasan AI yang natural",
        key="api_key_input"
    )
    if api_key:
        st.session_state.api_key = api_key
    
    # NER Status
    st.markdown("---")
    st.markdown("### 🏷️ Status NER")
    ner_ok = ner_pipeline is not None
    ner_icon = "✅" if ner_ok else "⚠️"
    ner_label = "Model NER Aktif" if ner_ok else "Fallback (Rule-based)"
    ner_color = "#10b981" if ner_ok else "#f59e0b"
    st.markdown(f"""
    <div style="background:{bg_card};border-radius:8px;padding:8px 12px;border:1px solid {border_color};">
        <span style="color:{ner_color};font-weight:600;">{ner_icon} {ner_label}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Search Parameters
    st.markdown("### 📊 Parameter")
    
    top_k = st.slider(
        "Jumlah Artikel",
        min_value=3,
        max_value=15,
        value=st.session_state.top_k,
        help="Jumlah artikel teratas yang ditampilkan"
    )
    st.session_state.top_k = top_k
    
    rrf_k = st.number_input(
        "RRF K Constant",
        min_value=1,
        max_value=100,
        value=st.session_state.rrf_k,
        help="Smoothing constant untuk RRF"
    )
    st.session_state.rrf_k = rrf_k
    
    st.markdown("---")
    
    # Contoh Query
    st.markdown("### 💡 Contoh")
    st.markdown("""
    • `pilkada indonesia`  
    • `kenaikan harga minyak goreng`  
    • `timnas indonesia`  
    • `kasus korupsi 2024`  
    """)

# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown("""
<div class="app-header">
    <div class="app-title">🧬 ChromoNews</div>
    <div class="app-subtitle">Platform Pencarian Berita Hybrid dengan Ekstraksi Terstruktur &amp; Ringkasan AI</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SEARCH BAR
# =============================================================================
col1, col2 = st.columns([6, 1])
with col1:
    query = st.text_input(
        "🔍 Cari berita...",
        value=st.session_state.query,
        placeholder="Ketik topik berita, tokoh, atau peristiwa...",
        label_visibility="collapsed",
        key="search_input"
    )
with col2:
    search_clicked = st.button("🔍 Cari", use_container_width=True, type="primary")

if search_clicked and query:
    st.session_state.query = query
    st.session_state.search_performed = True

# =============================================================================
# TABS
# =============================================================================
tab_hasil, tab_dataset = st.tabs(["📝 Hasil & Ringkasan", "📂 Dataset"])

# =============================================================================
# TAB 1: HASIL & RINGKASAN
# =============================================================================
with tab_hasil:
    if st.session_state.search_performed and query:
        with st.spinner("🔍 Mencari dokumen yang relevan..."):
            start_time = time.time()
            
            query_processed = preprocess_for_bm25(query)
            bm25_results = search_bm25(query_processed, bm25_index, top_k=20)
            semantic_results = search_semantic(query, semantic_model, corpus_embeddings, top_k=20)
            hybrid_results = reciprocal_rank_fusion(bm25_results, semantic_results, k=st.session_state.rrf_k, top_k=st.session_state.top_k)
            
            retrieval_time = time.time() - start_time
        
        st.success(f"✅ Ditemukan **{len(hybrid_results)}** artikel relevan dalam **{retrieval_time:.2f}** detik")
        
        # Ekstrak 5W1H
        with st.spinner("🔬 Mengekstrak informasi terstruktur..."):
            extract_start = time.time()
            retrieved_articles = []
            for doc_idx, rrf_score in hybrid_results:
                row = df.iloc[doc_idx]
                article_data = {
                    'title': row['title'],
                    'content': row['content'],
                    'date': row['date']
                }
                w5h1 = extract_5w1h(article_data)
                retrieved_articles.append({
                    'title': row['title'],
                    'date': row['date'],
                    'content': row['content'],
                    'score': rrf_score,
                    'w5h1': w5h1,
                })
            extract_time = time.time() - extract_start
        
        # AI Paraphrase
        ai_ready = False
        if st.session_state.api_key:
            try:
                configure_gemini(api_key=st.session_state.api_key)
                ai_ready = True
            except Exception as e:
                st.warning(f"⚠️ Gemini API: {str(e)[:80]}...")
        
        with st.spinner("📝 Merangkai ringkasan..."):
            for art in retrieved_articles:
                if ai_ready:
                    try:
                        paragraph = paraphrase_5w1h(art['w5h1'], title=art['title'], query=query, content=art['content'])
                    except Exception:
                        paragraph = _fallback_paragraph(art['w5h1'], art['title'], art['content'])
                else:
                    paragraph = _fallback_paragraph(art['w5h1'], art['title'], art['content'])
                art['paragraph'] = paragraph
        
        # ===== TAMPILAN HASIL =====
        st.markdown(f"### 📑 Top {len(retrieved_articles)} Artikel")
        st.caption(f"⚙️ Ekstraksi: {extract_time:.2f}s • AI: {'✅ Aktif' if ai_ready else '⏸️ Nonaktif'}")
        
        for i, art in enumerate(retrieved_articles):
            w5h1 = art['w5h1']
            
            # ===== ARTICLE CARD =====
            st.markdown(f"""
            <div class="article-card">
                <div class="article-title">{i+1}. {art['title']}</div>
                <div class="article-meta">
                    <span class="article-date">🕒 {art['date']}</span>
                    <span class="article-score">📊 RRF: {art['score']:.4f}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # ===== 5W1H + PARAPHRASE =====
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                def fmt(v):
                    if not v or "Tidak disebutkan" in str(v):
                        return "<em style='color:{};'>Tidak terdeteksi</em>".format(text_muted)
                    if isinstance(v, list):
                        cleaned = [x for x in v if x and "Tidak disebutkan" not in x]
                        if not cleaned:
                            return "<em style='color:{};'>Tidak terdeteksi</em>".format(text_muted)
                        return " • ".join(cleaned)
                    return str(v)
                
                st.markdown(f"""
                <span class="w5h1-label">🔍 Informasi Terstruktur</span>
                <div class="w5h1-grid">
                    <div class="w5h1-item"><span class="w5h1-badge badge-what">WHAT</span>
                        <span class="w5h1-text">{fmt(w5h1.get('what'))}</span></div>
                    <div class="w5h1-item"><span class="w5h1-badge badge-who">WHO</span>
                        <span class="w5h1-text">{fmt(w5h1.get('who'))}</span></div>
                    <div class="w5h1-item"><span class="w5h1-badge badge-when">WHEN</span>
                        <span class="w5h1-text">{fmt(w5h1.get('when'))}</span></div>
                    <div class="w5h1-item"><span class="w5h1-badge badge-where">WHERE</span>
                        <span class="w5h1-text">{fmt(w5h1.get('where'))}</span></div>
                    <div class="w5h1-item"><span class="w5h1-badge badge-why">WHY</span>
                        <span class="w5h1-text">{fmt(w5h1.get('why'))}</span></div>
                    <div class="w5h1-item"><span class="w5h1-badge badge-how">HOW</span>
                        <span class="w5h1-text">{fmt(w5h1.get('how'))}</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_right:
                st.markdown(f"""
                <div class="paraphrase-box">
                    <span class="paraphrase-label">📝 Ringkasan {'' if ai_ready else '(Fallback)'}</span>
                    {art['paragraph']}
                </div>
                """, unsafe_allow_html=True)
            
            # ===== EXPANDER =====
            with st.expander("📖 Baca Selengkapnya"):
                st.write(art['content'])
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")
        
        st.session_state.search_performed = False
    
    else:
        # ===== EMPTY STATE =====
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-title">Menunggu Kueri Pencarian</div>
            <div class="empty-state-desc">
                Masukkan kata kunci di atas untuk mencari berita.<br>
                Sistem akan menampilkan ringkasan 5W+1H dan AI paraphrase.
            </div>
            <div style="margin-top: 1rem; font-size: 0.85rem; color: {text_muted};">
                💡 Coba: <strong>korupsi KPK</strong> &bull; <strong>Rafael Alun</strong> &bull; <strong>mudik 2023</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: DATASET
# =============================================================================
with tab_dataset:
    st.markdown("### 📊 Informasi Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Berita</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">2</div>
            <div class="metric-label">Bulan</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_len = int(df['content'].str.len().mean())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_len:,}</div>
            <div class="metric-label">Rata-rata Karakter</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.dataframe(
        df[['date', 'title', 'content']],
        use_container_width=True,
        hide_index=True,
        height=400
    )

# =============================================================================
# RESET
# =============================================================================
if not query:
    st.session_state.search_performed = False