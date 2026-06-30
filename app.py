import os
# Thiết lập cấu hình protobuf để sửa lỗi tương thích phiên bản (TypeError: Descriptors cannot be created directly)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tempfile
from dotenv import load_dotenv

# Nạp các thành phần từ hệ thống RAG
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.yt_loader import YouTubeLoader
from src.retrieval.vector_db import get_vector_db_singleton
from src.graph.workflow import build_rag_graph

import pandas as pd
from src.utils.benchmark import (
    run_benchmark_for_config,
    run_ragas_evaluation,
    run_docling_benchmark,
    run_youtube_benchmark,
    GOLDEN_DATASET
)

# Nạp biến môi trường (API Key...)
load_dotenv()

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Multimedia RAG Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tải chậm (Lazy Load) các dịch vụ và lưu vào cache để cải thiện hiệu năng
@st.cache_resource
def get_vector_db():
    return get_vector_db_singleton()

@st.cache_resource
def get_pdf_loader():
    return PDFLoader()

@st.cache_resource
def get_yt_loader():
    return YouTubeLoader(model_size="medium")

@st.cache_resource
def get_rag_app():
    return build_rag_graph()

# Khởi tạo dịch vụ
vector_db = get_vector_db()
pdf_loader = get_pdf_loader()
yt_loader = get_yt_loader()
rag_app = get_rag_app()

# Custom CSS tối giản, hiện đại (Hạn chế tối đa gradient lòe loẹt theo góp ý của người dùng)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* =========================================================================
       GLOBAL BACKGROUND & TEXT (Forcing Dark Theme to prevent Light Theme leaks)
       ========================================================================= */
    .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stMain"],
    .main,
    .block-container {
        background-color: #0a0f1d !important;
        color: #ffffff !important;
        font-family: 'Outfit', -apple-system, sans-serif;
        scroll-behavior: auto !important;
        overflow-anchor: none !important;
    }

    /* Force scroll stability on root elements */
    html, body {
        scroll-behavior: auto !important;
        overflow-anchor: none !important;
    }

    /* Force all text nodes to be white by default */
    p, span, label, li, h1, h2, h3, h4, h5, h6, .stMarkdown, .stWrite {
        color: #ffffff !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0f1d;
    }
    ::-webkit-scrollbar-thumb {
        background: #223156;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #38bdf8;
    }

    /* =========================================================================
       SIDEBAR & BRANDING
       ========================================================================= */
    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] > div {
        background-color: #0f1626 !important;
        border-right: 1px solid #223156 !important;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        padding-bottom: 16px;
        border-bottom: 1px solid #223156;
        margin-bottom: 16px;
    }
    .sidebar-brand span {
        color: #ffffff !important;
    }
    
    .sidebar-header span {
        color: #ffffff !important;
    }

    /* =========================================================================
       WIDGETS & CONTAINERS
       ========================================================================= */
    /* Expander / Source Section */
    [data-testid="stExpander"] {
        background-color: #151f38 !important;
        border: 1px solid #223156 !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        box-shadow: none !important;
    }
    [data-testid="stExpander"] details {
        background-color: #151f38 !important;
        border: none !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #151f38 !important;
        color: #ffffff !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #38bdf8 !important;
    }
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background-color: #151f38 !important;
        color: #ffffff !important;
    }
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] * {
        color: #ffffff !important;
    }

    /* Buttons (Pill style, solid dark background to ensure contrast) */
    .stButton > button {
        background-color: #151f38 !important;
        color: #ffffff !important;
        border: 1px solid #223156 !important;
        border-radius: 20px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #1e2d52 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }
    .stButton > button * {
        color: #ffffff !important;
    }
    
    /* Secondary/Danger buttons (e.g., Clear DB) */
    .stButton button[kind="secondary"] {
        background-color: rgba(239, 68, 68, 0.1) !important;
        color: #ff8a80 !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
    }
    .stButton button[kind="secondary"]:hover {
        background-color: rgba(239, 68, 68, 0.2) !important;
        border-color: #ff8a80 !important;
        color: #ffffff !important;
    }
    .stButton button[kind="secondary"] * {
        color: #ff8a80 !important;
    }

    /* Input Fields (Explicit backgrounds & text-fill-color) */
    div[data-baseweb="input"], 
    div[data-baseweb="input"] > div {
        background-color: #151f38 !important;
        border: 1px solid #223156 !important;
        border-radius: 20px !important;
        color: #ffffff !important;
        transition: all 0.2s !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    }
    div[data-baseweb="input"] input {
        color: #ffffff !important;
        background-color: transparent !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* File Uploader Dropzone */
    [data-testid="stFileUploader"] {
        background-color: #151f38 !important;
        border-radius: 12px !important;
        padding: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #0f1626 !important;
        border: 2px dashed #223156 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        transition: all 0.2s !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #38bdf8 !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    /* Checkboxes */
    label[data-baseweb="checkbox"] * {
        color: #ffffff !important;
    }

    /* Radio Buttons */
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background-color: #151f38 !important;
        border: 1px solid #223156 !important;
        padding: 6px 14px !important;
        border-radius: 20px !important;
        color: #cbd5e1 !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        border-color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.04) !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] * {
        color: #cbd5e1 !important;
    }
    /* Radio checked state label color */
    div[data-testid="stRadio"] label[data-baseweb="radio"] [checked] ~ div,
    div[data-testid="stRadio"] label[data-baseweb="radio"] [checked] ~ div * {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }

    /* =========================================================================
       CHAT & MESSAGES
       ========================================================================= */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 16px 0px !important;
        margin-bottom: 8px !important;
        color: #ffffff !important;
    }
    .stChatMessage * {
        color: #ffffff !important;
    }

    /* NotebookLM Action Bar */
    .notebook-actions {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 12px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .notebook-actions .action-item {
        cursor: pointer;
        border: 1px solid #223156;
        border-radius: 16px;
        padding: 4px 12px;
        transition: all 0.2s;
        display: inline-flex;
        align-items: center;
        background-color: rgba(255, 255, 255, 0.03) !important;
    }
    .notebook-actions .action-item:hover {
        background-color: rgba(56, 189, 248, 0.1) !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
    }
    .notebook-actions .action-item *,
    .notebook-actions .action-icon {
        color: #cbd5e1 !important;
    }
    .notebook-actions .action-item:hover * {
        color: #ffffff !important;
    }

    /* Sources / Citations Card */
    .citation-block {
        background-color: #0f1626 !important;
        border: 1px solid #223156 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin: 10px 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
    }
    .citation-block * {
        color: #ffffff !important;
    }
    .source-meta {
        font-weight: 500;
        color: #38bdf8 !important;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .source-meta * {
        color: #38bdf8 !important;
    }
    .source-meta .source-icon {
        font-size: 1.1rem;
        color: #ff8a80 !important;
    }
    .source-meta .source-name {
        font-weight: 600;
        color: #ffffff !important;
    }
    .source-meta .source-type {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .source-meta .source-location {
        color: #9ee5b4 !important;
        font-size: 0.8rem;
    }
    .source-content {
        color: #cbd5e1 !important;
        line-height: 1.5;
        border-top: 1px solid #223156 !important;
        padding-top: 8px;
        margin-top: 8px;
    }

    /* Style bottom containers to be transparent and borderless */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    .stChatInputContainer {
        background-color: transparent !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Chat Input Bar (NotebookLM Pill Style) */
    [data-testid="stChatInput"],
    .stChatInput {
        background-color: #151f38 !important;
        border: 1px solid #223156 !important;
        border-radius: 28px !important;
        padding: 8px 16px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        max-width: 1000px !important;
        margin: 0 auto 20px auto !important;
        width: 100% !important;
    }

    /* Prevent messages from being hidden behind the fixed chat input */
    [data-testid="stMain"] .block-container {
        padding-bottom: 120px !important;
    }
    
    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="textarea"] > div {
        background-color: transparent !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 0.95rem !important;
        caret-color: #38bdf8 !important;
        border: none !important;
        resize: none !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #223156 !important;
        border-radius: 50% !important;
        color: #ffffff !important;
        border: none !important;
        transition: all 0.2s !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: #38bdf8 !important;
        color: #0a0f1d !important;
    }

    /* Popover Menu Styling */
    div[data-testid="stPopover"] > button {
        background-color: transparent !important;
        color: #ffffff !important;
        border: none !important;
        padding: 4px 8px !important;
        font-size: 1.1rem !important;
        width: auto !important;
        min-width: 0px !important;
        border-radius: 4px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stPopover"] > button:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hàm lưu tệp tải lên vào thư mục tạm thời
def save_uploaded_file(uploaded_file):
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    except Exception as e:
        st.sidebar.error(f"Lỗi lưu file tạm: {str(e)}")
        return None

# ==========================================
# 🛠️ SIDEBAR: QUẢN LÝ NẠP DỮ LIỆU (INGESTION)
# ==========================================
st.sidebar.markdown("""
<div class="sidebar-brand">
    <span style="font-weight: 700; font-size: 1.25rem; color: #ffffff; vertical-align: middle;">Multimedia ChatBot</span>
</div>
""", unsafe_allow_html=True)

# Lựa chọn Menu chính
menu_option = st.sidebar.selectbox(
    "Menu:",
    ["💬 Trò chuyện", "📊 Benchmark & Đánh giá"],
    index=0
)

st.sidebar.markdown("""
<div class="sidebar-header" style="margin-top: 16px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 1.1rem; font-weight: 500; color: #ffffff;">Nguồn</span>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H12V5H19V19Z" fill="#38bdf8"/>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)

# 0. PHÂN HỆ HIỂN THỊ TÀI LIỆU ĐÃ CÓ TRONG CSDL
existing_sources = vector_db.get_all_sources()
with st.sidebar.expander(f"📚 Tài liệu trong hệ thống ({len(existing_sources)})", expanded=True):
    if not existing_sources:
        st.markdown("<div style='font-size: 0.85rem; color: #cbd5e1;'>Chưa có tài liệu nào trong CSDL.</div>", unsafe_allow_html=True)
    else:
        for idx, src in enumerate(existing_sources):
            col_name, col_menu = st.columns([0.8, 0.2])
            
            orig_name = src["source_name"]
            display_name = orig_name
            if len(display_name) > 28:
                display_name = display_name[:25] + "..."
                
            col_name.markdown(
                f"<div style='font-size: 0.85rem; padding-top: 6px; color: #ffffff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;' title='{orig_name}'>{display_name}</div>", 
                unsafe_allow_html=True
            )
            
            with col_menu.popover("⋮", help="Tùy chọn"):
                # Option 1: Đổi tên
                st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #ffffff; margin-bottom: 2px;'>✏️ Đổi tên tài liệu</div>", unsafe_allow_html=True)
                new_name = st.text_input(
                    "Tên mới:", 
                    value=orig_name, 
                    key=f"rename_input_{src['source_type']}_{idx}",
                    label_visibility="collapsed"
                )
                if st.button("Lưu tên mới", key=f"btn_rename_{src['source_type']}_{idx}", use_container_width=True):
                    if new_name and new_name != orig_name:
                        with st.spinner("Đang đổi tên..."):
                            if vector_db.rename_source(orig_name, new_name, src["source_type"]):
                                st.success("Đã đổi tên!")
                                st.rerun()
                            else:
                                st.error("Đổi tên thất bại!")
                    else:
                        st.warning("Tên mới không hợp lệ hoặc trùng tên cũ.")
                
                st.markdown("<hr style='margin: 8px 0; border-color: #223156;' />", unsafe_allow_html=True)
                
                # Option 2: Xóa
                st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #ffffff; margin-bottom: 4px;'>🗑️ Xóa tài liệu</div>", unsafe_allow_html=True)
                if st.button("Xóa tài liệu", key=f"btn_del_{src['source_type']}_{idx}", type="secondary", use_container_width=True):
                    with st.spinner("Đang xóa..."):
                        if vector_db.delete_source(orig_name, src["source_type"]):
                            st.success("Đã xóa!")
                            st.rerun()
                        else:
                            st.error("Xóa thất bại!")

if menu_option == "💬 Trò chuyện":
    # 1. PHÂN HỆ NẠP TÀI LIỆU PDF
    with st.sidebar.expander("Tải file lên", expanded=True):
        pdf_file = st.file_uploader("Kéo thả tệp PDF vào đây", type="pdf")
        
        if pdf_file:
            source_name = pdf_file.name
            source_type = "pdf"
            
            # Kiểm tra trùng lặp bằng metadata
            exists = vector_db.check_source_exists(source_name, source_type)
            
            overwrite = False
            if exists:
                st.warning("⚠️ Tài liệu này đã tồn tại trong CSDL.")
                overwrite = st.checkbox("Ghi đè (Xóa dữ liệu cũ và nạp lại)", value=False, key="pdf_overwrite")
                
            # Nếu chưa tồn tại HOẶC người dùng đồng ý ghi đè
            if not exists or overwrite:
                if st.button("Nạp PDF", use_container_width=True):
                    temp_path = save_uploaded_file(pdf_file)
                    if temp_path:
                        with st.spinner("⏳ Đang phân tích file..."):
                            try:
                                # 1. Phân tích PDF theo trang thông qua Docling
                                pages = pdf_loader.load_and_convert(temp_path)
                                
                                # 2. Lưu các trang kèm metadata số trang vào ChromaDB
                                vector_db.add_documents(
                                    text=pages,
                                    source_type=source_type,
                                    source_name=source_name
                                )
                                st.toast("✅ Nạp file thành công!", icon="✅")
                                st.success("✅ Nạp file thành công!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Thất bại khi nạp file: {str(e)}")
                            finally:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)

    # 2. PHÂN HỆ NẠP VIDEO YOUTUBE
    with st.sidebar.expander("Tải video lên", expanded=False):
        yt_url = st.text_input("Nhập link Video YouTube", placeholder="URL Video...")
        
        if yt_url:
            # Trích xuất video_id để làm tiêu đề mặc định hoặc dùng check trùng lặp sơ bộ
            video_id = None
            if "v=" in yt_url:
                video_id = yt_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in yt_url:
                video_id = yt_url.split("youtu.be/")[1].split("?")[0]
            else:
                video_id = yt_url
                
            default_title = video_id if video_id else "YouTube Video"
            yt_title = st.text_input("Tiêu đề cho Video", value=default_title)
            
            # Kiểm tra trùng lặp
            exists = vector_db.check_source_exists(yt_title, "youtube")
            
            overwrite = False
            if exists:
                st.warning("⚠️ Video này đã tồn tại trong CSDL.")
                overwrite = st.checkbox("Ghi đè (Xóa dữ liệu cũ và nạp lại)", value=False, key="yt_overwrite")
                
            # Nếu chưa tồn tại HOẶC đồng ý ghi đè
            if not exists or overwrite:
                if st.button("Tải video", use_container_width=True):
                    with st.spinner("⏳ Đang tải video"):
                        try:
                            # 1. Trích xuất phụ đề dạng mảnh có nhãn thời gian
                            chunks = yt_loader.load_video(yt_url)
                            
                            if chunks and isinstance(chunks, list):
                                # 2. Lưu vào ChromaDB
                                vector_db.add_documents(
                                    text=chunks,
                                    source_type="youtube",
                                    source_name=yt_title,
                                    extra_metadata={"video_id": video_id}
                                )
                                st.toast("✅ Nạp Video YouTube thành công!", icon="✅")
                                st.success("✅ Nạp Video YouTube thành công!")
                                st.rerun()
                            else:
                                st.error("❌ Không lấy được dữ liệu phụ đề từ Video này.")
                        except Exception as e:
                            st.error(f"❌ Lỗi nạp dữ liệu YouTube: {str(e)}")

    # 3. DỌN DẸP HỆ THỐNG
    with st.sidebar.expander("Cấu hình Hệ thống", expanded=False):
        st.markdown("Xóa toàn bộ dữ liệu trong cơ sở dữ liệu.")
        if st.button("Xóa sạch Vector Database", type="secondary", use_container_width=True):
            with st.spinner("Đang xóa sạch dữ liệu..."):
                try:
                    vector_db.clear_db()
                    st.session_state.messages = []
                    st.toast("🧹 Đã dọn dẹp sạch sẽ cơ sở dữ liệu!", icon="🧹")
                    st.success("Đã dọn dẹp sạch sẽ cơ sở dữ liệu!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi dọn dẹp database: {str(e)}")
else:
    with st.sidebar.expander("Thông tin Benchmark", expanded=True):
        st.markdown("""
        <div style='font-size: 0.85rem; color: #cbd5e1;'>
        Hệ thống đánh giá hiệu năng và chất lượng RAG toàn diện:<br><br>
        1. <b>Thời gian phản hồi (Latency)</b>: Đo đạc chi tiết tốc độ truy xuất vector, tốc độ xếp hạng lại (Reranking) và tốc độ sinh (TTFT, Total Gen).<br><br>
        2. <b>Tài nguyên & Chi phí (Resource)</b>: Kiểm tra số lượng input/output tokens và đo Peak RAM/GPU khi chạy Docling & Faster-Whisper.<br><br>
        3. <b>Chất lượng RAG (Ragas)</b>: Đánh giá độ trung thực (Faithfulness) và độ liên quan (Answer Relevance) bằng mô hình giám khảo.
        </div>
        """, unsafe_allow_html=True)


if menu_option == "💬 Trò chuyện":
    # ==========================================
    # 💬 MAIN PANEL: TRÒ CHUYỆN RAG ĐA PHƯƠNG TIỆN
    # ==========================================
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
            <span style="font-size: 1.4rem; font-weight: 500; color: #ffffff;">Cuộc trò chuyện</span>
            <div style="display: flex; gap: 16px; align-items: center;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="cursor: pointer;">
                    <path d="M3 17V19H9V17H3ZM3 5V7H13V5H3ZM13 21V19H21V17H13V15H11V21H13ZM7 9V11H3V13H7V15H9V9H7ZM21 13V11H11V13H21ZM15 9H17V5H21V3H17V1H15V9Z" fill="#38bdf8"/>
                </svg>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="cursor: pointer;">
                    <path d="M12 8C13.1 8 14 7.1 14 6C14 4.9 13.1 4 12 4C10.9 4 10 4.9 10 6C10 7.1 10.9 8 12 8ZM12 10C10.9 10 10 10.9 10 12C10 13.1 10.9 14 12 14C13.1 14 14 13.1 14 12C14 10.9 13.1 10 12 10ZM12 16C10.9 16 10 16.9 10 18C10 19.1 10.9 20 12 20C13.1 20 14 19.1 14 18C14 16.9 13.1 16 12 16Z" fill="#38bdf8"/>
                </svg>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bộ lọc nguồn dữ liệu tìm kiếm và giao diện Chat được đưa vào st.fragment để tránh tải lại toàn bộ trang (hoặc sidebar) khi thay đổi lựa chọn nguồn
    @st.fragment
    def show_chat_interface():
        source_filter = st.radio(
            "Chọn nguồn:",
            options=["Tất cả", "Chỉ tài liệu PDF", "Chỉ Video YouTube"],
            horizontal=True
        )

        # Chuyển đổi bộ lọc nguồn sang tham số đầu vào đồ thị
        source_type = None
        if source_filter == "Chỉ tài liệu PDF":
            source_type = "pdf"
        elif source_filter == "Chỉ Video YouTube":
            source_type = "youtube"

        # Hiển thị lịch sử trò chuyện từ session state
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
                # Nếu là câu trả lời của trợ lý
                if msg["role"] == "assistant":
                    # Thêm thanh hành động kiểu NotebookLM
                    st.markdown("""
                    <div class="notebook-actions">
                        <span class="action-item"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="vertical-align:-1px; margin-right:4px;"><path d="M12 2L2 22h20L12 2z"/></svg>Lưu vào ghi chú</span>
                        <span class="action-icon">📋</span>
                        <span class="action-icon">👍</span>
                        <span class="action-icon">👎</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hiển thị các nguồn trích lục được
                    if "sources" in msg and msg["sources"]:
                        with st.expander("<> Nguồn tham chiếu"):
                            for i, doc in enumerate(msg["sources"]):
                                meta = doc.metadata
                                st.markdown(
                                    f"""
                                    <div class="citation-block">
                                        <div class="source-meta">
                                            <span class="source-icon">📄</span>
                                            <span class="source-name">{meta.get('source_name')}</span>
                                            <span class="source-type">{str(meta.get('source_type')).upper()}</span>
                                            <span class="source-location">{f"Trang {meta.get('page')}" if meta.get('source_type') == 'pdf' else f"Mốc {meta.get('timestamp')}"}</span>
                                        </div>
                                        <div class="source-content">{doc.page_content}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

        # Nhận tin nhắn mới từ người dùng
        if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
            # 1. Thêm tin nhắn của người dùng vào giao diện và lưu lịch sử
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
                
            # 2. Gọi Agent LangGraph sinh câu trả lời
            with st.chat_message("assistant"):
                with st.spinner("...Đang quét Database và tổng hợp câu trả lời"):
                    try:
                        # Khởi chạy đồ thị Agent
                        inputs = {
                            "query": prompt,
                            "source_type": source_type
                        }
                        result = rag_app.invoke(inputs)
                        
                        # Trích xuất kết quả trả về
                        response_text = result.get("response", "❌ Xin lỗi, hệ thống gặp lỗi khi sinh câu trả lời.")
                        reranked_docs = result.get("reranked_docs", [])
                        
                        # Hiển thị câu trả lời
                        st.write(response_text)
                        
                        # Thêm thanh hành động kiểu NotebookLM
                        st.markdown("""
                        <div class="notebook-actions">
                            <span class="action-item"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="vertical-align:-1px; margin-right:4px;"><path d="M12 2L2 22h20L12 2z"/></svg>Lưu vào ghi chú</span>
                            <span class="action-icon">📋</span>
                            <span class="action-icon">👍</span>
                            <span class="action-icon">👎</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Hiển thị các tài liệu trích lục được
                        if reranked_docs:
                            with st.expander("<> Nguồn tham chiếu"):
                                for i, doc in enumerate(reranked_docs):
                                    meta = doc.metadata
                                    st.markdown(
                                        f"""
                                        <div class="citation-block">
                                            <div class="source-meta">
                                                <span class="source-icon">📄</span>
                                                <span class="source-name">{meta.get('source_name')}</span>
                                                <span class="source-type">{str(meta.get('source_type')).upper()}</span>
                                                <span class="source-location">{f"Trang {meta.get('page')}" if meta.get('source_type') == 'pdf' else f"Mốc {meta.get('timestamp')}"}</span>
                                            </div>
                                            <div class="source-content">{doc.page_content}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        
                        # Lưu vào lịch sử trò chuyện
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": reranked_docs
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi trong quá trình xử lý Agent RAG: {str(e)}")

    show_chat_interface()
else:
    # ==========================================
    # 📊 MAIN PANEL: BENCHMARK & ĐÁNH GIÁ CHẤT LƯỢNG
    # ==========================================
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
            <span style="font-size: 1.4rem; font-weight: 500; color: #ffffff;">📊 Benchmark & Đánh giá Chất lượng RAG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Trang này cho phép đánh giá chi tiết hiệu năng của hệ thống RAG: 
    **Độ trễ (Latency)**, **Tài nguyên tiêu thụ (RAM/GPU, Tokens)** và **Chất lượng phản hồi (Ragas evaluation)**.
    """)
    
    # 1. Pipeline Benchmark
    with st.expander("1. RAG Pipeline Benchmark (BGE Reranker)", expanded=True):
        st.markdown("""
        Đánh giá chất lượng phản hồi và thời gian xử lý của hệ thống RAG trên bộ dữ liệu kiểm thử (5 câu hỏi thực tế).
        """)
        
        # Hiển thị Golden Dataset
        with st.popover("Xem bộ câu hỏi Dataset"):
            st.markdown("**Bộ câu hỏi và Câu trả lời chuẩn (Ground Truth):**")
            for i, item in enumerate(GOLDEN_DATASET):
                st.markdown(f"**Q{i+1}:** {item['question']}")
                st.markdown(f"- *Nguồn dự kiến:* {item['expected_source'] or 'Tất cả (Lạc đề)'}")
                st.markdown(f"- *Ground Truth:* {item['ground_truth']}")
                st.markdown("---")
                
        # Nút bấm chạy benchmark
        if st.button("Chạy RAG Pipeline Benchmark", key="btn_run_rag_bench", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("⏳ 1/2: Đang chạy thử nghiệm...")
            progress_bar.progress(10)
            
            try:
                results_with = run_benchmark_for_config(with_reranker=True)
                progress_bar.progress(60)
                
                # Chạy Ragas evaluation
                status_text.markdown("⏳ 2/2: Đang chạy đánh giá chất lượng tự động bằng **Ragas** (LLM-as-a-judge)...")
                api_key = os.environ.get("GOOGLE_API_KEY")
                ragas_with = run_ragas_evaluation(results_with, api_key)
                
                progress_bar.progress(100)
                status_text.success("✅ Đã hoàn tất đánh giá RAG Pipeline Benchmark!")
                
                # Tính toán số liệu trung bình
                avg_ret  = sum(r["retrieval_time"]  for r in results_with) / len(results_with)
                avg_rank = sum(r["reranking_time"]  for r in results_with) / len(results_with)
                avg_ttft = sum(r["ttft"]             for r in results_with) / len(results_with)
                avg_gen  = sum(r["generation_time"] for r in results_with) / len(results_with)
                avg_total = sum(r["total_time"]     for r in results_with) / len(results_with)
                avg_in_tokens  = sum(r["input_tokens"]  for r in results_with) / len(results_with)
                avg_out_tokens = sum(r["output_tokens"] for r in results_with) / len(results_with)
                peak_ram = max(r["peak_ram"] for r in results_with)
                peak_cpu = max(r["peak_cpu"] for r in results_with)
                
                # Bảng số liệu chi tiết
                st.markdown("### 📊 Chi tiết các chỉ số trung bình")
                summary_data = {
                    "Chỉ số đo lường": [
                        "Retrieval Latency (ChromaDB search)",
                        "Reranking Latency (BGE)",
                        "TTFT (Time to First Token)",
                        "Generation Latency (Gemini)",
                        "Tổng thời gian phản hồi (Total)",
                        "Input Tokens trung bình",
                        "Output Tokens trung bình",
                        "RAM Peak tiêu thụ",
                        "CPU Peak tiêu thụ",
                        "Ragas Faithfulness (Độ trung thực)",
                        "Ragas Answer Relevancy (Độ liên quan)",
                        "Ragas Context Precision (Độ chính xác ngữ cảnh)",
                        "Ragas Context Recall (Độ phủ ngữ cảnh)"
                    ],
                    "Giá trị": [
                        f"{avg_ret:.4f} s",
                        f"{avg_rank:.4f} s",
                        f"{avg_ttft:.4f} s",
                        f"{avg_gen:.4f} s",
                        f"{avg_total:.4f} s",
                        f"{avg_in_tokens:.1f} tokens",
                        f"{avg_out_tokens:.1f} tokens",
                        f"{peak_ram:.1f} MB",
                        f"{peak_cpu:.1f}%",
                        f"{ragas_with['faithfulness']:.4f}",
                        f"{ragas_with['answer_relevance']:.4f}",
                        f"{ragas_with['context_precision']:.4f}",
                        f"{ragas_with['context_recall']:.4f}"
                    ]
                }
                st.table(pd.DataFrame(summary_data))

                # Chi tiết từng câu hỏi
                st.markdown("#### 💬 Phân tích chi tiết từng câu hỏi")
                for i, rw in enumerate(results_with):
                    with st.expander(f"Q{i+1}: {rw['question']}", expanded=False):
                        st.markdown(f"**Trả lời:**\n{rw['answer']}")
                        st.markdown(f"- *Nguồn đã lấy:* {', '.join(rw['retrieved_sources']) if rw['retrieved_sources'] else 'Không tìm thấy'}")
                        st.markdown(f"- *TTFT:* {rw['ttft']:.3f}s | *Tổng thời gian:* {rw['total_time']:.3f}s")
                        st.markdown(f"- *Tokens:* {rw['input_tokens']} in / {rw['output_tokens']} out")
            except Exception as e:
                st.error(f"❌ Có lỗi xảy ra khi chạy benchmark: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


    # 2. Ingestion Resource Benchmark
    with st.expander("2. Ingestion Resource Benchmark (Docling & Faster-Whisper)", expanded=False):
        st.markdown("""
        Đo đạc chính xác thời gian xử lý và **lượng RAM/GPU tiêu hao đỉnh điểm (Peak RAM/GPU)** khi chạy hai phân hệ nặng nhất trong hệ thống.
        """)
        
        col_doc, col_whis = st.columns(2)
        
        with col_doc:
            st.markdown("### 📄 Xử lý PDF")
            # Tìm file PDF mẫu
            pdf_files = [f for f in os.listdir("data/raw") if f.endswith(".pdf")] if os.path.exists("data/raw") else []
            default_pdf = "data/raw/cs229-notes1.pdf"
            
            selected_pdf = st.selectbox("Chọn file PDF mẫu:", options=pdf_files if pdf_files else [default_pdf])
            pdf_full_path = os.path.join("data/raw", selected_pdf) if selected_pdf in pdf_files else selected_pdf
            
            if st.button("Chạy Benchmark", key="btn_docling_bench"):
                if not os.path.exists(pdf_full_path):
                    st.error(f"❌ File '{pdf_full_path}' không tồn tại. Vui lòng tải file lên trước!")
                else:
                    with st.spinner("⏳ Docling đang xử lý phân tích cấu trúc PDF và đo tài nguyên..."):
                        try:
                            bench_res = run_docling_benchmark(pdf_full_path)
                            st.success("✅ Đã hoàn thành benchmark Docling!")
                            st.markdown(f"- **Thời gian xử lý:** {bench_res['duration']:.2f} giây")
                            st.markdown(f"- **RAM tiêu thụ đỉnh điểm:** **{bench_res['peak_ram']:.2f} MB**")
                            if bench_res['peak_gpu'] > 0:
                                st.markdown(f"- **GPU Memory đỉnh điểm:** **{bench_res['peak_gpu']:.2f} MB**")
                            else:
                                st.markdown(f"- **GPU Memory:** Chạy trên CPU (CPU Usage đỉnh điểm: **{bench_res.get('peak_cpu', 0.0):.1f}%**)")
                            st.markdown(f"- **Số trang phân tích:** {bench_res['num_pages']} trang")
                        except Exception as e:
                            st.error(f"❌ Lỗi chạy benchmark Docling: {str(e)}")
                            
        with col_whis:
            st.markdown("### 🎥 Trích xuất YouTube")
            yt_url_test = st.text_input("YouTube URL cho benchmark:", value="https://www.youtube.com/watch?v=iNyUmbmQQZg")
            
            if st.button("Chạy Benchmark", key="btn_whisper_bench"):
                with st.spinner("⏳ Đang xử lý phân tích Video và đo tài nguyên..."):
                    try:
                        bench_res = run_youtube_benchmark(yt_url_test)
                        st.success("✅ Đã hoàn thành benchmark Faster-Whisper!")
                        st.markdown(f"- **Thời gian xử lý:** {bench_res['duration']:.2f} giây")
                        st.markdown(f"- **RAM tiêu thụ đỉnh điểm:** **{bench_res['peak_ram']:.2f} MB**")
                        if bench_res['peak_gpu'] > 0:
                            st.markdown(f"- **GPU Memory đỉnh điểm:** **{bench_res['peak_gpu']:.2f} MB**")
                        else:
                            st.markdown(f"- **GPU Memory:** Chạy trên CPU (CPU Usage đỉnh điểm: **{bench_res.get('peak_cpu', 0.0):.1f}%**)")
                        st.markdown(f"- **Số lượng mảnh:** {bench_res['num_chunks']} mảnh")
                    except Exception as e:
                        st.error(f"❌ Lỗi chạy benchmark Faster-Whisper: {str(e)}")
