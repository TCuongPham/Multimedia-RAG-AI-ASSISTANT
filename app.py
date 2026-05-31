import os
# Thiết lập cấu hình protobuf để sửa lỗi tương thích phiên bản (TypeError: Descriptors cannot be created directly)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tempfile
from dotenv import load_dotenv

# Nạp các thành phần từ hệ thống RAG
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.yt_loader import YouTubeLoader
from src.retrieval.vector_db import VectorDBManager
from src.graph.workflow import build_rag_graph

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
    return VectorDBManager()

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
    /* Nền ứng dụng tối sang trọng, không dùng gradient */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Tùy chỉnh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Đầu mục tiêu đề chính */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Khối trích nguồn tham chiếu */
    .citation-block {
        background-color: #1e293b;
        border-left: 3px solid #6366f1;
        padding: 12px 16px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.95em;
        line-height: 1.5;
        border: 1px solid #334155;
        border-left-width: 4px;
    }
    
    /* Định dạng metadata của nguồn trích lục */
    .source-meta {
        font-weight: 600;
        color: #818cf8;
        margin-bottom: 6px;
        display: flex;
        gap: 12px;
    }
    
    /* Thiết lập lại kiểu hiển thị chat cho đồng bộ */
    .stChatMessage {
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        background-color: #0f172a;
        border: 1px solid #1e293b;
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
st.sidebar.markdown("# ⚙️ Trung tâm Quản trị")
st.sidebar.markdown("Cung cấp các công cụ nạp và đồng bộ hóa tài liệu đa phương tiện.")

# 1. PHÂN HỆ NẠP TÀI LIỆU PDF
with st.sidebar.expander("📄 Nạp tài liệu PDF mới", expanded=True):
    pdf_file = st.file_uploader("Kéo & thả tệp PDF vào đây", type="pdf")
    
    if pdf_file:
        source_name = pdf_file.name
        source_type = "pdf"
        
        # Kiểm tra trùng lặp bằng metadata
        exists = vector_db.check_source_exists(source_name, source_type)
        
        overwrite = False
        if exists:
            st.warning("⚠️ Tài liệu này đã tồn tại trong CSDL.")
            overwrite = st.checkbox("Ghi đè (Xóa dữ liệu cũ & nạp lại)", value=False, key="pdf_overwrite")
            
        # Nếu chưa tồn tại HOẶC người dùng đồng ý ghi đè
        if not exists or overwrite:
            if st.button("Tiến hành nạp PDF", use_container_width=True):
                temp_path = save_uploaded_file(pdf_file)
                if temp_path:
                    with st.spinner("⏳ Đang phân tích PDF và vector hóa từng trang..."):
                        try:
                            # 1. Phân tích PDF theo trang thông qua Docling
                            pages = pdf_loader.load_and_convert(temp_path)
                            
                            # 2. Lưu các trang kèm metadata số trang vào ChromaDB
                            vector_db.add_documents(
                                text=pages,
                                source_type=source_type,
                                source_name=source_name
                            )
                            st.success("✅ Nạp tài liệu PDF thành công!")
                        except Exception as e:
                            st.error(f"❌ Thất bại khi nạp PDF: {str(e)}")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

# 2. PHÂN HỆ NẠP VIDEO YOUTUBE
with st.sidebar.expander("🎥 Nạp Video YouTube mới", expanded=False):
    yt_url = st.text_input("Nhập đường dẫn Video YouTube", placeholder="https://www.youtube.com/watch?v=...")
    
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
        yt_title = st.text_input("Tiêu đề gợi nhớ cho Video", value=default_title)
        
        # Kiểm tra trùng lặp
        exists = vector_db.check_source_exists(yt_title, "youtube")
        
        overwrite = False
        if exists:
            st.warning("⚠️ Video này đã tồn tại trong CSDL.")
            overwrite = st.checkbox("Ghi đè (Xóa dữ liệu cũ & nạp lại)", value=False, key="yt_overwrite")
            
        # Nếu chưa tồn tại HOẶC đồng ý ghi đè
        if not exists or overwrite:
            if st.button("Tiến hành cào & nạp Video", use_container_width=True):
                with st.spinner("⏳ Đang tải phụ đề / chạy Whisper chuyển âm thanh thành văn bản..."):
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
                            st.success("✅ Nạp Video YouTube thành công!")
                        else:
                            st.error("❌ Không lấy được dữ liệu phụ đề từ Video này.")
                    except Exception as e:
                        st.error(f"❌ Lỗi nạp dữ liệu YouTube: {str(e)}")

# 3. DỌN DẸP HỆ THỐNG
with st.sidebar.expander("🚨 Cấu hình Hệ thống", expanded=False):
    st.markdown("Xóa toàn bộ các vector nhúng hiện tại trong cơ sở dữ liệu ChromaDB.")
    if st.button("Xóa sạch Vector Database", type="secondary", use_container_width=True):
        with st.spinner("Đang xóa sạch dữ liệu..."):
            try:
                vector_db.clear_db()
                st.session_state.messages = []
                st.success("🔥 Đã dọn dẹp sạch sẽ cơ sở dữ liệu!")
            except Exception as e:
                st.error(f"Lỗi dọn dẹp database: {str(e)}")


# ==========================================
# 💬 MAIN PANEL: TRÒ CHUYỆN RAG ĐA PHƯƠNG TIỆN
# ==========================================
st.markdown("# 💬 Trợ lý Ảo đa phương tiện RAG")
st.markdown("Hệ thống trả lời câu hỏi học thuật dựa trên tài liệu PDF và phụ đề Video YouTube.")

# Bộ lọc nguồn dữ liệu tìm kiếm
source_filter = st.radio(
    "Phạm vi tìm kiếm của Agent:",
    options=["Tất cả nguồn dữ liệu", "Chỉ tài liệu PDF", "Chỉ Video YouTube"],
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
        
        # Nếu là câu trả lời của trợ lý và có các nguồn trích dẫn
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("🔍 Chi tiết nguồn trích lục (Reranked Sources)"):
                for i, doc in enumerate(msg["sources"]):
                    meta = doc.metadata
                    st.markdown(
                        f"""
                        <div class="citation-block">
                            <div class="source-meta">
                                <span>📌 Tài liệu {i+1}: {meta.get('source_name')}</span>
                                <span>🏷️ Loại: {str(meta.get('source_type')).upper()}</span>
                                <span>📄 Vị trí: {f"Trang {meta.get('page')}" if meta.get('source_type') == 'pdf' else f"Mốc {meta.get('timestamp')}"}</span>
                            </div>
                            <div>{doc.page_content}</div>
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
        with st.spinner("🤖 Đang quét VectorDB & tổng hợp câu trả lời từ LLM Gemini..."):
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
                
                # Hiển thị các tài liệu trích lục được
                if reranked_docs:
                    with st.expander("🔍 Chi tiết nguồn trích lục (Reranked Sources)"):
                        for i, doc in enumerate(reranked_docs):
                            meta = doc.metadata
                            st.markdown(
                                f"""
                                <div class="citation-block">
                                    <div class="source-meta">
                                        <span>📌 Tài liệu {i+1}: {meta.get('source_name')}</span>
                                        <span>🏷️ Loại: {str(meta.get('source_type')).upper()}</span>
                                        <span>📄 Vị trí: {f"Trang {meta.get('page')}" if meta.get('source_type') == 'pdf' else f"Mốc {meta.get('timestamp')}"}</span>
                                    </div>
                                    <div>{doc.page_content}</div>
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
