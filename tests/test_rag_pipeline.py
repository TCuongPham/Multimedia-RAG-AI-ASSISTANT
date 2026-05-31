import sys
import os
# Cấu hình stdout hỗ trợ ký tự UTF-8 (tiếng Việt & Emoji) trên Windows Terminal
if sys.platform.startswith("win"):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph.workflow import build_rag_graph
from src.retrieval.vector_db import VectorDBManager
from dotenv import load_dotenv

# Nạp biến môi trường (Gemini API Key)
load_dotenv()

def setup_test_data():
    """Chuẩn bị dữ liệu mẫu trong DB để chạy thử nghiệm RAG"""
    print("\n--- Đang thiết lập dữ liệu Vector DB thử nghiệm... ---")
    db = VectorDBManager(db_path="data/vectorstore")
    
    # Reset dữ liệu cũ để đảm bảo tính chính xác của các lần chạy test
    db.clear_db()
    
    # 1. Nạp tài liệu RAG từ PDF thô giả lập
    pdf_text = (
        "Kiến trúc RAG (Retrieval-Augmented Generation) là một kỹ thuật đột phá trong AI. "
        "Nó kết hợp sức mạnh của Retriever để truy xuất thông tin từ kho cơ sở dữ liệu vector "
        "và Generator (LLM) để sinh câu trả lời dựa trên ngữ cảnh đó. "
        "Retriever đóng vai trò vô cùng quan trọng: nó giúp LLM giảm thiểu hiện tượng ảo tưởng (hallucination) "
        "bằng cách chỉ cung cấp thông tin thực tế được trích dẫn chính xác từ tài liệu nguồn."
    )
    db.add_documents(
        text=pdf_text,
        source_type="pdf",
        source_name="sach_giao_trinh_rag.pdf",
        extra_metadata={"page": 10}
    )
    
    # 2. Nạp dữ liệu transcript từ YouTube thô giả lập
    yt_text = (
        "Xin chào tất cả mọi người! Trong bài giảng này, mình sẽ hướng dẫn các bạn cách sử dụng "
        "Google Gemini API để tạo vector nhúng (embedding) và lưu trữ trực tiếp vào cơ sở dữ liệu vector ChromaDB. "
        "Các bạn chỉ cần cài đặt thư viện langchain-google-genai, sau đó truyền API Key vào biến môi trường. "
        "ChromaDB sẽ giúp chúng ta truy xuất dữ liệu cực kỳ nhanh chóng phục vụ cho các ứng dụng chatbot thông minh."
    )
    db.add_documents(
        text=yt_text,
        source_type="youtube",
        source_name="Video Học Gemini API từ Cơ Bản",
        extra_metadata={"video_id": "gemini_vid_99", "timestamp": "[08:45]"}
    )
    print("✅ Đã chuẩn bị dữ liệu Vector DB thành công!")

def test_rag_pipeline():
    print("==================================================")
    print("=== BẮT ĐẦU KIỂM THỬ TOÀN DIỆN RAG PIPELINE ===")
    print("==================================================")
    
    # 1. Chuẩn bị dữ liệu Vector DB
    setup_test_data()
    
    # 2. Biên dịch luồng đồ thị LangGraph
    print("\n🕸️  Đang biên dịch đồ thị LangGraph...")
    rag_app = build_rag_graph()
    print("✅ Đã biên dịch đồ thị LangGraph thành công!")
    
    # 3. KỊCH BẢN TÌM KIẾM 1: Hỏi về PDF (Hệ thống tự lọc và trả lời có trích nguồn số trang)
    print("\n==============================================")
    print("🔎 KỊCH BẢN 1: Hỏi về Kiến trúc RAG (PDF)")
    print("==============================================")
    query_1 = "Bộ phận Retriever trong kiến trúc RAG đóng vai trò quan trọng như thế nào?"
    
    inputs_1 = {
        "query": query_1,
        "source_type": "pdf"  # Chủ động lọc nguồn PDF
    }
    
    # Khởi chạy đồ thị
    result_1 = rag_app.invoke(inputs_1)
    
    print("\n✨ CÂU TRẢ LỜI CỦA TRỢ LÝ RAG:")
    print("----------------------------------------------")
    print(result_1["response"])
    print("----------------------------------------------")
    
    # 4. KỊCH BẢN TÌM KIẾM 2: Hỏi về YouTube (Hệ thống tự lọc và trả lời có trích nguồn mốc thời gian)
    print("\n==============================================")
    print("🔎 KỊCH BẢN 2: Hỏi về Gemini API & ChromaDB (YouTube)")
    print("==============================================")
    query_2 = "Các bước để kết nối ChromaDB với Google Gemini API là gì?"
    
    inputs_2 = {
        "query": query_2,
        "source_type": "youtube"  # Chủ động lọc nguồn YouTube
    }
    
    # Khởi chạy đồ thị
    result_2 = rag_app.invoke(inputs_2)
    
    print("\n✨ CÂU TRẢ LỜI CỦA TRỢ LÝ RAG:")
    print("----------------------------------------------")
    print(result_2["response"])
    print("----------------------------------------------")
    
    # 5. KỊCH BẢN TÌM KIẾM 3: Hỏi câu hỏi lạc đề (Hệ thống kích hoạt chế độ chặn nhiễu và từ chối an toàn)
    print("\n==============================================")
    print("🔎 KỊCH BẢN 3: Kiểm thử khả năng chặn câu hỏi lạc đề (Nhiễu)")
    print("==============================================")
    # Câu hỏi hoàn toàn không liên quan đến RAG hay Gemini API đã nạp trong DB
    query_3 = "Hướng dẫn tôi cách nấu món phở bò Hà Nội chuẩn vị?"
    
    inputs_3 = {
        "query": query_3,
        "source_type": None  # Để None để quét toàn bộ kho xem có bị lọt nhiễu không
    }
    
    # Khởi chạy đồ thị
    result_3 = rag_app.invoke(inputs_3)
    
    print("\n✨ CÂU TRẢ LỜI CỦA TRỢ LÝ RAG:")
    print("----------------------------------------------")
    print(result_3["response"])
    print("----------------------------------------------")
    
    # Dọn dẹp cơ sở dữ liệu test
    print("\n🧹 Đang dọn dẹp dữ liệu test...")
    db = VectorDBManager(db_path="data/vectorstore")
    db.clear_db()
    print("=== HOÀN TẤT KIỂM THỬ TOÀN DIỆN RAG PIPELINE ===")

if __name__ == "__main__":
    test_rag_pipeline()
