import sys
import os
# Thêm thư mục gốc vào đường dẫn hệ thống để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.vector_db import VectorDBManager
from dotenv import load_dotenv

# Nạp biến môi trường (API Key của Google)
load_dotenv()

def test_vector_db():
    print("=== BẮT ĐẦU TEST VECTOR DATABASE ===")
    
    # 1. Khởi tạo DB Manager (dùng thư mục test riêng biệt để tránh ảnh hưởng dữ liệu thật)
    db = VectorDBManager(db_path="data/test_vectorstore")
    
    # 2. Dọn sạch DB cũ nếu có để test sạch sẽ
    db.clear_db()
    
    # 3. Nạp dữ liệu mẫu dạng PDF
    print("\n--- 📄 Đang nạp dữ liệu PDF mẫu... ---")
    pdf_text = (
        "Kiến trúc RAG (Retrieval-Augmented Generation) là một kỹ thuật tiên tiến trong AI. "
        "Nó bao gồm hai thành phần chính: Retriever (truy xuất thông tin) và Generator (mô hình ngôn ngữ lớn để tạo câu trả lời). "
        "Retriever sẽ tìm các đoạn văn bản liên quan nhất từ Vector Database để cung cấp ngữ cảnh cho Generator."
    )
    db.add_documents(
        text=pdf_text,
        source_type="pdf",
        source_name="bao_cao_rag.pdf",
        extra_metadata={"page": 5}
    )
    
    # 4. Nạp dữ liệu mẫu dạng YouTube
    print("\n--- 🎥 Đang nạp dữ liệu YouTube mẫu... ---")
    yt_text = (
        "Chào mừng các bạn đã quay trở lại kênh AI Tutorial. Hôm nay chúng ta sẽ cùng học cách "
        "kết nối cơ sở dữ liệu vector ChromaDB và sử dụng Google Gemini API để nhúng (embed) văn bản "
        "cực kỳ đơn giản bằng ngôn ngữ lập trình Python."
    )
    db.add_documents(
        text=yt_text,
        source_type="youtube",
        source_name="Hướng dẫn sử dụng Gemini API",
        extra_metadata={"video_id": "xyz12345", "timestamp": "[05:12]"}
    )
    
    # 5. TEST CHỨC NĂNG TÌM KIẾM (SEARCH)
    print("\n=== 🔍 TIẾN HÀNH TEST TÌM KIẾM ===")
    
    # Test 5.1: Tìm kiếm CHUNG (Không lọc)
    query_1 = "RAG là gì?"
    print(f"\n🔎 Câu hỏi: '{query_1}' (Tìm kiếm chung cả PDF & YouTube):")
    results = db.search(query_1, k=2)
    for i, doc in enumerate(results):
        print(f"  Mảnh {i+1} [{doc.metadata.get('source_type').upper()}]: {doc.metadata.get('source_name')}")
        print(f"  Nội dung: {doc.page_content[:150]}...")
        
    # Test 5.2: Tìm kiếm có LỌC CHỈ LẤY PDF
    query_2 = "Cách kết nối ChromaDB và Gemini"
    print(f"\n🔎 Câu hỏi: '{query_2}' (Chỉ lọc tìm trong tài liệu PDF):")
    # LƯU Ý: Câu hỏi này về "Gemini" (thuộc video YouTube), nhưng ta cố tình lọc chỉ tìm trong PDF
    results_pdf = db.search(query_2, source_type="pdf", k=1)
    if results_pdf:
        print(f"  Đã tìm thấy [{results_pdf[0].metadata.get('source_type').upper()}]: {results_pdf[0].metadata.get('source_name')}")
        print(f"  Nội dung: {results_pdf[0].page_content[:150]}...")
    else:
        print("  ❌ Không tìm thấy kết quả phù hợp trong tài liệu PDF!")
        
    # Test 5.3: Tìm kiếm có LỌC CHỈ LẤY YOUTUBE
    print(f"\n🔎 Câu hỏi: '{query_2}' (Chỉ lọc tìm trong Video YouTube):")
    results_yt = db.search(query_2, source_type="youtube", k=1)
    if results_yt:
        print(f"  Đã tìm thấy [{results_yt[0].metadata.get('source_type').upper()}]: {results_yt[0].metadata.get('source_name')} - Phút: {results_yt[0].metadata.get('timestamp')}")
        print(f"  Nội dung: {results_yt[0].page_content[:150]}...")
    else:
        print("  ❌ Không tìm thấy kết quả phù hợp trong video YouTube!")

    # 6. Dọn dẹp sau khi test (Tùy chọn)
    print("\n--- 🧹 Đang dọn dẹp dữ liệu test... ---")
    db.clear_db()
    print("=== HOÀN TẤT TEST VECTOR DATABASE ===")

if __name__ == "__main__":
    test_vector_db()
