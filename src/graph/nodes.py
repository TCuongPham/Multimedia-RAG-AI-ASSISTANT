import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import RAGState
from src.retrieval.vector_db import VectorDBManager
from src.retrieval.reranker import BGEReranker

# Khởi tạo các biến dịch vụ toàn cục theo cơ chế Lazy Load (chỉ tải khi chạy node)
vector_db = None
reranker = None
llm = None

def get_services():
    """Tải chậm các dịch vụ để tránh làm chậm hệ thống khi import"""
    global vector_db, reranker, llm
    if vector_db is None:
        vector_db = VectorDBManager()
    if reranker is None:
        reranker = BGEReranker()
    if llm is None:
        # Cấu hình Gemini 2.5 Flash thế hệ mới nhất dùng cho QA theo đề xuất kế hoạch nâng cấp
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    return vector_db, reranker, llm

def retrieve_node(state: RAGState) -> RAGState:
    """
    Node 1: Truy xuất tài liệu từ ChromaDB
    Lấy ra top 8 tài liệu thô để chuẩn bị cho bước xếp hạng sau
    """
    print("\n--- [NODE: RETRIEVE] Đang truy xuất tài liệu từ VectorDB... ---")
    db, _, _ = get_services()
    
    query = state["query"]
    source_type = state.get("source_type")
    
    # TRƯỜNG HỢP 1: Người dùng chủ động chọn lọc 1 nguồn cụ thể (PDF hoặc YT)
    if source_type is not None:
        docs = db.search(query, source_type=source_type, k=8)
        print(f"📥 Đã tìm thấy {len(docs)} mảnh thô từ nguồn lọc [{source_type.upper()}].")
        return {"retrieved_docs": docs}
        
    # TRƯỜNG HỢP 2: Tìm kiếm kết hợp (Trộn lẫn cả YT và PDF)
    else:
        print("🔄Luồng trộn lẫn: Đang lấy độc lập dữ liệu từ cả PDF và YouTube...")
        # Lấy 5 mảnh tốt nhất từ PDF
        pdf_docs = db.search(query, source_type="pdf", k=5)
        # Lấy 5 mảnh tốt nhất từ YouTube
        yt_docs = db.search(query, source_type="youtube", k=5)
        
        # Hợp nhất 2 danh sách lại thành 10 mảnh thô để đưa cho Reranker tối ưu
        all_docs = pdf_docs + yt_docs
        print(f"📥 Thu hoạch tổng cộng {len(all_docs)} mảnh thô (PDF: {len(pdf_docs)}, YT: {len(yt_docs)}).")
        
        return {"retrieved_docs": all_docs}

def rerank_node(state: RAGState) -> RAGState:
    """
    Node 2: Tái xếp hạng (Rerank) tài liệu bằng BGE-Reranker
    Sắp xếp lại và chỉ lọc lấy 3 mảnh có độ liên quan cao nhất
    """
    print("\n--- [NODE: RERANK] Đang tối ưu thứ tự tài liệu bằng BGE-Reranker... ---")
    _, ranker, _ = get_services()
    
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        print("⚠️ Không có tài liệu nào để xếp hạng.")
        return {"reranked_docs": []}
        
    sorted_docs = ranker.rerank(query, docs, top_k=8) 
    
    # 2. Tính toán điểm số của mảnh xuất sắc nhất (vị trí đầu tiên) để check lạc đề
    # Ta truyền cặp [query, nội dung mảnh tốt nhất] vào mô hình CrossEncoder ngầm để lấy điểm số thật
    best_doc_content = sorted_docs[0].page_content
    best_score = float(ranker.model.predict([query, best_doc_content]))
    
    # 3. Kiểm tra ngưỡng điểm chặn nhiễu
    SCORE_THRESHOLD = 0.35
    if best_score < SCORE_THRESHOLD:
        print(f"Điểm của mảnh tốt nhất ({best_score:.4f}) < {SCORE_THRESHOLD} -> Câu hỏi LẠC ĐỀ.")
        final_docs = [] # Trả về rỗng để kích hoạt chế độ từ chối ở Node Generate
    else:
        # Nếu vượt ngưỡng, giữ lại tối đa top 3 mảnh tốt nhất để đưa vào ngữ cảnh cho Gemini
        final_docs = sorted_docs[:3]
        print(f"Điểm mảnh tốt nhất đạt {best_score:.4f} (>= {SCORE_THRESHOLD}) -> Chấp nhận cấp {len(final_docs)} mảnh cho LLM.")
        
    return {"reranked_docs": final_docs}

def generate_node(state: RAGState) -> RAGState:
    """
    Node 3: Sinh câu trả lời (LLM) kèm cơ chế trích dẫn nguồn chuẩn xác
    Sử dụng Gemini 2.5 Flash để tổng hợp câu trả lời tiếng Việt có dẫn số trang/timestamp
    """
    print("\n--- [NODE: GENERATE] Đang sinh câu trả lời bằng Gemini 2.5 Flash... ---")
    _, _, model = get_services()
    
    query = state["query"]
    docs = state.get("reranked_docs", [])
    
    if not docs:
        return {"response": "❌ Rất tiếc, tôi không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu để trả lời câu hỏi này."}
        
    # Xây dựng ngữ cảnh kèm nhãn trích nguồn rõ ràng
    context_parts = []
    for i, doc in enumerate(docs):
        src_type = doc.metadata.get("source_type", "unknown")
        src_name = doc.metadata.get("source_name", "Không rõ nguồn")
        
        # Tạo định dạng trích dẫn theo đúng yêu cầu đề tài Project_II
        if src_type == "pdf":
            page = doc.metadata.get("page", "chưa rõ")
            citation = f"[Nguồn: {src_name}, Trang: {page}]"
        elif src_type == "youtube":
            timestamp = doc.metadata.get("timestamp", "chưa rõ")
            citation = f"[Nguồn: {src_name}, Mốc thời gian: {timestamp}]"
        else:
            citation = f"[Nguồn: {src_name}]"
            
        context_parts.append(f"TÀI LIỆU KHẢO SÁT {i+1} {citation}:\n{doc.page_content}")
        
    context_str = "\n\n".join(context_parts)
    
    # Prompt chuẩn hóa kỹ năng RAG học thuật và trích dẫn chuẩn
    prompt = f"""
    Bạn là "Trợ lý Ảo đa phương tiện thông minh". Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa TRÊN các tài liệu khảo sát được cung cấp dưới đây.
    
    YÊU CẦU BẮT BUỘC VỀ TRÍCH DẪN (CITATION):
    1. Câu trả lời của bạn phải hoàn toàn bằng Tiếng Việt và hành văn chuyên nghiệp.
    2. Ở mỗi khẳng định, luận điểm hoặc thông tin cụ thể bạn lấy từ tài liệu, bạn PHẢI dán nhãn trích dẫn chính xác ở ngay cuối câu hoặc cuối ý đó.
       Ví dụ: 
       - "...RAG gồm hai thành phần chính là Retriever và Generator [Nguồn: bao_cao.pdf, Trang: 5]."
       - "...Chúng ta cài đặt ChromaDB bằng lệnh pip install [Nguồn: Hướng dẫn cài đặt, Mốc thời gian: [05:12]]."
    3. Chỉ sử dụng thông tin trong tài liệu cung cấp. Tuyệt đối không tự bịa ra thông tin, số trang hoặc thời gian không có trong tài liệu.
    
    TÀI LIỆU KHẢO SÁT:
    {context_str}
    
    CÂU HỎI CỦA NGƯỜI DÙNG:
    {query}
    
    CÂU TRẢ LỜI CỦA BẠN (HÃY TRÍCH DẪN NGUỒN ĐẦY ĐỦ):
    """
    
    response = model.invoke(prompt)
    
    return {"response": response.content}
