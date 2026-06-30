import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import RAGState
from src.retrieval.vector_db import get_vector_db_singleton
from src.retrieval.reranker import BGEReranker

# Khởi tạo các biến dịch vụ toàn cục theo cơ chế Lazy Load (chỉ tải khi chạy node)
vector_db = None
reranker = None
llm = None

def get_services():
    """Tải chậm các dịch vụ để tránh làm chậm hệ thống khi import"""
    global vector_db, reranker, llm
    if vector_db is None:
        vector_db = get_vector_db_singleton()
    if reranker is None:
        reranker = BGEReranker()
    if llm is None:
        # Cấu hình Gemini 3.1 Flash Lite có giới hạn Free Tier tốt nhất (15 RPM / 500 RPD theo bảng giới hạn của bạn)
        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
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
    
    # Tự động nhận diện tài liệu theo tên từ danh sách nguồn đã có trong CSDL
    matched_source_name = None
    best_match_len = 0
    try:
        sources = db.get_all_sources()
        for src in sources:
            name = src["source_name"]
            # Thử cả tên đầy đủ và tên không có phần mở rộng (ví dụ: docling thay vì docling.pdf)
            name_variants = [name]
            name_no_ext, ext = os.path.splitext(name)
            if ext:
                name_variants.append(name_no_ext)
                
            # Đối chiếu thêm video_id nếu có
            vid = src.get("video_id")
            if vid:
                name_variants.append(vid)
                
            for variant in name_variants:
                # Tránh khớp những ký tự đơn lẻ quá ngắn (ví dụ: file tên là "a.pdf" thì không tự khớp chữ "a" trong query)
                if len(variant) >= 3 and variant.lower() in query.lower():
                    if len(variant) > best_match_len:
                        best_match_len = len(variant)
                        matched_source_name = name
    except Exception as e:
        print(f"⚠️ [NODE: RETRIEVE] Lỗi khi nhận diện tên tài liệu tự động: {str(e)}")

    if matched_source_name:
        print(f"🎯 [Auto-Filter] Phát hiện query đề cập đến tài liệu: '{matched_source_name}'. Tự động khóa tìm kiếm vào tài liệu này.")
        
    # TRƯỜNG HỢP 1: Người dùng chủ động chọn lọc 1 nguồn cụ thể (PDF hoặc YT) từ giao diện
    if source_type is not None:
        docs = db.search(query, source_type=source_type, source_name=matched_source_name, k=8)
        print(f"📥 Đã tìm thấy {len(docs)} mảnh thô từ nguồn lọc [{source_type.upper()}] (Tên: {matched_source_name or 'Tất cả'}).")
        return {"retrieved_docs": docs, "matched_source_name": matched_source_name}
        
    # TRƯỜNG HỢP 2: Tìm kiếm kết hợp (Trộn lẫn cả YT và PDF)
    else:
        # Nếu đã tự động lọc trúng một tài liệu cụ thể
        if matched_source_name:
            # Xác định loại của tài liệu khớp
            matched_type = None
            for src in sources:
                if src["source_name"] == matched_source_name:
                    matched_type = src["source_type"]
                    break
            docs = db.search(query, source_type=matched_type, source_name=matched_source_name, k=8)
            print(f"📥 Đã tìm thấy {len(docs)} mảnh thô từ tài liệu tự lọc '{matched_source_name}'.")
            return {"retrieved_docs": docs, "matched_source_name": matched_source_name}
            
        print("🔄Luồng trộn lẫn: Đang lấy độc lập dữ liệu từ cả PDF và YouTube...")
        # Lấy 5 mảnh tốt nhất từ PDF
        pdf_docs = db.search(query, source_type="pdf", k=5)
        # Lấy 5 mảnh tốt nhất từ YouTube
        yt_docs = db.search(query, source_type="youtube", k=5)
        
        # Hợp nhất 2 danh sách lại thành 10 mảnh thô để đưa cho Reranker tối ưu
        all_docs = pdf_docs + yt_docs
        print(f"📥 Thu hoạch tổng cộng {len(all_docs)} mảnh thô (PDF: {len(pdf_docs)}, YT: {len(yt_docs)}).")
        
        return {"retrieved_docs": all_docs, "matched_source_name": None}

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
    
    # 3. Kiểm tra ngưỡng điểm chặn nhiễu (Bỏ qua hoặc hạ thấp nếu người dùng chủ động chọn/khớp đúng tên tài liệu)
    matched_source_name = state.get("matched_source_name")
    source_type = state.get("source_type")
    
    # Nếu người dùng đã chỉ định hoặc khớp đúng tài liệu cụ thể, ta không lọc bỏ "lạc đề" vì họ đang cố tình hỏi tài liệu này
    is_explicit_query = (matched_source_name is not None) or (source_type is not None)
    
    SCORE_THRESHOLD = 0.35
    if is_explicit_query:
        # Cho phép vượt ngưỡng lọc nhiễu vì đây là tài liệu người dùng yêu cầu đích danh
        final_docs = sorted_docs[:3]
        print(f"🎯 [Explicit Query] Bỏ qua bộ lọc lạc đề (Điểm tốt nhất: {best_score:.4f}). Chấp nhận cấp {len(final_docs)} mảnh từ tài liệu yêu cầu cho LLM.")
    elif best_score < SCORE_THRESHOLD:
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
    Sử dụng Gemini 3.1 Flash Lite để tổng hợp câu trả lời tiếng Việt có dẫn số trang/timestamp
    """
    print("\n--- [NODE: GENERATE] Đang sinh câu trả lời bằng Gemini 3.1 Flash Lite... ---")
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
    
    # Đảm bảo chỉ lấy chuỗi text thuần túy
    response_text = ""
    if isinstance(response.content, list):
        # Nếu là dạng list [ {"type": "text", "text": "..."} ]
        for part in response.content:
            if isinstance(part, dict) and "text" in part:
                response_text += part["text"]
            elif isinstance(part, str):
                response_text += part
    else:
        response_text = str(response.content)
        
    return {"response": response_text}
