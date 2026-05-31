from sentence_transformers import CrossEncoder

class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        """
        Khởi tạo mô hình BGE Reranker dùng để tái xếp hạng độ chính xác tài liệu.
        Sử dụng BAAI/bge-reranker-base (hoặc tương đương) qua sentence-transformers.
        """
        print(f"🔄 Đang tải mô hình BGE Reranker ({model_name})...")
        self.model = CrossEncoder(model_name)
        print("✅ Đã tải mô hình Reranker thành công!")

    def rerank(self, query, documents, top_k=3):
        """
        Tái xếp hạng danh sách tài liệu trả về dựa trên độ liên quan với truy vấn
        :param query: Câu hỏi truy vấn
        :param documents: Danh sách tài liệu thô được retrieve từ Vector DB
        :param top_k: Số lượng tài liệu tối ưu muốn giữ lại
        :return: Danh sách tài liệu đã được sắp xếp lại và lọc
        """
        if not documents:
            return []

        # Chuẩn bị cặp dữ liệu (câu hỏi, nội dung tài liệu) để đưa vào mô hình CrossEncoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Dự đoán điểm số liên quan
        scores = self.model.predict(pairs)
        
        # Kết hợp tài liệu và điểm số tương ứng
        scored_docs = list(zip(documents, scores))
        
        # Sắp xếp giảm dần theo điểm số (điểm càng cao càng liên quan)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # In ra màn hình điểm số rerank để phục vụ gỡ lỗi và theo dõi
        print("\n--- 📊 Kết quả Tái xếp hạng (BGE Reranker Scores) ---")
        for i, (doc, score) in enumerate(scored_docs):
            src_name = doc.metadata.get("source_name", "Không rõ")
            print(f"  [{i+1}] {src_name} | Điểm Rerank: {score:.4f} | Nội dung: {doc.page_content[:60]}...")
        
        # Trả về top_k tài liệu tốt nhất
        return [doc for doc, score in scored_docs[:top_k]]
