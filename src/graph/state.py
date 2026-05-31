from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document

class RAGState(TypedDict):
    """
    Đại diện cho trạng thái (State) được truyền qua lại giữa các Node trong đồ thị LangGraph.
    """
    query: str                       # Câu hỏi của người dùng
    source_type: Optional[str]        # Bộ lọc nguồn dữ liệu ('pdf', 'youtube', hoặc None)
    retrieved_docs: List[Document]   # Danh sách các tài liệu thô được quét từ Vector DB (ChromaDB)
    reranked_docs: List[Document]    # Danh sách các tài liệu tinh tuyển sau khi qua BGE Reranker
    response: str                     # Câu trả lời cuối cùng từ LLM Gemini kèm trích dẫn
