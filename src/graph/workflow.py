from langgraph.graph import StateGraph, END
from src.graph.state import RAGState
from src.graph.nodes import retrieve_node, rerank_node, generate_node

def build_rag_graph():
    """
    Xây dựng và biên dịch luồng điều phối RAG Agent sử dụng LangGraph
    Luồng chạy: retrieve -> rerank -> generate -> Kết thúc
    """
    # Khởi tạo đồ thị trạng thái (StateGraph) dựa trên cấu trúc RAGState
    workflow = StateGraph(RAGState)
    
    # 1. Thêm các Node xử lý logic vào Đồ thị
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    
    # 2. Thiết lập điểm bắt đầu của luồng điều phối
    workflow.set_entry_point("retrieve")
    
    # 3. Định nghĩa các cạnh nối tiếp tuần tự (Edges) giữa các Node
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    
    # 4. Biên dịch đồ thị thành ứng dụng có thể khởi chạy
    app = workflow.compile()
    
    return app
