import os
import time
import threading
import psutil
import torch
import pandas as pd
from typing import List, Dict, Any, Optional

from google import genai
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from src.retrieval.vector_db import get_vector_db_singleton
from src.graph.nodes import retrieve_node, rerank_node, get_services

# Cấu hình Golden Dataset cho kiểm thử RAG
GOLDEN_DATASET = [
    {
        "question": "Hàm chi phí (Cost function) đóng vai trò quan trọng như thế nào trong hồi quy tuyến tính?",
        "source_type": "pdf",
        "ground_truth": "Hàm chi phí (thường ký hiệu là J) đóng vai trò đo lường mức độ sai lệch giữa các giá trị dự đoán từ giả thuyết và giá trị mục tiêu thực tế. Việc giảm thiểu hàm chi phí này cho phép thuật toán tìm ra bộ tham số tối ưu để mô hình có khả năng dự đoán chính xác nhất.",
        "expected_source": "cs229-notes1.pdf"
    },
    {
        "question": "Bài toán học có giám sát (Supervised learning) là gì?",
        "source_type": "pdf",
        "ground_truth": "Học có giám sát là quá trình học một hàm số h (được gọi là giả thuyết) dựa trên một tập dữ liệu huấn luyện đã biết trước các cặp biến đầu vào (x) và mục tiêu đầu ra tương ứng (y). Mục tiêu của quá trình này là tạo ra một hàm có khả năng dự đoán tốt giá trị của y cho các đầu vào mới.",
        "expected_source": "cs229-notes1.pdf"
    },
    {
        "question": "Các bước thực hiện thuật toán Xuống dốc đạo hàm theo lô (Batch Gradient Descent) là gì?",
        "source_type": "pdf",
        "ground_truth": "Thuật toán bắt đầu bằng cách đưa ra một giá trị khởi tạo cho tham số, sau đó lặp lại việc cập nhật tham số đó theo hướng giảm mạnh nhất của hàm chi phí (ngược hướng đạo hàm) cho đến khi hội tụ. Trong mỗi bước cập nhật, thuật toán này phải tính toán trên toàn bộ tập dữ liệu huấn luyện.",
        "expected_source": "cs229-notes1.pdf"
    },
    {
        "question": "Cần giả định điều gì về sai số để chứng minh hồi quy bình phương tối thiểu là một phương pháp hợp lý?",
        "source_type": "pdf",
        "ground_truth": "Cần giả định các thuật ngữ sai số được phân phối độc lập và đồng nhất (IID) theo phân phối Gaussian (phân phối chuẩn) với giá trị trung bình bằng không và một phương sai nhất định. Dưới các giả định này, việc giảm thiểu hàm chi phí bình phương tối thiểu tương đương với việc tìm ước lượng hợp lý cực đại của tham số.",
        "expected_source": "cs229-notes1.pdf"
    },
    {
        "question": "Hướng dẫn tôi cách nấu món phở bò Hà Nội chuẩn vị?",
        "source_type": None,
        "ground_truth": "❌ Rất tiếc, tôi không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu (tệp cs229-notes1.pdf) để trả lời câu hỏi này.",
        "expected_source": None
    }
]

# -------------------------------------------------------------------------
# LỚP ĐO LƯỜNG TÀI NGUYÊN (RAM & GPU TRACKER)
# -------------------------------------------------------------------------
class RAMTracker:
    """Theo dõi Peak RAM tiêu thụ (MB) và Peak CPU (%) bằng thread chạy nền"""
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_ram = 0.0
        self.peak_cpu = 0.0
        self.running = False
        self.thread = None

    def _track(self):
        process = psutil.Process(os.getpid())
        # Khởi tạo CPU percent
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        while self.running:
            try:
                # Đơn vị: Megabytes (MB)
                ram_mb = process.memory_info().rss / (1024 * 1024)
                if ram_mb > self.peak_ram:
                    self.peak_ram = ram_mb
                
                # Đơn vị: % sử dụng CPU hệ thống
                cpu_pct = psutil.cpu_percent(interval=None)
                if cpu_pct > self.peak_cpu:
                    self.peak_cpu = cpu_pct
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_ram = 0.0
        self.peak_cpu = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self) -> tuple[float, float]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.peak_ram, self.peak_cpu

class GPUTracker:
    """Theo dõi Peak GPU tiêu thụ bằng torch.cuda"""
    def __init__(self):
        self.available = torch.cuda.is_available()

    def start(self):
        if self.available:
            torch.cuda.reset_peak_memory_stats()

    def stop(self) -> float:
        if self.available:
            # Đơn vị: Megabytes (MB)
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

# -------------------------------------------------------------------------
# NẠP DỮ LIỆU BENCHMARK GIẢ LẬP
# -------------------------------------------------------------------------
def seed_benchmark_data(db_path: str = "data/vectorstore") -> bool:
    """Đảm bảo dữ liệu Golden Dataset đã được nạp trong vector database"""
    db = get_vector_db_singleton(db_path)
    
    # Kiểm tra tài liệu PDF mẫu cs229-notes1.pdf
    pdf_exists = db.check_source_exists("cs229-notes1.pdf", "pdf")
    if not pdf_exists:
        print("🌱 Đang nạp dữ liệu cs229-notes1.pdf cho benchmark...")
        pdf_path = "data/raw/cs229-notes1.pdf"
        if os.path.exists(pdf_path):
            from src.ingestion.pdf_loader import PDFLoader
            loader = PDFLoader()
            pages = loader.load_and_convert(pdf_path)
            db.add_documents(
                text=pages,
                source_type="pdf",
                source_name="cs229-notes1.pdf"
            )
            print("🌱 Nạp cs229-notes1.pdf thành công.")
        else:
            print("⚠️ Cảnh báo: Không tìm thấy file data/raw/cs229-notes1.pdf để nạp.")
            # Nạp dữ liệu giả lập dự phòng nếu không có tệp
            pdf_text = (
                "Học có giám sát là quá trình học một hàm số h (giả thuyết) dựa trên dữ liệu huấn luyện (x, y). "
                "Hàm chi phí J đo lường sai lệch giữa dự đoán và thực tế, việc giảm thiểu J giúp tìm tham số tối ưu. "
                "Batch Gradient Descent cập nhật tham số theo hướng ngược đạo hàm của J trên toàn bộ dữ liệu. "
                "Để hồi quy bình phương tối thiểu là hợp lý, giả định sai số độc lập và đồng nhất (IID) theo phân phối chuẩn Gaussian."
            )
            db.add_documents(
                text=pdf_text,
                source_type="pdf",
                source_name="cs229-notes1.pdf",
                extra_metadata={"page": 1}
            )
            print("🌱 Nạp dữ liệu giả lập cs229-notes1.pdf thành công.")
            
    return True

# -------------------------------------------------------------------------
# HÀM CHẠY GENERATION ĐO TTFT & TOKEN USAGE
# -------------------------------------------------------------------------
def extract_text_from_chunk_content(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(content)

def generate_with_ttft_and_tokens(state: dict, model: ChatGoogleGenerativeAI) -> tuple[str, float, float, int, int]:
    """
    Sinh câu trả lời sử dụng Streaming để đo TTFT và tổng thời gian phản hồi.
    Đồng thời tính toán số lượng tokens gửi/nhận.
    """
    docs = state.get("reranked_docs", [])
    query = state["query"]
    
    if not docs:
        response_text = "❌ Rất tiếc, tôi không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu để trả lời câu hỏi này."
        # Tính toán token nhanh
        input_tokens = model.get_num_tokens(query)
        output_tokens = model.get_num_tokens(response_text)
        return response_text, 0.0, 0.0, input_tokens, output_tokens

    # Tạo ngữ cảnh giống generate_node
    context_parts = []
    for i, doc in enumerate(docs):
        src_type = doc.metadata.get("source_type", "unknown")
        src_name = doc.metadata.get("source_name", "Không rõ nguồn")
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
    
    prompt = f"""
    Bạn là "Trợ lý Ảo đa phương tiện thông minh". Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa TRÊN các tài liệu khảo sát được cung cấp dưới đây.
    
    YÊU CẦU BẮT BUỘC VỀ TRÍCH DẪN (CITATION):
    1. Câu trả lời của bạn phải hoàn toàn bằng Tiếng Việt và hành văn chuyên nghiệp.
    2. Ở mỗi khẳng định, luận điểm hoặc thông tin cụ thể bạn lấy từ tài liệu, bạn PHẢI dán nhãn trích dẫn chính xác ở ngay cuối câu hoặc cuối ý đó.
    3. Chỉ sử dụng thông tin trong tài liệu cung cấp. Tuyệt đối không tự bịa ra thông tin, số trang hoặc thời gian không có trong tài liệu.
    
    TÀI LIỆU KHẢO SÁT:
    {context_str}
    
    CÂU HỎI CỦA NGƯỜI DÙNG:
    {query}
    
    CÂU TRẢ LỜI CỦA BẠN (HÃY TRÍCH DẪN NGUỒN ĐẦY ĐỦ):
    """

    # Tính Input Token
    input_tokens = model.get_num_tokens(prompt)
    
    # Thực hiện gọi stream để đo TTFT
    start_time = time.perf_counter()
    first_token_time = None
    chunks = []
    
    try:
        for chunk in model.stream(prompt):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            content_str = extract_text_from_chunk_content(chunk.content)
            chunks.append(content_str)
    except Exception as e:
        print(f"⚠️ Lỗi streaming Gemini: {str(e)}")
        # Fallback to invoke
        res = model.invoke(prompt)
        content_str = extract_text_from_chunk_content(res.content)
        chunks = [content_str]
        first_token_time = time.perf_counter()
        
    end_time = time.perf_counter()
    
    response_text = "".join(chunks)
    ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
    generation_time = end_time - start_time
    
    # Tính Output Token
    output_tokens = model.get_num_tokens(response_text)
    
    return response_text, ttft, generation_time, input_tokens, output_tokens

# -------------------------------------------------------------------------
# RUN BENCHMARK CHO TỪNG CẤU HÌNH
# -------------------------------------------------------------------------
def run_benchmark_for_config(with_reranker: bool, db_path: str = "data/vectorstore") -> List[Dict[str, Any]]:
    """Chạy toàn bộ 5 câu hỏi của Golden Dataset trên cấu hình chỉ định và đo đạc chỉ số"""
    seed_benchmark_data(db_path)
    
    # Khởi tạo dịch vụ
    _, _, model = get_services()
    
    results = []
    
    for item in GOLDEN_DATASET:
        query = item["question"]
        source_type = item["source_type"]
        
        # 1. Đo lường RAM/GPU Peak & thời gian bước Retrieval
        ram_tracker = RAMTracker()
        gpu_tracker = GPUTracker()
        ram_tracker.start()
        gpu_tracker.start()
        
        state = {"query": query, "source_type": source_type}
        
        # Bắt đầu Retrieval
        start_ret = time.perf_counter()
        ret_res = retrieve_node(state)
        retrieval_time = time.perf_counter() - start_ret
        
        state.update(ret_res)
        
        # 2. Xử lý Reranking
        reranking_time = 0.0
        if with_reranker:
            # Bắt đầu Reranking
            start_rank = time.perf_counter()
            rerank_res = rerank_node(state)
            reranking_time = time.perf_counter() - start_rank
            state.update(rerank_res)
        else:
            # Bỏ qua reranking, lấy thẳng top 3 kết quả retrieve
            state["reranked_docs"] = state["retrieved_docs"][:3]
            
        # 3. Xử lý Generation (Đo TTFT & Token)
        response_text, ttft, generation_time, input_tokens, output_tokens = generate_with_ttft_and_tokens(state, model)
        
        peak_ram, peak_cpu = ram_tracker.stop()
        peak_gpu = gpu_tracker.stop()
        
        total_time = retrieval_time + reranking_time + generation_time
        
        # Thu thập thông tin ngữ cảnh để phục vụ đánh giá Ragas
        contexts = [doc.page_content for doc in state.get("reranked_docs", [])]
        if not contexts:
            contexts = [""] # Ragas yêu cầu context không rỗng
            
        results.append({
            "question": query,
            "answer": response_text,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "retrieval_time": retrieval_time,
            "reranking_time": reranking_time,
            "ttft": ttft,
            "generation_time": generation_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "peak_ram": peak_ram,
            "peak_cpu": peak_cpu,
            "peak_gpu": peak_gpu,
            "expected_source": item["expected_source"],
            "retrieved_sources": [doc.metadata.get("source_name") for doc in state.get("reranked_docs", [])]
        })
        
    return results

# -------------------------------------------------------------------------
# HÀM CHẠY ĐÁNH GIÁ RAGAS
# -------------------------------------------------------------------------
def run_ragas_evaluation(eval_results: List[Dict[str, Any]], api_key: str) -> Dict[str, float]:
    """
    Sử dụng khung đánh giá Ragas để chấm điểm chất lượng phản hồi
    Đo: Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    # Chuẩn bị dữ liệu định dạng Ragas
    data = {
        "question": [r["question"] for r in eval_results],
        "answer": [r["answer"] for r in eval_results],
        "contexts": [r["contexts"] for r in eval_results],
        "ground_truth": [r["ground_truth"] for r in eval_results]
    }
    
    # Khởi tạo mô hình đánh giá (dùng Gemini 3.1 Flash Lite)
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    
    # Khởi tạo các metric Ragas chính thức (sử dụng các lớp cũ tương thích tốt với Langchain)
    f = Faithfulness()
    f.llm = ragas_llm
    
    ar = AnswerRelevancy()
    ar.llm = ragas_llm
    ar.embeddings = ragas_emb
    
    cp = ContextPrecision()
    cp.llm = ragas_llm
    
    cr = ContextRecall()
    cr.llm = ragas_llm
    
    metrics = [f, ar, cp, cr]
    
    dataset = Dataset.from_dict(data)
    
    # Chạy đánh giá Ragas
    try:
        score_res = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_emb
        )
        
        # Tính toán thủ công giá trị trung bình từ danh sách chi tiết (score_res.scores)
        scores_list = getattr(score_res, "scores", [])
        
        def get_mean(metric_name: str) -> float:
            if not scores_list:
                return 0.0
            vals = [s.get(metric_name) for s in scores_list if s.get(metric_name) is not None]
            if not vals:
                return 0.0
            return float(sum(vals) / len(vals))
            
        return {
            "faithfulness": get_mean("faithfulness"),
            "answer_relevance": get_mean("answer_relevancy"),
            "context_precision": get_mean("context_precision"),
            "context_recall": get_mean("context_recall")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Lỗi chạy Ragas: {str(e)}")
        # Trả về điểm lỗi dự phòng
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }

# -------------------------------------------------------------------------
# BENCHMARK CHO PHÂN HỆ NẠP TÀI LIỆU (DOCLING & FASTER-WHISPER)
# -------------------------------------------------------------------------
def run_docling_benchmark(pdf_path: str) -> Dict[str, Any]:
    """Benchmark Docling: Phân tích PDF mẫu và đo lượng RAM/GPU tiêu hao đỉnh điểm"""
    from src.ingestion.pdf_loader import PDFLoader
    
    ram_tracker = RAMTracker()
    gpu_tracker = GPUTracker()
    
    ram_tracker.start()
    gpu_tracker.start()
    
    start_time = time.perf_counter()
    
    # Gọi hàm nạp PDF
    loader = PDFLoader()
    pages = loader.load_and_convert(pdf_path)
    
    duration = time.perf_counter() - start_time
    peak_ram, peak_cpu = ram_tracker.stop()
    peak_gpu = gpu_tracker.stop()
    
    return {
        "duration": duration,
        "peak_ram": peak_ram,
        "peak_cpu": peak_cpu,
        "peak_gpu": peak_gpu,
        "num_pages": len(pages)
    }

def run_youtube_benchmark(yt_url: str) -> Dict[str, Any]:
    """Benchmark Faster-Whisper: Tải & trích xuất phụ đề YouTube và đo lượng RAM/GPU tiêu hao"""
    from src.ingestion.yt_loader import YouTubeLoader
    
    ram_tracker = RAMTracker()
    gpu_tracker = GPUTracker()
    
    ram_tracker.start()
    gpu_tracker.start()
    
    start_time = time.perf_counter()
    
    # Gọi hàm nạp YouTube (Sử dụng model medium của Faster-Whisper)
    loader = YouTubeLoader(model_size="medium")
    chunks = loader.load_video(yt_url)
    
    duration = time.perf_counter() - start_time
    peak_ram, peak_cpu = ram_tracker.stop()
    peak_gpu = gpu_tracker.stop()
    
    return {
        "duration": duration,
        "peak_ram": peak_ram,
        "peak_cpu": peak_cpu,
        "peak_gpu": peak_gpu,
        "num_chunks": len(chunks) if chunks else 0
    }
