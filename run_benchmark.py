import os
import json
import time
import sys
from dotenv import load_dotenv

# Cấu hình stdout hỗ trợ ký tự UTF-8 trên Windows Terminal
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Nạp biến môi trường
load_dotenv()

from src.utils.benchmark import run_benchmark_for_config, run_ragas_evaluation

def format_summary_table(results_with, results_without, ragas_with=None, ragas_without=None):
    """Tạo bảng so sánh Markdown tổng hợp hiệu năng giữa 2 phiên bản"""
    
    # Tính toán các chỉ số trung bình cho With Reranker
    avg_ret_with = sum(r["retrieval_time"] for r in results_with) / len(results_with)
    avg_rank_with = sum(r["reranking_time"] for r in results_with) / len(results_with)
    avg_ttft_with = sum(r["ttft"] for r in results_with) / len(results_with)
    avg_gen_with = sum(r["generation_time"] for r in results_with) / len(results_with)
    avg_total_with = sum(r["total_time"] for r in results_with) / len(results_with)
    avg_in_tokens_with = sum(r["input_tokens"] for r in results_with) / len(results_with)
    avg_out_tokens_with = sum(r["output_tokens"] for r in results_with) / len(results_with)
    peak_ram_with = max(r["peak_ram"] for r in results_with)
    peak_cpu_with = max(r["peak_cpu"] for r in results_with)
    peak_gpu_with = max(r["peak_gpu"] for r in results_with)
    
    # Tính toán các chỉ số trung bình cho Without Reranker
    avg_ret_without = sum(r["retrieval_time"] for r in results_without) / len(results_without)
    avg_rank_without = sum(r["reranking_time"] for r in results_without) / len(results_without)
    avg_ttft_without = sum(r["ttft"] for r in results_without) / len(results_without)
    avg_gen_without = sum(r["generation_time"] for r in results_without) / len(results_without)
    avg_total_without = sum(r["total_time"] for r in results_without) / len(results_without)
    avg_in_tokens_without = sum(r["input_tokens"] for r in results_without) / len(results_without)
    avg_out_tokens_without = sum(r["output_tokens"] for r in results_without) / len(results_without)
    peak_ram_without = max(r["peak_ram"] for r in results_without)
    peak_cpu_without = max(r["peak_cpu"] for r in results_without)
    peak_gpu_without = max(r["peak_gpu"] for r in results_without)
    
    table = []
    table.append("| Chỉ số đo lường (Trung bình) | Có Reranker (With Reranker) | Không Reranker (Without Reranker) | Đánh giá học thuật |")
    table.append("| :--- | :---: | :---: | :--- |")
    
    table.append(f"| **Retrieval Latency (ChromaDB)** | {avg_ret_with:.4f} giây | {avg_ret_without:.4f} giây | Tốc độ quét vector ổn định |")
    table.append(f"| **Reranking Latency (BGE)** | {avg_rank_with:.4f} giây | {avg_rank_without:.4f} giây | Không Reranker = 0s nhưng giảm độ liên quan |")
    table.append(f"| **TTFT (Time to First Token)** | {avg_ttft_with:.4f} giây | {avg_ttft_without:.4f} giây | Thời gian phản hồi token đầu tiên |")
    table.append(f"| **Generation Latency (Gemini)** | {avg_gen_with:.4f} giây | {avg_gen_without:.4f} giây | Thời gian tổng hợp và sinh chữ |")
    table.append(f"| **Tổng thời gian phản hồi (Total)** | **{avg_total_with:.4f} giây** | **{avg_total_without:.4f} giây** | Có Reranker chậm hơn do chạy BGE CrossEncoder |")
    table.append(f"| **Input Tokens trung bình** | {avg_in_tokens_with:.1f} tokens | {avg_in_tokens_without:.1f} tokens | Số lượng token truyền vào prompt |")
    table.append(f"| **Output Tokens trung bình** | {avg_out_tokens_with:.1f} tokens | {avg_out_tokens_without:.1f} tokens | Chi phí sinh câu thoại |")
    table.append(f"| **Tiêu thụ RAM đỉnh điểm (Peak)** | {peak_ram_with:.2f} MB | {peak_ram_without:.2f} MB | Đo lượng bộ nhớ hệ thống tiêu tốn |")
    table.append(f"| **Tiêu thụ CPU đỉnh điểm (Peak)** | {peak_cpu_with:.1f}% | {peak_cpu_without:.1f}% | Đo lượng phần trăm CPU hệ thống tiêu thụ |")
    
    if peak_gpu_with > 0 or peak_gpu_without > 0:
        table.append(f"| **Tiêu thụ GPU đỉnh điểm (Peak)** | {peak_gpu_with:.2f} MB | {peak_gpu_without:.2f} MB | Chỉ áp dụng nếu chạy trên card NVIDIA CUDA |")
    else:
        table.append("| **Tiêu thụ GPU đỉnh điểm (Peak)** | CPU Only | CPU Only | Không phát hiện phần cứng CUDA |")
        
    if ragas_with and ragas_without:
        table.append(f"| **Ragas Faithfulness (Độ trung thực)** | **{ragas_with.get('faithfulness', 0.0):.4f}** | **{ragas_without.get('faithfulness', 0.0):.4f}** | Điểm cao hơn = ít bị ảo tưởng thông tin |")
        table.append(f"| **Ragas Answer Relevancy (Độ liên quan)** | **{ragas_with.get('answer_relevance', 0.0):.4f}** | **{ragas_without.get('answer_relevance', 0.0):.4f}** | Điểm cao hơn = trả lời đúng trọng tâm câu hỏi |")
        table.append(f"| **Ragas Context Precision (Chính xác ngữ cảnh)** | {ragas_with.get('context_precision', 0.0):.4f} | {ragas_without.get('context_precision', 0.0):.4f} | Khả năng xếp tài liệu đúng lên hàng đầu |")
        table.append(f"| **Ragas Context Recall (Phủ ngữ cảnh)** | {ragas_with.get('context_recall', 0.0):.4f} | {ragas_without.get('context_recall', 0.0):.4f} | Đo đạc độ bao phủ câu trả lời chuẩn |")
        
    return "\n".join(table)

def main():
    print("==================================================")
    print("🚀 BẮT ĐẦU CHẠY KIỂM THỬ RAG BENCHMARK & EVALUATION")
    print("==================================================")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Lỗi: Không tìm thấy GOOGLE_API_KEY trong file .env hoặc hệ thống!")
        return

    # 1. Chạy benchmark cấu hình Có Reranker
    print("\n--- 1. Đang chạy Benchmark: Cấu hình CÓ RERANKER (With Reranker) ---")
    results_with = run_benchmark_for_config(with_reranker=True)
    print("✅ Đã hoàn thành cấu hình Có Reranker.")
    
    # 2. Chạy benchmark cấu hình Không Reranker
    print("\n--- 2. Đang chạy Benchmark: Cấu hình KHÔNG RERANKER (Without Reranker) ---")
    results_without = run_benchmark_for_config(with_reranker=False)
    print("✅ Đã hoàn thành cấu hình Không Reranker.")
    
    # 3. Chạy đánh giá chất lượng bằng Ragas
    print("\n--- 3. Đang đánh giá chất lượng tự động bằng khung Ragas (LLM-as-a-judge) ---")
    print("Đang kết nối Gemini Evaluator để chấm điểm Ragas...")
    
    ragas_with = run_ragas_evaluation(results_with, api_key)
    print(f"Ragas scores (Có Reranker): {ragas_with}")
    
    ragas_without = run_ragas_evaluation(results_without, api_key)
    print(f"Ragas scores (Không Reranker): {ragas_without}")
    
    # 4. Hiển thị báo cáo kết quả ra Terminal dưới dạng Markdown
    print("\n==========================================================================")
    print("📊 BÁO CÁO KẾT QUẢ SO SÁNH HIỆU NĂNG VÀ CHẤT LƯỢNG RAG")
    print("==========================================================================\n")
    
    summary_md = format_summary_table(results_with, results_without, ragas_with, ragas_without)
    print(summary_md)
    
    # 5. Lưu báo cáo dạng JSON
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_comparison": {
            "with_reranker": {
                "ragas_metrics": ragas_with,
                "detailed_queries": [
                    {
                        "question": r["question"],
                        "answer": r["answer"],
                        "retrieved_sources": r["retrieved_sources"],
                        "latency": {
                            "retrieval": r["retrieval_time"],
                            "reranking": r["reranking_time"],
                            "ttft": r["ttft"],
                            "generation": r["generation_time"],
                            "total": r["total_time"]
                        },
                        "token_usage": {
                            "input_tokens": r["input_tokens"],
                            "output_tokens": r["output_tokens"]
                        },
                        "peak_resources": {
                            "ram_mb": r["peak_ram"],
                            "gpu_mb": r["peak_gpu"]
                        }
                    } for r in results_with
                ]
            },
            "without_reranker": {
                "ragas_metrics": ragas_without,
                "detailed_queries": [
                    {
                        "question": r["question"],
                        "answer": r["answer"],
                        "retrieved_sources": r["retrieved_sources"],
                        "latency": {
                            "retrieval": r["retrieval_time"],
                            "reranking": r["reranking_time"],
                            "ttft": r["ttft"],
                            "generation": r["generation_time"],
                            "total": r["total_time"]
                        },
                        "token_usage": {
                            "input_tokens": r["input_tokens"],
                            "output_tokens": r["output_tokens"]
                        },
                        "peak_resources": {
                            "ram_mb": r["peak_ram"],
                            "gpu_mb": r["peak_gpu"]
                        }
                    } for r in results_without
                ]
            }
        }
    }
    
    os.makedirs("data", exist_ok=True)
    report_path = "data/benchmark_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n💾 Đã lưu báo cáo chi tiết vào tệp: {report_path}")
    print("\n==================================================")
    print("🎉 HOÀN TẤT QUÁ TRÌNH CHẠY RAG BENCHMARK!")
    print("==================================================")

if __name__ == "__main__":
    main()
