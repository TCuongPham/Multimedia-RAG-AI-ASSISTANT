import sys
import os

# Cấu hình path để nhận diện thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.yt_loader import YouTubeLoader
from src.utils.helpers import save_to_txt

def format_transcript_to_str(transcript):
    if not transcript:
        return ""
    if isinstance(transcript, str):
        return transcript
    
    result_text = ""
    for entry in transcript:
        result_text += f"**{entry['timestamp']}**: {entry['text']}\n"
    return result_text

def test_youtube_loader():
    # Link video test (Đảm bảo thư mục data/raw/ đã được tạo sẵn)
    url = "https://www.youtube.com/watch?v=iNyUmbmQQZg"
    
    # Trích xuất video_id chuẩn xác
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = url

    print(f"\n============= 🧪 BẮT ĐẦU KIỂM THỬ YOUTUBE LOADER =============")
    print(f"🎬 Video ID mục tiêu: {video_id}")
    print(f"==============================================================\n")
    
    # Khởi tạo loader (Lúc này Whisper chưa bị nạp vào RAM nhờ Lazy Loading)
    loader = YouTubeLoader(model_size="medium")
    
    # -------------------------------------------------------------------------
    # PHẦN 1: Thu thập dữ liệu độc lập để làm Thực nghiệm & Đánh giá (Báo cáo Đồ án)
    # -------------------------------------------------------------------------
    print("--- 📊 THỰC NGHIỆM 1: Trích xuất Phụ đề gốc YouTube ---")
    yt_transcript = loader.get_official_transcript(video_id)
    
    if yt_transcript:
        save_to_txt(format_transcript_to_str(yt_transcript), f"data/raw/{video_id}_youtube_official.txt")
        print(f"💾 Đã lưu phụ đề gốc vào: data/raw/{video_id}_youtube_official.txt\n")
    else:
        print("⚠️ Không thể lưu vì video không có phụ đề gốc công khai.\n")
    
    print("--- 📊 THỰC NGHIỆM 2: Trích xuất bằng AI Whisper (Tải Audio + Nhận diện) ---")
    whisper_transcript = loader.download_and_transcribe_with_whisper(url, video_id)
    
    if isinstance(whisper_transcript, list):
        save_to_txt(format_transcript_to_str(whisper_transcript), f"data/raw/{video_id}_whisper_ai.txt")
        print(f"💾 Đã lưu kết quả AI Whisper vào: data/raw/{video_id}_whisper_ai.txt\n")
    else:
        print(f"💥 Thất bại ở luồng Whisper: {whisper_transcript}\n")

    # -------------------------------------------------------------------------
    # PHẦN 2: Kiểm thử Luồng chạy thực tế của Hệ thống RAG (Production Pipeline)
    # -------------------------------------------------------------------------
    print("--- 🔄 THỰC NGHIỆM 3: Kiểm thử Hàm điều phối thông minh (load_video) ---")
    print("Luồng này sẽ tự động quyết định lấy Sub gốc hay chạy Whisper...")
    
    final_rag_context = loader.load_video(url)
    save_to_txt(format_transcript_to_str(final_rag_context), f"data/raw/{video_id}_final_rag_input.txt")
    
    print(f"💾 Đã lưu văn bản ngữ cảnh cuối cùng phục vụ RAG vào: data/raw/{video_id}_final_rag_input.txt")
    print(f"\n====================== ✨ HOÀN TẤT KIỂM THỬ ======================")

if __name__ == "__main__":
    test_youtube_loader()