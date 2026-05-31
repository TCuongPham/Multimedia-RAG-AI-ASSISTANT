import os
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
import youtube_transcript_api 
from src.utils.helpers import save_to_txt 

class YouTubeLoader:
    def __init__(self, model_size="medium"):
        """
        Khởi tạo Bộ nạp dữ liệu YouTube.
        Áp dụng Lazy Loading: Không nạp mô hình Whisper ngay lập tức để tiết kiệm RAM.
        """
        self.model_size = model_size
        self.whisper_model = None  # Sẽ nạp thực sự khi phương án 1 thất bại

    def format_timestamp(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"[{minutes:02d}:{secs:02d}]"

    def get_official_transcript(self, video_id):
        """Phương án 1: Lấy phụ đề gốc bằng youtube-transcript-api"""
        try:
            print("[YT Loader] Phương án 1: Đang thử lấy phụ đề gốc từ YouTube...")

            transcript = youtube_transcript_api.YouTubeTranscriptApi().fetch(video_id, languages=['en', 'vi'])
            
            result_chunks = []
            for entry in transcript:
                if isinstance(entry, dict):
                    start = entry.get('start', 0)
                    text = entry.get('text', '')
                else:
                    start = getattr(entry, 'start', 0)
                    text = getattr(entry, 'text', '')
                    
                start_time = self.format_timestamp(start)
                text = text.replace('\n', ' ')
                result_chunks.append({
                    "text": text,
                    "timestamp": start_time
                })
                
            print("THÀNH CÔNG: Đã cào được phụ đề gốc có sẵn!")
            return result_chunks
            
        except Exception as e:
            print(f"⚠️ Thất bại khi lấy phụ đề gốc: {str(e)}")
            return None 

    def download_and_transcribe_with_whisper(self, url, video_id):
        """Phương án 2: Tải Audio và nhận diện giọng nói bằng AI Whisper"""
        print("[YT Loader] Phương án 2: Kích hoạt AI Whisper làm phương án dự phòng...")
        
        # LAZY LOADING: Chỉ nạp mô hình vào RAM khi thực sự bắt đầu xử lý Audio
        if self.whisper_model is None:
            print(f"🔄 Đang nạp mô hình Whisper ({self.model_size}) vào bộ nhớ máy...")
            self.whisper_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            print("✅ Đã nạp mô hình Reranker/Whisper thành công!")

        # Cấu hình tải Audio bằng yt-dlp
        out_template = f"data/raw/{video_id}.%(ext)s"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_template,
            'quiet': True, 
            'ffmpeg_location': r'C:\Users\Pham Cuong\Desktop\UNI\AI\PRJ 2\ffmpeg-2026-04-30-git-cc3ca17127-full_build\bin\ffmpeg.exe',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        try:
            print("📥 Đang tải Audio từ YouTube (Quá trình này có thể tốn ít phút)...")
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            mp3_path = f"data/raw/{video_id}.mp3"
            
            if not os.path.exists(mp3_path):
                raise FileNotFoundError(f"Không tìm thấy file audio đã tải tại: {mp3_path}")

            print("Đang chạy AI Whisper để dịch âm thanh thành văn bản...")
            segments, info = self.whisper_model.transcribe(mp3_path, beam_size=5)
            
            result_chunks = []
            for segment in segments:
                start_time = self.format_timestamp(segment.start)
                result_chunks.append({
                    "text": segment.text,
                    "timestamp": start_time
                })

            # Dọn dẹp file rác .mp3 sau khi trích xuất xong
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
                print("Đã dọn dẹp file audio tạm.")
                
            return result_chunks

        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng khi chạy luồng Whisper: {str(e)}")
            return []

    def load_video(self, url):
        """
        HÀM ĐIỀU PHỐI CHÍNH (MAIN ENTRYPOINT):
        Nhận vào link YouTube, tự trích xuất ID, quyết định luồng chạy tối ưu.
        """
        # Trích xuất video_id từ URL
        video_id = None
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = url # Phòng trường hợp người dùng truyền thẳng ID
            
        # Thử thực hiện Phương án 1
        transcript_text = self.get_official_transcript(video_id)
        
        # Nếu Phương án 1 thất bại (trả về None), chuyển sang Phương án 2
        if transcript_text is None:
            transcript_text = self.download_and_transcribe_with_whisper(url, video_id)
            
        return transcript_text