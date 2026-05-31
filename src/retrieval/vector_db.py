from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

class VectorDBManager:
    def __init__(self, db_path="data/vectorstore"):
        self.db_path = db_path
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_store = None

    def _get_vector_store(self):
        """Khởi tạo hoặc tải lại Vector Store hiện có"""
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
        return self.vector_store

    def check_source_exists(self, source_name: str, source_type: str) -> bool:
        """
        Kiểm tra nhanh xem tên tài liệu hoặc ID video này đã từng được nạp chưa
        """
        store = self._get_vector_store()
        try:
            # Truy vấn bằng bộ lọc metadata thay vì kiểm tra ID ch0 cứng để tăng độ tin cậy
            result = store.get(
                where={
                    "$and": [
                        {"source_type": source_type},
                        {"source_name": source_name}
                    ]
                },
                limit=1
            )
            return len(result['ids']) > 0
        except Exception:
            return False

    def add_documents(self, text, source_type, source_name, extra_metadata=None):
        """
        Chia nhỏ văn bản và lưu NỐI TIẾP vào ChromaDB kèm theo nhãn phân loại nguồn (pdf/youtube)
        Hỗ trợ tham số `text` truyền vào là chuỗi (string) hoặc danh sách các mảnh có cấu trúc (list[dict]).
        :param text: Nội dung chữ thô hoặc danh sách mảnh dữ liệu có cấu trúc
        :param source_type: Loại nguồn ('pdf' hoặc 'youtube')
        :param source_name: Tên file PDF hoặc tiêu đề Video YouTube
        :param extra_metadata: Các thông tin bổ sung khác (ví dụ: page, video_id, timestamp...)
        """
        # 1. Cấu hình Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Độ dài mỗi mảnh (khoảng 200-300 từ)
            chunk_overlap=100,    # Đoạn chồng lấn để không mất ngữ cảnh giữa các mảnh
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        all_chunks = []
        
        if isinstance(text, list):
            for item in text:
                if not isinstance(item, dict):
                    item_text = str(item)
                    item_meta = {}
                else:
                    if source_type == "pdf":
                        item_text = item.get("content", item.get("text", ""))
                        item_meta = item.get("metadata", item.get("extra_metadata", {}))
                    elif source_type == "youtube":
                        item_text = item.get("text", item.get("content", ""))
                        item_meta = {}
                        if "timestamp" in item:
                            item_meta["timestamp"] = item["timestamp"]
                        if "video_id" in item:
                            item_meta["video_id"] = item["video_id"]
                    else:
                        item_text = item.get("text", item.get("content", ""))
                        item_meta = {k: v for k, v in item.items() if k not in ["text", "content"]}

                if not item_text or not item_text.strip():
                    continue

                # Tạo bộ nhãn (metadata) kết hợp
                merged_meta = {
                    "source_type": source_type,
                    "source_name": source_name
                }
                if extra_metadata:
                    merged_meta.update(extra_metadata)
                merged_meta.update(item_meta)

                # Chia nhỏ text cho phần tử hiện tại và giữ lại metadata riêng lẻ
                item_chunks = text_splitter.create_documents([item_text], metadatas=[merged_meta])
                all_chunks.extend(item_chunks)
        else:
            # Xây dựng bộ nhãn (metadata) chuẩn hóa thông thường cho chuỗi text đơn lẻ
            metadata = {
                "source_type": source_type,
                "source_name": source_name
            }
            if extra_metadata:
                metadata.update(extra_metadata)
                
            all_chunks = text_splitter.create_documents([text], metadatas=[metadata])

        if not all_chunks:
            print(f"⚠️ [RAG Store] Không có mảnh văn bản hợp lệ nào được sinh ra từ nguồn [{source_type.upper()}] - {source_name}")
            return
            
        safe_source_name = "".join([c if c.isalnum() else "_" for c in source_name])
        chunk_ids = [f"{source_type}_{safe_source_name}_ch{i}" for i in range(len(all_chunks))]
        
        # 4. Lấy Vector Store hiện tại, thực hiện dọn dẹp mảnh cũ để tránh rác trùng lặp và ghi đè
        store = self._get_vector_store()
        try:
            store._collection.delete(where={
                "$and": [
                    {"source_type": source_type},
                    {"source_name": source_name}
                ]
            })
            print(f"🧹 [RAG Store] Đã dọn dẹp các mảnh cũ của nguồn [{source_type.upper()}] - {source_name}")
        except Exception as e:
            # Nếu chưa có collection hoặc trống, bỏ qua
            print(f"⚠️ [RAG Store] Bỏ qua dọn dẹp: {str(e)}")
            
        store.add_documents(all_chunks, ids=chunk_ids)
        
        print(f"✅ [RAG Store] Đã nạp thành công {len(all_chunks)} mảnh từ nguồn [{source_type.upper()}] - {source_name}")

    def search(self, query, source_type=None, k=5):
        """
        Tìm kiếm các đoạn văn liên quan nhất (Hỗ trợ lọc theo loại nguồn)
        :param query: Câu hỏi truy vấn
        :param source_type: Lọc tìm kiếm ('pdf', 'youtube', hoặc None để tìm kiếm chung)
        :param k: Số lượng kết quả muốn trả về
        """
        store = self._get_vector_store()
        
        # Cấu hình bộ lọc (Chroma Metadata Filter)
        filter_dict = None
        if source_type:
            filter_dict = {"source_type": source_type}
            
        results = store.similarity_search(query, k=k, filter=filter_dict)
        return results

    def clear_db(self):
        """Dọn dẹp/Xóa hoàn toàn database để làm lại từ đầu"""
        # 1. Giải phóng bộ nhớ và đối tượng Chroma để giải phóng khóa file (file locks) trên Windows
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except Exception:
                pass
            self.vector_store = None
            
        # 2. Gọi Garbage Collector của Python để giải phóng triệt để các file handle đang mở
        import gc
        gc.collect()
        
        # 3. Xóa thư mục lưu trữ một cách an toàn (tránh lỗi WinError 32)
        import shutil
        if os.path.exists(self.db_path):
            try:
                shutil.rmtree(self.db_path)
                print("🗑️ Đã xóa hoàn toàn dữ liệu Vector Database!")
            except PermissionError:
                print("⚠️ [Windows Lock] Thư mục database đang được hệ thống sử dụng và sẽ tự động được dọn dẹp ở lần chạy tiếp theo.")
            except Exception as e:
                print(f"⚠️ Không thể xóa thư mục dữ liệu: {str(e)}")