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
        if self.vector_store is not None:
            # Kiểm tra xem collection có thực sự tồn tại trên đĩa và hoạt động không
            # (Đề phòng trường hợp database bị xóa một phần trên Windows dẫn đến lệch pha dữ liệu)
            try:
                self.vector_store._collection.count()
            except Exception:
                print("⚠️ [RAG Store] Phát hiện collection bị hỏng hoặc mất thư mục dữ liệu vật lý. Đang tái khởi tạo...")
                self.vector_store = None

        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name="multimedia_rag"
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
            # Nếu là YouTube, tiến hành gộp nhóm các câu phụ đề cực ngắn (2-5 giây) thành các đoạn văn (khoảng 800 ký tự)
            # Việc này vừa tăng chất lượng ngữ cảnh cho RAG vừa giảm số lượng mảnh từ hàng nghìn xuống chỉ còn vài chục mảnh, loại bỏ lỗi Rate Limit
            if source_type == "youtube":
                merged_segments = []
                current_text = []
                current_len = 0
                start_timestamp = None
                
                for item in text:
                    if not isinstance(item, dict):
                        continue
                    item_text = item.get("text", item.get("content", ""))
                    timestamp = item.get("timestamp", "")
                    
                    if not item_text or not item_text.strip():
                        continue
                        
                    if not start_timestamp:
                        start_timestamp = timestamp
                        
                    current_text.append(item_text)
                    current_len += len(item_text)
                    
                    # Gộp đến khoảng 800 ký tự (tầm 150 từ, giữ ngữ cảnh tốt nhất)
                    if current_len >= 800:
                        merged_segments.append({
                            "text": " ".join(current_text),
                            "timestamp": start_timestamp,
                            "video_id": item.get("video_id", extra_metadata.get("video_id") if extra_metadata else "")
                        })
                        current_text = []
                        current_len = 0
                        start_timestamp = None
                        
                if current_text:
                    merged_segments.append({
                        "text": " ".join(current_text),
                        "timestamp": start_timestamp or "",
                        "video_id": extra_metadata.get("video_id") if extra_metadata else ""
                    })
                
                text = merged_segments

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
            
        # Thêm các mảnh vào vector store theo lô (batching) kèm delay để tránh lỗi Rate Limit 429 (RESOURCE_EXHAUSTED) của Gemini Embedding API
        import time
        batch_size = 90  # Tăng lô lên 90 mảnh (gần hạn mức tối đa 100 của Gemini) để giảm thiểu tối đa số lượng API request (giảm ~5 lần)
        try:
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                try:
                    store.add_documents(batch_chunks, ids=batch_ids)
                    # Nghỉ giữa các lô để hồi quota API
                    if i + batch_size < len(all_chunks):
                        time.sleep(2.0)
                except Exception as embed_err:
                    # Nếu bị Rate Limit, tự động nghỉ lâu hơn và thử lại một lần nữa
                    if "429" in str(embed_err) or "RESOURCE_EXHAUSTED" in str(embed_err):
                        print("⚠️ [RAG Store] Gặp lỗi Rate Limit (429). Đang tạm dừng 15 giây trước khi thử lại...")
                        time.sleep(15.0)
                        store.add_documents(batch_chunks, ids=batch_ids)
                    else:
                        raise embed_err
        except Exception as e:
            # TRANSACTION ROLLBACK: Nếu quá trình nạp bị lỗi giữa chừng, xóa toàn bộ các mảnh đã lưu trước đó của nguồn này
            print(f"❌ [RAG Store] Nạp thất bại giữa chừng. Đang dọn dẹp rollback các mảnh đã lưu để tránh dữ liệu rác...")
            try:
                store._collection.delete(where={
                    "$and": [
                        {"source_type": source_type},
                        {"source_name": source_name}
                    ]
                })
            except Exception as rollback_err:
                print(f"⚠️ [RAG Store] Lỗi rollback: {str(rollback_err)}")
            raise e
        
        print(f"✅ [RAG Store] Đã nạp thành công {len(all_chunks)} mảnh từ nguồn [{source_type.upper()}] - {source_name}")

    def search(self, query, source_type=None, source_name=None, k=5):
        """
        Tìm kiếm các đoạn văn liên quan nhất (Hỗ trợ lọc theo loại nguồn và tên nguồn cụ thể)
        :param query: Câu hỏi truy vấn
        :param source_type: Lọc tìm kiếm ('pdf', 'youtube', hoặc None để tìm kiếm chung)
        :param source_name: Lọc cụ thể theo tên file PDF hoặc tiêu đề YouTube
        :param k: Số lượng kết quả muốn trả về
        """
        store = self._get_vector_store()
        
        # Cấu hình bộ lọc (Chroma Metadata Filter)
        filter_dict = {}
        if source_type:
            filter_dict["source_type"] = source_type
        if source_name:
            filter_dict["source_name"] = source_name
            
        if not filter_dict:
            filter_dict = None
        elif len(filter_dict) == 1:
            # Chroma nhận dạng trực tiếp dictionary {khóa: giá trị} nếu chỉ có 1 điều kiện
            filter_dict = filter_dict
        else:
            # Chroma yêu cầu dùng cú pháp $and nếu lọc nhiều điều kiện
            filter_dict = {
                "$and": [{k: v} for k, v in filter_dict.items()]
            }
            
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
            
            # Đóng kết nối client để giải phóng file locks trên Windows
            try:
                if hasattr(self.vector_store, "_client") and hasattr(self.vector_store._client, "close"):
                    self.vector_store._client.close()
            except Exception as e:
                print(f"⚠️ [RAG Store] Không thể đóng client ChromaDB: {e}")
                
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

    def get_all_sources(self):
        """
        Lấy danh sách tất cả các nguồn tài liệu đã được nạp trong cơ sở dữ liệu.
        Trả về danh sách các dict chứa thông tin {source_name, source_type, video_id}
        """
        store = self._get_vector_store()
        try:
            result = store.get(include=["metadatas"])
            metadatas = result.get("metadatas", [])
            if not metadatas:
                return []
            
            unique_sources = {}
            for meta in metadatas:
                if not meta:
                    continue
                name = meta.get("source_name")
                stype = meta.get("source_type")
                if name and stype:
                    unique_sources[(name, stype)] = {
                        "source_name": name,
                        "source_type": stype,
                        "video_id": meta.get("video_id", "")
                    }
            return list(unique_sources.values())
        except Exception as e:
            print(f"⚠️ [RAG Store] Lỗi lấy danh sách nguồn: {str(e)}")
            return []

    def delete_source(self, source_name: str, source_type: str) -> bool:
        """
        Xóa toàn bộ các chunk liên quan đến một nguồn cụ thể
        """
        store = self._get_vector_store()
        try:
            store._collection.delete(where={
                "$and": [
                    {"source_type": source_type},
                    {"source_name": source_name}
                ]
            })
            print(f"🧹 [RAG Store] Đã xóa nguồn [{source_type.upper()}] - {source_name}")
            return True
        except Exception as e:
            print(f"⚠️ [RAG Store] Lỗi khi xóa nguồn: {str(e)}")
            return False

    def rename_source(self, old_name: str, new_name: str, source_type: str) -> bool:
        """
        Đổi tên của nguồn tài liệu trong cơ sở dữ liệu bằng cách cập nhật metadata của tất cả các chunk liên quan.
        """
        if not new_name or old_name == new_name:
            return False
            
        store = self._get_vector_store()
        try:
            result = store.get(
                where={
                    "$and": [
                        {"source_type": source_type},
                        {"source_name": old_name}
                    ]
                },
                include=["metadatas"]
            )
            
            ids = result.get("ids", [])
            metadatas = result.get("metadatas", [])
            
            if not ids:
                print(f"⚠️ [RAG Store] Không tìm thấy chunk nào cho nguồn [{source_type.upper()}] - {old_name}")
                return False
                
            updated_metadatas = []
            for meta in metadatas:
                new_meta = meta.copy()
                new_meta["source_name"] = new_name
                updated_metadatas.append(new_meta)
                
            store._collection.update(ids=ids, metadatas=updated_metadatas)
            print(f"📝 [RAG Store] Đã đổi tên nguồn [{source_type.upper()}] từ '{old_name}' thành '{new_name}'")
            return True
        except Exception as e:
            print(f"⚠️ [RAG Store] Lỗi khi đổi tên nguồn: {str(e)}")
            return False

# Singleton helper for application-wide resource sharing
_vector_db_instance = None

def get_vector_db_singleton(db_path="data/vectorstore"):
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDBManager(db_path)
    return _vector_db_instance