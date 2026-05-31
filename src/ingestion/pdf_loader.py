import os
from docling.document_converter import DocumentConverter
from src.utils.helpers import save_to_txt

class PDFLoader:
    def __init__(self):
        print("⚙️ Khởi tạo Bộ nạp tài liệu Docling nâng cao...")
        self.converter = DocumentConverter()

    def load_and_convert(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file tại: {file_path}")

        print(f"🔍 [Docling Loader] Đang phân tích và trích xuất Markdown theo trang: {os.path.basename(file_path)}...")
        result = self.converter.convert(file_path)
        doc = result.document
        
        pages = []
        filename = os.path.basename(file_path)
        
        for page_no in sorted(doc.pages.keys()):
            page_text = doc.export_to_markdown(page_no=page_no)
            
            pages.append({
                "content": page_text,
                "metadata": {
                    "source": file_path,
                    "filename": filename,
                    "page": str(page_no)
                }
            })
            
        print(f"🎉 THÀNH CÔNG: Đã bóc tách thành công {len(pages)} trang Markdown sạch!")
        return pages

if __name__ == "__main__":
    loader = PDFLoader()
    
    try:
        pages = loader.load_and_convert("data/raw/docling.pdf")
        
        # Ghép nối nội dung các trang để ghi nhận kết quả kiểm thử thô
        combined_text = ""
        for page in pages:
            combined_text += f"\n\n"
            combined_text += page["content"]
            combined_text += f"\n\n"
            
        save_to_txt(combined_text, "data/raw/test_output.txt")
        print("💾 Đã ghi nhận kết quả kiểm thử tại: data/raw/test_output.txt")
    except FileNotFoundError as e:
        print(f"⚠️ Chạy thử nghiệm docling.pdf thất bại: {e}")