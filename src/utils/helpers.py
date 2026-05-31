import os

def save_to_txt(content, file_path):
    """
    Lưu nội dung vào file txt.
    :param content: Nội dung muốn lưu (string)
    :param file_path: Đường dẫn lưu file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Mở file với mode 'w' (ghi đè) hoặc 'a' (ghi tiếp)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Da luu thanh cong tai: {file_path}")