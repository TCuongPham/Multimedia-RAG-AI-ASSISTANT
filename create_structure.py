import os

# Cấu trúc thư mục
folders = [
    "src/ingestion",
    "src/retrieval",
    "src/graph",
    "src/utils",
    "data/raw",
    "data/vectorstore",
    "tests"
]

# Các file cần tạo
files = [
    ".env",
    "requirements.txt",
    "README.md",
    "app.py",
    "src/__init__.py",
    "src/ingestion/__init__.py",
    "src/retrieval/__init__.py",
    "src/graph/__init__.py",
    "src/utils/__init__.py",
    "tests/test_ingestion.py"
]

# Tạo thư mục
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Đã tạo thư mục: {folder}")

# Tạo file
for file in files:
    with open(file, 'w') as f:
        pass  # Tạo file rỗng
    print(f"Đã tạo file: {file}")