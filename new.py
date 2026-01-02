import pandas as pd
import numpy as np
import os

print("🔄 Đang bắt đầu quy trình tạo lại dữ liệu...")

# --- 1. TÌM FILE DỮ LIỆU GỐC ---
# Code sẽ tự tìm xem bạn đang có file nào
file_path = "books_cleaned.csv"
if not os.path.exists(file_path):
    file_path = "books.csv"

# Nếu không tìm thấy file nào, tạo dữ liệu giả để bạn test App
if not os.path.exists(file_path):
    print("⚠️ Không tìm thấy file gốc. Đang tạo dữ liệu mẫu (Dummy Data)...")
    data = {
        "isbn13": ["9780439785969", "9780345391803", "9780060935467", "9780375826689", "9780451524935"],
        "title": ["Harry Potter and the Half-Blood Prince", "The Hitchhiker's Guide to the Galaxy", "To Kill a Mockingbird", "Eragon", "1984"],
        "authors": ["J.K. Rowling", "Douglas Adams", "Harper Lee", "Christopher Paolini", "George Orwell"],
        "categories": ["Fiction, Fantasy", "Science Fiction", "Classics", "Fantasy", "Fiction"],
        "description": [
            "Harry Potter returns to Hogwarts. A story about magic and wizards.",
            "A comedy science fiction series created by Douglas Adams.",
            "A novel about racial injustice in the Deep South.",
            "One boy, one dragon, and a world of adventure.",
            "A dystopian social science fiction novel and cautionary tale."
        ],
        "thumbnail": ["", "", "", "", ""],
        "published_year": [2005, 1979, 1960, 2002, 1949],
        "average_rating": [4.5, 4.2, 4.3, 3.9, 4.6],
        "num_pages": [652, 224, 281, 503, 328],
        "ratings_count": [2000, 1500, 3000, 1000, 2500]
    }
    books = pd.DataFrame(data)
else:
    print(f"📖 Đã tìm thấy file: {file_path}")
    books = pd.read_csv(file_path, encoding="utf-8")

# --- 2. SỬA LỖI FORMAT (Cực quan trọng) ---
# Đảm bảo ISBN là chuỗi văn bản, không phải số khoa học
if "isbn13" in books.columns:
    books["isbn13"] = books["isbn13"].astype(str).str.replace(r'\.0$', '', regex=True)

# Đảm bảo có cột thumbnail
if "thumbnail" not in books.columns:
    books["thumbnail"] = ""

# --- 3. TỰ ĐỘNG PHÂN LOẠI & GÁN CẢM XÚC (Không cần AI) ---
def classify_and_emotion(row):
    text = (str(row.get('categories', '')) + " " + str(row.get('description', ''))).lower()
    
    # Phân loại Thể loại
    cat = 'Non-Fiction'
    if 'sci' in text or 'space' in text: cat = 'Science Fiction'
    elif 'fantasy' in text or 'dragon' in text: cat = 'Fantasy'
    elif 'mystery' in text or 'crime' in text: cat = 'Mystery'
    elif 'horror' in text: cat = 'Horror'
    elif 'love' in text: cat = 'Romance'
    elif 'fiction' in text: cat = 'Fiction'
    
    # Chấm điểm Cảm xúc (Dựa trên từ khóa)
    scores = {"joy": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, "surprise": 0.0}
    if "happy" in text or "love" in text: scores["joy"] += 1
    if "sad" in text or "death" in text: scores["sadness"] += 1
    if "dark" in text or "kill" in text: scores["fear"] += 1
    if "war" in text or "hate" in text: scores["anger"] += 1
    if "shock" in text: scores["surprise"] += 1
    
    return pd.Series([cat, scores["joy"], scores["sadness"], scores["fear"], scores["anger"], scores["surprise"]], 
                     index=['simple_categories', 'joy', 'sadness', 'fear', 'anger', 'surprise'])

# Áp dụng vào bảng dữ liệu
print("⚙️ Đang xử lý dữ liệu...")
processed = books.apply(classify_and_emotion, axis=1)
books = pd.concat([books, processed], axis=1)

# --- 4. LƯU FILE KẾT QUẢ ---
output_file = "books_with_emotions.csv"
books.to_csv(output_file, index=False, encoding="utf-8")

print("-" * 30)
print(f"✅ THÀNH CÔNG! Đã tạo file '{output_file}' với {len(books)} dòng.")
print("👉 Bây giờ bạn có thể chạy App Gradio được rồi!")