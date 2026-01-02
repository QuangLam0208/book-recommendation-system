import pandas as pd
import numpy as np
import random
import os

# 1. Setup đường dẫn tuyệt đối (tránh lỗi không tìm thấy file)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(base_dir, "books_with_emotions.csv")
ratings_file_path = os.path.join(base_dir, "ratings.csv")

# 2. Đọc file sách
if not os.path.exists(csv_file_path):
    # Fallback nếu tên file khác
    csv_file_path = os.path.join(base_dir, "books_cleaned.csv")

try:
    print(f"📖 Đang đọc sách từ: {csv_file_path}")
    books = pd.read_csv(csv_file_path)
    
    # Lấy TẤT CẢ ISBN và xử lý chuỗi (bỏ .0)
    if "isbn13" in books.columns:
        all_isbns = books['isbn13'].astype(str).str.replace(r'\.0$', '', regex=True).unique().tolist()
    else:
        print("❌ File sách không có cột 'isbn13'.")
        exit()
        
    print(f"📚 Tổng số sách cần tạo rating: {len(all_isbns)}")

except Exception as e:
    print(f"❌ Lỗi đọc file: {e}")
    exit()

# 3. SINH DỮ LIỆU RATINGS (ĐẢM BẢO PHỦ KÍN 100%)
user_ids = []
book_isbns = []
ratings = []

print("⏳ Đang sinh dữ liệu (Chế độ: Phủ kín 100% sách)...")

# Giai đoạn 1: Ép buộc MỖI cuốn sách phải được rate bởi ít nhất 2 người
# (Để đảm bảo sách nào tìm cũng thấy có dữ liệu)
for isbn in all_isbns:
    # Giả sử User 1 đến User 5 là những "nhà phê bình" đọc hết mọi sách
    for critic_id in range(1, 4): 
        user_ids.append(critic_id)
        book_isbns.append(isbn)
        ratings.append(np.random.randint(3, 6)) # Rate từ 3 đến 5 sao

# Giai đoạn 2: Tạo thêm nhiễu ngẫu nhiên (cho tự nhiên)
NUM_EXTRA_RATINGS = 5000 
for _ in range(NUM_EXTRA_RATINGS):
    user_ids.append(np.random.randint(10, 1000)) # User ngẫu nhiên từ 10-1000
    book_isbns.append(random.choice(all_isbns))
    ratings.append(np.random.randint(1, 6))

# 4. Lưu file
df_ratings = pd.DataFrame({
    "user_id": user_ids,
    "isbn": book_isbns,
    "rating": ratings
})

df_ratings.to_csv(ratings_file_path, index=False)
print(f"✅ Đã tạo xong '{ratings_file_path}'")
print(f"📊 Tổng số dòng đánh giá: {len(df_ratings)}")
print("👉 Bây giờ bạn hãy chạy lại gradio-dashboard.py để test nhé!")