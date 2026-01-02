import pandas as pd
import numpy as np

print("🔄 Đang cập nhật dữ liệu cảm xúc...")

# 1. Đọc file hiện tại
try:
    df = pd.read_csv("books_with_emotions.csv", encoding="utf-8")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file books_with_emotions.csv")
    exit()

# 2. Điền dữ liệu ngẫu nhiên vào các cột cảm xúc
# (Giúp việc sắp xếp thay đổi rõ rệt khi bạn chọn Tone khác nhau)
emotions = ['joy', 'sadness', 'fear', 'anger', 'surprise']

for emo in emotions:
    # Tạo điểm số từ 0.0 đến 1.0 cho mỗi cuốn sách
    df[emo] = np.random.uniform(0, 1, size=len(df))

# 3. Lưu lại file
df.to_csv("books_with_emotions.csv", index=False, encoding="utf-8")

print(f"✅ Đã cập nhật xong {len(df)} dòng dữ liệu!")
print("👉 Hãy khởi động lại App để thấy sự thay đổi.")