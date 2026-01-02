import pandas as pd

# Đọc file
df = pd.read_csv("books_with_emotions.csv", encoding="utf-8")

# Kiểm tra các cột cảm xúc
emotions = ['joy', 'sadness', 'fear', 'anger', 'surprise']
print(f"Tổng số dòng: {len(df)}")
print("-" * 30)

for emo in emotions:
    if emo in df.columns:
        # Lấy 5 giá trị đầu tiên để xem
        vals = df[emo].head(5).tolist()
        print(f"Cột '{emo}': {vals}")
        if df[emo].sum() == 0:
            print(f"⚠️ CẢNH BÁO: Cột '{emo}' toàn số 0! Sắp xếp sẽ không hoạt động.")
    else:
        # Tìm cột tương tự (ví dụ joy_x)
        similar = [c for c in df.columns if emo in c]
        print(f"❌ Không tìm thấy cột '{emo}'. Có thể là: {similar}")