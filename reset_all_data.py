import pandas as pd
import numpy as np
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "books_with_emotions.csv")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
RATINGS_FILE = os.path.join(BASE_DIR, "ratings.csv")

def reset_data():
    print("🚀 Bắt đầu quá trình Reset toàn bộ dữ liệu...")

    # 1. ĐỌC FILE SÁCH GỐC
    if not os.path.exists(CSV_FILE):
        # Fallback nếu tên file khác
        backup_file = os.path.join(BASE_DIR, "books_cleaned.csv")
        if os.path.exists(backup_file):
            print(f"⚠️ Không thấy 'books_with_emotions.csv', dùng tạm '{backup_file}'")
            df = pd.read_csv(backup_file)
        else:
            print("❌ LỖI: Không tìm thấy file csv dữ liệu sách!")
            return
    else:
        df = pd.read_csv(CSV_FILE)
    
    print(f"📖 Đã đọc {len(df)} cuốn sách.")

    # 2. XÓA DATABASE CŨ (Để tránh xung đột)
    if os.path.exists(CHROMA_DIR):
        print("🗑️ Đang xóa ChromaDB cũ lỗi...")
        try:
            shutil.rmtree(CHROMA_DIR)
        except:
            print("⚠️ Không thể xóa folder cũ, hãy thử xóa tay nếu code báo lỗi.")
    
    # 3. TẠO LẠI CHROMADB (Chuẩn định dạng ISBN + Mô tả)
    print("zzz Đang xây dựng lại Vector Database (Khoảng 1-2 phút)...")
    
    # Đảm bảo cột tagged_description tồn tại và xử lý NaN
    if "tagged_description" not in df.columns:
        # Nếu chưa có, tự tạo cột này: ISBN + Title + Description
        print("⚠️ Cột 'tagged_description' thiếu, đang tự tạo lại...")
        df["tagged_description"] = df["isbn13"].astype(str) + " " + df["title"] + " " + df["description"]
    
    df["tagged_description"] = df["tagged_description"].fillna("")
    
    # Tạo Documents cho ChromaDB
    documents = []
    for _, row in df.iterrows():
        content = str(row["tagged_description"])
        # Chỉ thêm nếu content hợp lệ
        if len(content) > 10: 
            documents.append(Document(page_content=content, metadata={"isbn": str(row["isbn13"])}))

    # Nạp vào ChromaDB
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Chia nhỏ batch để nạp cho nhẹ máy
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        Chroma.from_documents(batch, embedding_model, persist_directory=CHROMA_DIR)
        print(f"   -> Đã nạp {min(i+batch_size, len(documents))}/{len(documents)} sách...")
        
    print("✅ ChromaDB đã được xây mới hoàn toàn!")

    # 4. TẠO FILE RATINGS.CSV (Phủ kín 100% sách)
    print("📊 Đang sinh dữ liệu đánh giá giả lập (Collaborative Filtering)...")
    
    all_isbns = df['isbn13'].astype(str).str.replace(r'\.0$', '', regex=True).unique().tolist()
    
    user_ids = []
    book_isbns = []
    ratings = []

    # Tạo rating cho MỌI cuốn sách (mỗi sách ít nhất 2 đánh giá)
    for isbn in all_isbns:
        for u in range(1, 3): # User 1 và User 2 đọc hết sách
            user_ids.append(u)
            book_isbns.append(isbn)
            ratings.append(np.random.randint(3, 6)) # Rate 3-5 sao
            
    # Tạo thêm rating ngẫu nhiên
    for _ in range(2000):
        user_ids.append(np.random.randint(10, 500))
        book_isbns.append(np.random.choice(all_isbns))
        ratings.append(np.random.randint(1, 6))
        
    df_ratings = pd.DataFrame({'user_id': user_ids, 'isbn': book_isbns, 'rating': ratings})
    df_ratings.to_csv(RATINGS_FILE, index=False)
    print(f"✅ Đã tạo 'ratings.csv' với {len(df_ratings)} dòng.")

    # 5. TỰ KIỂM TRA (SELF-TEST)
    print("\n🔎 Đang chạy thử kiểm tra hệ thống...")
    try:
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
        results = db.similarity_search("Harry Potter", k=1)
        if results:
            content = results[0].page_content
            extracted_isbn = content.split()[0]
            print(f"   + Test tìm 'Harry Potter': Tìm thấy nội dung: {content[:50]}...")
            print(f"   + ISBN trích xuất được: '{extracted_isbn}'")
            if extracted_isbn.replace(".0", "").isdigit():
                print("   => ✅ KẾT QUẢ: Hệ thống hoạt động tốt!")
            else:
                print("   => ❌ CẢNH BÁO: ISBN trích xuất không phải số. Cần kiểm tra lại.")
        else:
            print("   => ❌ LỖI: Không tìm thấy sách nào trong DB mới tạo.")
    except Exception as e:
        print(f"   => ❌ Lỗi khi test: {e}")

    print("\n🎉 HOÀN TẤT! Bây giờ bạn hãy chạy lại file 'gradio-dashboard.py' nhé.")

if __name__ == "__main__":
    reset_data()