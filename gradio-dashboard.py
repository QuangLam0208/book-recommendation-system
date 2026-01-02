import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import sqlite3
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# --- CẤU HÌNH ĐƯỜNG DẪN (FIX LỖI WINDOWS/ONEDRIVE) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_abs_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- 1. DATABASE LỊCH SỬ ---
def init_db():
    try:
        conn = sqlite3.connect(get_abs_path('user_history.db'))
        c = conn.cursor()
        # Tạo bảng có thêm cột top_book
        c.execute('''CREATE TABLE IF NOT EXISTS search_history 
                     (id INTEGER PRIMARY KEY, user_id TEXT, query TEXT, top_book TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
    except: pass

def log_search(user_id, query, top_book_title):
    try:
        conn = sqlite3.connect(get_abs_path('user_history.db'))
        c = conn.cursor()
        if query.strip():
            # Lưu cả query và tên sách Top 1
            c.execute("INSERT INTO search_history (user_id, query, top_book) VALUES (?, ?, ?)", 
                      (user_id, query, top_book_title))
            conn.commit()
        conn.close()
    except Exception as e: print(f"Lỗi log: {e}")

def get_recent_interests(user_id, current_query="", limit=3):
    # Hàm này giữ nguyên để phục vụ gợi ý
    try:
        conn = sqlite3.connect(get_abs_path('user_history.db'))
        c = conn.cursor()
        c.execute("SELECT DISTINCT query FROM search_history WHERE user_id = ? AND query != ? ORDER BY id DESC LIMIT ?", (user_id, current_query, limit))
        rows = c.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except: return []

def get_history_logs(user_id="guest", limit=10):
    """Hàm lấy lịch sử SÁCH TOP 1 để hiển thị"""
    try:
        conn = sqlite3.connect(get_abs_path('user_history.db'))
        c = conn.cursor()
        # Lấy thời gian và Tên sách (top_book) thay vì query
        c.execute("SELECT strftime('%Y-%m-%d %H:%M:%S', timestamp), top_book FROM search_history WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
        rows = c.fetchall()
        conn.close()
        return rows 
    except: return []

# --- 2. HỆ THỐNG COLLABORATIVE FILTERING ---
cf_model = None
book_pivot = None
book_index_map = None 

def init_collaborative_filtering():
    global cf_model, book_pivot, book_index_map
    print("🔄 Đang khởi tạo Collaborative Filtering...")
    ratings_path = get_abs_path("ratings.csv")
    
    if not os.path.exists(ratings_path):
        print("⚠️ Không tìm thấy ratings.csv -> Bỏ qua CF.")
        return

    try:
        # Ép kiểu dữ liệu isbn thành string ngay từ đầu
        df_ratings = pd.read_csv(ratings_path, dtype={'isbn': str, 'user_id': str})
        
        # Xử lý sạch ISBN (bỏ .0 nếu có)
        df_ratings['isbn'] = df_ratings['isbn'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        book_pivot = df_ratings.pivot_table(index='isbn', columns='user_id', values='rating').fillna(0)
        book_index_map = {isbn: i for i, isbn in enumerate(book_pivot.index)}
        book_sparse = csr_matrix(book_pivot.values)
        cf_model = NearestNeighbors(metric='cosine', algorithm='brute')
        cf_model.fit(book_sparse)
        print(f"✅ CF Model OK! ({len(book_pivot)} sách trong hệ thống gợi ý)")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo CF: {e}")

def get_collaborative_recs(isbn, n_neighbors=6):
    if cf_model is None or book_index_map is None: return []
    
    # Chuẩn hóa input ISBN
    isbn = str(isbn).replace(".0", "").strip()
    
    if isbn not in book_index_map:
        # print(f"DEBUG: ISBN {isbn} không có trong dữ liệu ratings")
        return []
    try:
        query_index = book_index_map[isbn]
        distances, indices = cf_model.kneighbors(book_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors=n_neighbors)
        
        recs = []
        for i in range(1, len(distances.flatten())):
            idx = indices.flatten()[i]
            recs.append(book_pivot.index[idx])
        return recs
    except: return []

# --- 3. KHỞI ĐỘNG ---
print("🚀 Đang khởi động ứng dụng...")
init_db()
init_collaborative_filtering()
load_dotenv()

# LOAD SÁCH (CỰC KỲ QUAN TRỌNG: ÉP KIỂU STRING)
csv_path = get_abs_path("books_with_emotions.csv")
if not os.path.exists(csv_path): csv_path = get_abs_path("books_cleaned.csv")

try:
    # dtype={'isbn13': str} là chìa khóa để sửa lỗi tìm kiếm
    books = pd.read_csv(csv_path, dtype={'isbn13': str})
    
    # Chuẩn hóa cột ISBN một lần nữa cho chắc chắn
    if "isbn13" in books.columns:
        books["isbn13"] = books["isbn13"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    if "large_thumbnail" not in books.columns:
        books["large_thumbnail"] = books["thumbnail"]
        
    print(f"✅ Đã load {len(books)} cuốn sách vào bộ nhớ.")
except Exception as e:
    print(f"❌ LỖI KHÔNG ĐỌC ĐƯỢC FILE SÁCH: {e}")
    exit()

# LOAD VECTOR DB
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
PERSIST_DIR = get_abs_path("chroma_db")
db_books = None

if os.path.exists(PERSIST_DIR):
    try:
        db_books = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        # Test connection
        db_books.similarity_search("test", k=1)
        print("✅ Kết nối ChromaDB thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi kết nối DB: {e}")
        # Fallback RAM mode...
else:
    print("❌ Không tìm thấy thư mục chroma_db. Hãy chạy reset_all_data.py trước!")

# --- 4. LOGIC TÌM KIẾM (CÓ DEBUG LOG) ---
def retrieve_semantic_recommendations(query, category="All", tone="All", top_k=100):
    if not db_books: return pd.DataFrame()
    
    print(f"\n🔎 [DEBUG] Đang tìm kiếm: '{query}'")
    
    # 1. Tìm trong Vector DB
    try:
        recs = db_books.similarity_search(query, k=top_k)
        print(f"   -> Tìm thấy {len(recs)} vector tương đồng.")
    except Exception as e:
        print(f"   -> Lỗi vector search: {e}")
        return pd.DataFrame()

    # 2. Trích xuất ISBN
    isbn_list = []
    for i, rec in enumerate(recs):
        # Ưu tiên lấy từ metadata (do reset_all_data.py tạo ra)
        val = rec.metadata.get("isbn")
        
        # Nếu không có metadata, thử lấy từ nội dung (fallback cũ)
        if not val:
            content_parts = rec.page_content.split()
            if content_parts: val = content_parts[0]
            
        # Làm sạch chuỗi ISBN
        if val:
            val = str(val).replace(".0", "").strip()
            if val.isdigit(): 
                isbn_list.append(val)
    
    # In ra vài ISBN đầu tiên để kiểm tra
    if isbn_list:
        print(f"   -> Trích xuất được {len(isbn_list)} ISBN hợp lệ. Ví dụ: {isbn_list[:3]}")
    else:
        print("   -> ⚠️ CẢNH BÁO: Không trích xuất được ISBN nào từ kết quả vector!")
        return pd.DataFrame()

    # 3. Đối chiếu với DataFrame Books
    # Lọc những ISBN có tồn tại trong file CSV
    book_recs = books[books["isbn13"].isin(isbn_list)].copy()
    print(f"   -> Khớp được {len(book_recs)} cuốn sách trong file CSV.")

    # Sắp xếp theo đúng thứ tự tìm kiếm (quan trọng)
    if not book_recs.empty:
        book_recs = book_recs.set_index("isbn13")
        # Chỉ giữ lại những isbn có trong list tìm kiếm và sắp xếp theo thứ tự đó
        valid_isbns = [i for i in isbn_list if i in book_recs.index]
        book_recs = book_recs.reindex(valid_isbns).reset_index()

    # 4. Lọc Category
    if category != "All" and "simple_categories" in book_recs.columns:
        original_count = len(book_recs)
        book_recs = book_recs[book_recs["simple_categories"] == category]
        print(f"   -> Sau khi lọc Category '{category}': còn {len(book_recs)}/{original_count} cuốn.")

    # 5. Lọc Tone
    if tone != "All":
        tone_map = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
        col = tone_map.get(tone)
        if col and col in book_recs.columns:
            book_recs = book_recs.sort_values(by=col, ascending=False)
            print(f"   -> Đã sắp xếp lại theo cảm xúc '{tone}'.")

    return book_recs

def format_results(df):
    results = []
    if df.empty: return results
    for _, row in df.iterrows():
        title = str(row['title'])
        authors = str(row['authors']) if pd.notna(row['authors']) else "Unknown"
        # Xử lý mô tả ngắn
        desc = str(row.get('description', ''))
        trunc_desc = " ".join(desc.split()[:20]) + "..."
        
        caption = f"{title}\nby {authors}\n\n{trunc_desc}"
        img = row["large_thumbnail"] if pd.notna(row["large_thumbnail"]) else "cover-not-found.jpg"
        results.append((img, caption))
    return results

def recommend_books(query, category, tone):
    # 1. Content-Based Search (Tìm kiếm nội dung trước)
    content_df = retrieve_semantic_recommendations(query, category, tone)
    current_results = format_results(content_df)
    
    top_book_log = "Không tìm thấy"
    if not content_df.empty:
        # Lấy tiêu đề cuốn sách đầu tiên tìm thấy
        top_book_log = str(content_df.iloc[0]['title'])
    
    user_id = "guest"
    # Gọi hàm log với ĐỦ 3 THAM SỐ
    log_search(user_id, query, top_book_log)

    # 2. Collaborative Filtering (Gợi ý từ cộng đồng)
    secondary_results = []
    msg = ""
    
    if not content_df.empty:
        top_isbn = str(content_df.iloc[0]['isbn13'])
        top_title = str(content_df.iloc[0]['title'])
        
        print(f"🔗 [CF] Đang tìm sách liên quan đến: {top_title} ({top_isbn})")
        cf_isbns = get_collaborative_recs(top_isbn)
        
        if cf_isbns:
            cf_df = books[books['isbn13'].isin(cf_isbns)]
            if not cf_df.empty:
                secondary_results = format_results(cf_df)
                msg = f"Vì bạn quan tâm '{top_title}' (Cộng đồng cũng đọc)"

    # 3. Fallback History / Random (Nếu không tìm thấy gì)
    if not secondary_results:
        print("⚠️ Fallback: Dùng lịch sử hoặc Random.")
        
        recent = get_recent_interests(user_id, query)
        if recent:
            hist_query = " ".join(recent)
            hist_df = retrieve_semantic_recommendations(hist_query, top_k=50)
            if not hist_df.empty:
                secondary_results = format_results(hist_df.sample(frac=1).head(8))
                msg = "Dựa trên lịch sử tìm kiếm gần đây"
    
    if not secondary_results:
         secondary_results = format_results(books.sample(8))
         msg = "Có thể bạn sẽ thích (Ngẫu nhiên)"

    # 4. Lấy lại lịch sử mới nhất để cập nhật UI
    updated_history = get_history_logs(user_id)

    return current_results, secondary_results, msg, updated_history

# --- 5. UI ---
categories = ["All"]
if "simple_categories" in books.columns:
    categories += sorted(books["simple_categories"].dropna().unique().tolist())

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# AI Book Recommender (Hybrid System)")
    gr.Markdown("Tìm kiếm thông minh + Gợi ý cộng đồng")
    
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Bạn muốn tìm sách gì?", placeholder="Ví dụ: Harry Potter, magic, history...")
            cat = gr.Dropdown(categories, label="Thể loại", value="All")
            tone = gr.Dropdown(["All", "Happy", "Sad", "Suspenseful", "Surprising", "Angry"], label="Cảm xúc", value="All")
            btn = gr.Button("Tìm kiếm", variant="primary")
            
            # --- MỚI: Bảng hiển thị lịch sử ---
            gr.Markdown("### Sách vừa tìm được")
            history_table = gr.Dataframe(
                headers=["Thời gian", "Sách Top 1 Đề xuất"],  # Đổi tên cột
                datatype=["str", "str"],
                value=get_history_logs(), 
                interactive=False
            )
            
        with gr.Column(scale=3):
            out1 = gr.Gallery(label="Kết quả tìm kiếm", columns=5, height=450, object_fit="contain")
            lbl = gr.Markdown("### Gợi ý bổ sung")
            out2 = gr.Gallery(label="Gợi ý bổ sung", columns=5, height=300, object_fit="contain")

    # Cập nhật sự kiện click: Thêm history_table vào danh sách outputs
    btn.click(recommend_books, [inp, cat, tone], [out1, out2, lbl, history_table])

if __name__ == "__main__":
    print("🌐 App đang chạy tại: http://127.0.0.1:7860")
    dashboard.launch()