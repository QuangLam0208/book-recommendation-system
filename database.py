import sqlite3

DB_NAME = "user_history.db"

def init_db():
    """Khởi tạo database và bảng nếu chưa tồn tại"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Tạo bảng lưu lịch sử: ID, Tên người dùng, Từ khóa tìm kiếm, Thời gian
    c.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            query TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized!")

def log_search(user_id, query):
    """Lưu từ khóa tìm kiếm vào database"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO search_history (user_id, query) VALUES (?, ?)", (user_id, query))
    conn.commit()
    conn.close()

def get_recent_interests(user_id, limit=3):
    """Lấy các từ khóa gần nhất để phân tích sở thích"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Lấy các query gần nhất, loại bỏ trùng lặp
    c.execute('''
        SELECT DISTINCT query 
        FROM search_history 
        WHERE user_id = ? 
        ORDER BY id DESC 
        LIMIT ?
    ''', (user_id, limit))
    rows = c.fetchall()
    conn.close()
    # Trả về danh sách các từ khóa, ví dụ: ['python', 'trinh thám', 'nấu ăn']
    return [row[0] for row in rows]