# 📚 AI Hybrid Book Recommender System

> **Hệ thống gợi ý sách thông minh kết hợp giữa Tìm kiếm ngữ nghĩa (Semantic Search) và Lọc cộng tác (Collaborative Filtering).**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![LangChain](https://img.shields.io/badge/AI-LangChain-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-purple)

## 📖 Giới thiệu

Dự án này xây dựng một hệ thống gợi ý sách "lai" (Hybrid Recommender System), giải quyết vấn đề tìm kiếm sách không chỉ dựa trên từ khóa chính xác mà còn dựa trên **ngữ nghĩa** và **cảm xúc**. Đồng thời, hệ thống tích hợp thuật toán gợi ý dựa trên cộng đồng để đề xuất những cuốn sách liên quan mà người khác cũng thích.

### ✨ Các tính năng chính

* **🔍 Tìm kiếm theo ngữ nghĩa (Semantic Search):** Tìm sách bằng ngôn ngữ tự nhiên (ví dụ: "Sách về cậu bé phù thủy" sẽ tìm ra "Harry Potter").
* **🤝 Gợi ý lai (Hybrid Recommendation):** Kết hợp kết quả từ Vector Database (nội dung) và thuật toán KNN (hành vi cộng đồng).
* **🎭 Lọc theo Cảm xúc & Thể loại:** Cho phép lọc sách theo tone cảm xúc (Vui, Buồn, Hồi hộp, Bất ngờ...) và thể loại.
* **🕒 Lịch sử tìm kiếm thông minh:** Tự động lưu và hiển thị lại các cuốn sách Top 1 mà bạn đã tìm thấy trước đó.
* **🎨 Giao diện trực quan:** Xây dựng trên Gradio với hiển thị bìa sách dạng Gallery.

---

## 🛠️ Kiến trúc hệ thống

Hệ thống hoạt động dựa trên sự phối hợp của các thành phần sau:

1. **Xử lý dữ liệu (ETL):**

   * Dữ liệu sách (`books_with_emotions.csv`) được làm sạch và gắn thẻ cảm xúc.
   * Tạo dữ liệu giả lập đánh giá (`ratings.csv`) để phục vụ thuật toán Collaborative Filtering.
2. **Vector Database (ChromaDB):**

   * Sử dụng mô hình Embedding `all-MiniLM-L6-v2` (thông qua HuggingFace) để chuyển đổi mô tả sách thành vector.
   * Lưu trữ và truy xuất nhanh các sách có nội dung tương đồng.
3. **Recommender Engine:**

   * **Content-Based:** Truy vấn ChromaDB để tìm sách có nội dung khớp với câu query.
   * **Collaborative Filtering:** Sử dụng thuật toán `NearestNeighbors` (KNN) trên ma trận User-Item để tìm sách liên quan.
4. **Database Lịch sử (SQLite):**

   * Lưu trữ log tìm kiếm và kết quả trả về.

---

## ⚙️ Cài đặt

### 1. Yêu cầu tiên quyết

* Python 3.9 trở lên.
* Các thư viện cần thiết.

### 2. Cài đặt thư viện

```bash
pip install pandas numpy gradio langchain-huggingface langchain-chroma langchain-community scikit-learn scipy python-dotenv
```

### 3. Cấu trúc thư mục

```text
├── books_with_emotions.csv
├── reset_all_data.py
├── gradio-dashboard.py
├── user_history.db
├── chroma_db/
└── ratings.csv
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Khởi tạo dữ liệu

```bash
python reset_all_data.py
```

### Bước 2: Chạy ứng dụng

```bash
python gradio-dashboard.py
```

### Bước 3: Trải nghiệm

* Mở địa chỉ hiển thị trong terminal (thường là `http://127.0.0.1:7860`)
* Nhập từ khóa tìm kiếm
* Chọn bộ lọc nếu cần
* Xem kết quả và gợi ý

---

## 📝 Nhật ký thay đổi

* v1.0: Semantic Search cơ bản
* v1.1: Thêm Collaborative Filtering
* v1.2: Thêm Emotion Filter
* v1.3: Thêm Search History

---

## 🤝 Đóng góp

Dự án được phát triển bởi **t-redactyl** và **QuangLam0208**.

---

### 📄 File `requirements.txt` 

```text
pandas
numpy
gradio
langchain-huggingface
langchain-chroma
langchain-community
scikit-learn
scipy
python-dotenv
```

---
