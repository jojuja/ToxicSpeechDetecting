### 1. `train.py` (Script Huấn luyện Mô hình)
Đây là "trái tim" của quá trình học máy. File này chỉ cần chạy 1 lần để tạo ra mô hình.
* **Tải dữ liệu:** Sử dụng thư viện `datasets` để tải bộ dữ liệu chuẩn `uitnlp/vihsd` (hơn 30,000 bình luận tiếng Việt) từ Hugging Face.
* **Tiền xử lý:** Làm sạch dữ liệu, xử lý các dòng trống (None) và gộp nhãn (Offensive/Hate gộp chung thành 1 nhãn Toxic).
* **Trích xuất đặc trưng (Feature Extraction):** Sử dụng `TfidfVectorizer` để chuyển đổi văn bản tiếng Việt thành các ma trận số học (giới hạn 10,000 từ phổ biến nhất).
* **Huấn luyện (Training):** Sử dụng thuật toán `LogisticRegression` để học cách phân loại từ các ma trận số học đó.
* **Lưu mô hình:** Sử dụng thư viện `joblib` để xuất (dump) mô hình và bộ từ điển ra thành 2 file `.pkl` để web app có thể sử dụng lại mà không cần train lại.

### 2. `app.py` (Giao diện Web / MVP Dashboard)
Đây là file chạy giao diện người dùng bằng thư viện Streamlit.
* **Load Models:** Đọc 2 file `.pkl` đã được tạo ra từ `train.py`.
* **Giao diện:** Tạo ô nhập văn bản (Text Area) cho người dùng nhập bình luận.
* **Dự đoán:** Khi bấm nút "Kiểm duyệt", hệ thống vector hóa câu bình luận và đưa vào mô hình để dự đoán (0 = An toàn, 1 = Độc hại).
* **Explainable AI (Top Terms):** Trích xuất các từ có trọng số TF-IDF cao nhất trong câu bình luận để giải thích cho người dùng biết từ ngữ nào khiến hệ thống đánh dấu đây là câu độc hại.

### 3. `toxic_model.pkl` & `tfidf_vectorizer.pkl` (Trọng số Mô hình)
* **`toxic_model.pkl`**: Là "bộ não" đã được huấn luyện, chứa các quy tắc toán học để quyết định một câu là an toàn hay độc hại.
* **`tfidf_vectorizer.pkl`**: Là "bộ từ điển" lưu trữ danh sách các từ vựng và cách chuyển chúng thành số liệu. 

### 4. `requirements.txt` (Danh sách Môi trường)
Chứa danh sách các thư viện Python (`streamlit`, `scikit-learn`, `datasets`, `joblib`) để đảm bảo code chạy đồng nhất trên mọi máy tính.

## 🚀 Hướng dẫn cài đặt và chạy Website 

Để chạy giao diện web và test thử các câu bình luận phục vụ báo cáo/thuyết trình, các thành viên làm theo đúng các bước sau:

**Bước 1: Mở Terminal tại thư mục dự án**
1. Giải nén folder chứa code.
2. Click chuột vào thanh địa chỉ của folder, gõ `cmd` và nhấn **Enter** để mở Terminal.

**Bước 2: Cài đặt thư viện (Chỉ làm 1 lần)**
Copy dòng lệnh dưới đây dán vào Terminal và nhấn **Enter**:
```bash
pip install -r requirements.txt

Bước 3: Khởi chạy Website
Sau khi cài xong, copy lệnh này dán vào Terminal và nhấn Enter:
Bash

python -m streamlit run app.py