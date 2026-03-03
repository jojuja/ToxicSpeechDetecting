
* 📊 Kết quả Baseline vs PhoBERT
* ⚠️ Giải thích vì sao không upload PhoBERT
* 📦 Thông tin dataset
* ☁️ Ghi rõ train bằng Google Colab

---

# 📊 Experimental Results

## Dataset Information

* Dataset: **ViHSD (UIT NLP)**
* Tổng số mẫu: 33,400
* Tập Test: 6,680 mẫu

  * Safe: 5,548
  * Toxic: 1,132
* Nhãn được gộp:

  * 0 → Safe
  * 1,2 → Toxic

Kích thước dataset có thể kiểm tra bằng:

```python
len(dataset["train"])
len(dataset["test"])
```

---

# 🧠 Baseline Model (TF-IDF + Logistic Regression)

**Accuracy: 0.88**

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| Safe  | 0.89      | 0.98   | 0.93     | 5548    |
| Toxic | 0.83      | 0.40   | 0.54     | 1132    |

Macro F1-score: **0.74**
Weighted F1-score: **0.87**

### Nhận xét:

* Baseline có Recall rất cao với lớp Safe.
* Tuy nhiên Recall của Toxic chỉ 0.40 → bỏ sót nhiều bình luận độc hại.
* Phù hợp làm mô hình nền để so sánh.

---

# 🧠 PhoBERT Model (Transformers + PyTorch)

Huấn luyện bằng **Google Colab (GPU environment)**.

**Accuracy: 0.89**

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| Safe  | 0.92      | 0.95   | 0.93     | 5548    |
| Toxic | 0.70      | 0.59   | 0.64     | 1132    |

Macro F1-score: **0.79**
Weighted F1-score: **0.88**

### Nhận xét:

* PhoBERT cải thiện đáng kể Recall của lớp Toxic (0.59 so với 0.40).
* Macro F1-score cao hơn Baseline.
* Khả năng hiểu ngữ cảnh tốt hơn TF-IDF.

---

# 📈 So sánh nhanh

| Metric       | Baseline | PhoBERT |
| ------------ | -------- | ------- |
| Accuracy     | 0.88     | 0.89    |
| Toxic Recall | 0.40     | 0.59    |
| Macro F1     | 0.74     | 0.79    |

PhoBERT cho hiệu suất tốt hơn đặc biệt ở lớp Toxic.

---

# ⚠️ Lưu ý về PhoBERT Model

Thư mục PhoBERT không được upload lên GitHub do kích thước lớn (~460MB).

GitHub giới hạn:

* Tối đa 100MB / file

Do đó, repository này chỉ chứa:

* Code huấn luyện
* Kết quả thực nghiệm
* Mô hình Baseline (.pkl)

PhoBERT được huấn luyện trên Google Colab và chỉ sử dụng để đánh giá và so sánh trong báo cáo.

---

# 🚀 Khi pull repository này

Người dùng vẫn có thể chạy hệ thống bằng Baseline model:

```
pip install -r requirements.txt
python -m streamlit run app.py
```

Website sẽ hoạt động bình thường với mô hình TF-IDF + Logistic Regression.

---

# 🖥 Training Environment

* Baseline: Train local (CPU)
* PhoBERT: Train trên Google Colab (GPU)
* Framework: HuggingFace Transformers + PyTorch

---

