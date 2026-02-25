import streamlit as st
import joblib

st.set_page_config(page_title="Toxic Speech Detector")

@st.cache_resource
def load_models():
    model = joblib.load('toxic_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("⚠️ Lỗi: Không tìm thấy model. Vui lòng chạy file `train.py` trước để tạo model.")
    st.stop()

st.title("Trình Kiểm Duyệt Bình Luận")
st.markdown("Hệ thống phát hiện ngôn từ độc hại.")

user_input = st.text_area("Nhập bình luận cần kiểm tra:", "")

if st.button("Kiểm duyệt"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập văn bản để kiểm tra.")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        
        # Extract Top Terms
        feature_names = vectorizer.get_feature_names_out()
        dense_vec = vec.todense().tolist()[0]
        word_weights = [(feature_names[i], dense_vec[i]) for i in range(len(dense_vec)) if dense_vec[i] > 0]
        word_weights = sorted(word_weights, key=lambda x: x[1], reverse=True)
        top_words = [word for word, weight in word_weights[:3]] 

        st.markdown("---")
        if prediction == 1:
            st.error("🚩 **KẾT QUẢ: ĐỘC HẠI (TOXIC)**")
            if top_words:
                st.warning(f"**Giải thích (Top Terms):** Hệ thống đánh dấu dựa trên các từ khóa: {', '.join(top_words)}")
        else:
            st.success("✅ **KẾT QUẢ: AN TOÀN (SAFE)**")