import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("1. Downloading ViHSD dataset...")
# PASTE YOUR TOKEN BELOW:
dataset = load_dataset("uitnlp/vihsd", token="your_token")

print("2. Cleaning data...")
X_train_raw = dataset['train']['free_text']
y_train_raw = dataset['train']['label_id']
X_test_raw = dataset['test']['free_text']
y_test_raw = dataset['test']['label_id']

# Handle empty rows
X_train = [str(text) if text is not None else "" for text in X_train_raw]
X_test = [str(text) if text is not None else "" for text in X_test_raw]

# 0 = Safe, 1 = Toxic
y_train = [1 if label > 0 else 0 for label in y_train_raw]
y_test = [1 if label > 0 else 0 for label in y_test_raw]

print("3. Training Baseline Model (TF-IDF + Logistic Regression)...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("\n--- Accuracy Report ---")
predictions = model.predict(X_test_tfidf)
print(classification_report(y_test, predictions, target_names=["Safe", "Toxic"]))

print("\n4. Saving model files...")
joblib.dump(model, 'toxic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("✅ Done! 'toxic_model.pkl' and 'tfidf_vectorizer.pkl' are ready for the web app.")