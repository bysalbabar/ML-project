import streamlit as st
import pickle
from PyPDF2 import PdfReader

# Load model and vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# PDF to text function
def pdf_to_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.title("📄 Resume Quality Prediction")

uploaded_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")

if uploaded_file is not None:
    resume_text = pdf_to_text(uploaded_file)

    if st.button("Predict Resume Quality"):
        X_new = vectorizer.transform([resume_text])
        prediction = model.predict(X_new)[0]

        label_map = {0: "Poor", 1: "Average", 2: "Good"}
        st.success(f"Predicted Resume Quality: **{label_map[prediction]}**")
