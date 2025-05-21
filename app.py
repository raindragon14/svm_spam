# app.py
import streamlit as st
import joblib
import os
import re
import string # Standard library, no NLTK needed

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Spam Email Detector (No NLTK)", layout="wide", initial_sidebar_state="expanded")

# --- Define NLTK-Free Preprocessing Function ---
# (This MUST match the one used for retraining your model and vectorizer)
MY_STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 've', 'll', 're'
])

def preprocess_text_no_nltk(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' <NUM> ', text) # Or remove numbers: re.sub(r'\d+', '', text)
    tokens = text.split() # Simple whitespace tokenization
    tokens = [word for word in tokens if word not in MY_STOP_WORDS and len(word) > 1]
    # Stemming/Lemmatization is skipped here for simplicity
    return " ".join(tokens)

# --- Load Model and Vectorizer (the NEW NLTK-free versions) ---
@st.cache_resource
def load_model_and_vectorizer_no_nltk():
    # IMPORTANT: Use the filenames of your NEWLY saved model and vectorizer
    model_path = "svm_model.joblib"
    vectorizer_path = "tfidf_vectorizer.joblib"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {os.path.abspath(model_path)}")
        return None, None
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer file not found: {os.path.abspath(vectorizer_path)}")
        return None, None
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading NLTK-free model/vectorizer: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer_no_nltk()

# --- Streamlit App UI ---
st.title("📧 Spam Email Detector (NLTK-Free)")
st.markdown("""
    Welcome! Enter an email text below to classify it as **Spam** or **Ham**.
    This version uses basic Python string operations for preprocessing, without NLTK.
""")
st.write("---")

email_text_input = st.text_area(
    "Enter Email Text Here:",
    height=250,
    placeholder="Your email content..."
)

if st.button("🔎 Classify Email", type="primary"):
    if model is None or vectorizer is None:
        st.warning("The model or vectorizer (NLTK-free versions) could not be loaded.")
    elif not email_text_input.strip():
        st.warning("⚠️ Please enter some email text to classify.")
    else:
        with st.spinner("🔍 Analyzing email..."):
            # 1. Preprocess using the NLTK-free function
            cleaned_text = preprocess_text_no_nltk(email_text_input)

            if not cleaned_text.strip() and email_text_input.strip():
                 st.info("The input text resulted in no meaningful content after preprocessing.")

            try:
                vectorized_text = vectorizer.transform([cleaned_text])
            except Exception as e:
                st.error(f"Error during text vectorization: {e}")
                st.stop()

            try:
                prediction = model.predict(vectorized_text)
                prediction_proba = model.predict_proba(vectorized_text)
            except Exception as e:
                st.error(f"Error during model prediction: {e}")
                st.stop()

            st.subheader("✉️ Classification Result:")
            if prediction[0] == 1: # Assuming 1 is Spam
                st.error("🚨 This email is classified as: **Spam**")
                spam_probability = prediction_proba[0][1] * 100
                st.progress(int(spam_probability))
                st.markdown(f"Confidence (Spam): **{spam_probability:.2f}%**")
            else: # Assuming 0 is Ham
                st.success("✅ This email is classified as: **Ham (Not Spam)**")
                ham_probability = prediction_proba[0][0] * 100
                st.progress(int(ham_probability))
                st.markdown(f"Confidence (Ham): **{ham_probability:.2f}%**")

            with st.expander("🔬 Show Details"):
                st.write("**Original Input (Snippet):**")
                st.text(email_text_input[:500] + "..." if len(email_text_input) > 500 else email_text_input)
                st.write("**Cleaned (Preprocessed) Text (NLTK-Free):**")
                st.text(cleaned_text if cleaned_text.strip() else "No meaningful text after cleaning.")
                st.write("**Prediction Probabilities:**")
                st.write(f"Ham (Class 0): {prediction_proba[0][0]:.4f}")
                st.write(f"Spam (Class 1): {prediction_proba[0][1]:.4f}")
else:
    st.info("☝️ Enter email text above and click the 'Classify Email' button.")

# Sidebar
st.sidebar.header("About This App")
st.sidebar.info(
    "This Spam Email Detector uses an SVM classifier. This version performs "
    "text preprocessing without relying on the NLTK library."
)
st.sidebar.markdown("---")
st.sidebar.subheader("How it works (NLTK-Free):")
st.sidebar.markdown("""
- You provide the email text.
- The text is preprocessed (lowercase, remove punctuation & numbers, remove custom stopwords).
- The cleaned text is converted into numerical features using a pre-trained TF-IDF vectorizer.
- A pre-trained SVM model predicts whether the email is Spam or Ham.
""")
# GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY"
# st.sidebar.markdown(f"[View Source Code on GitHub]({GITHUB_REPO_URL})")

st.markdown("---")
st.caption("Built with Streamlit by [Your Name/Handle Here]")
