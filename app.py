import streamlit as st
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- NLTK Resource Download ---
# It's good practice to ensure NLTK resources are available,
# especially when deploying.
try:
    stopwords.words('english')
    word_tokenize("test")
    WordNetLemmatizer().lemmatize("tests")
except LookupError:
    st.info("Downloading NLTK resources for the first time...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True) # Needed by WordNetLemmatizer for some languages
    st.success("NLTK resources downloaded.")


# --- Load Model and Vectorizer ---
# Use st.cache_resource to load these only once and speed up the app
@st.cache_resource
def load_model_and_vectorizer():
    # Adjust paths if your artifacts are in a subdirectory or root
    model_path = "svm_spam_model.joblib"
    vectorizer_path = "tfidf_vectorizer.joblib"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model or Vectorizer file not found. Please check paths.")
        return None, None
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer()

# --- Text Preprocessing Function (MUST BE THE SAME AS USED IN TRAINING) ---
stop_words_english = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' <NUM> ', text) # Or however you handled numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words_english and len(word) > 1]
    tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# --- Streamlit App UI ---
st.set_page_config(page_title="Spam Email Detector", layout="wide")
st.title("📧 Spam Email Detector")
st.markdown("""
    Enter an email text below to classify it as **Spam** or **Ham** (Not Spam).
    This app uses a Support Vector Machine (SVM) model trained on TF-IDF features.
""")

# Input text area
email_text_input = st.text_area("Enter Email Text Here:", height=200, placeholder="Type or paste your email content...")

# Prediction button
if st.button("🔎 Classify Email"):
    if model is None or vectorizer is None:
        st.warning("Model/Vectorizer not loaded. Cannot classify.")
    elif not email_text_input.strip():
        st.warning("Please enter some email text to classify.")
    else:
        with st.spinner("Classifying..."):
            # 1. Preprocess the input text
            cleaned_text = preprocess_text(email_text_input)

            # 2. Vectorize the preprocessed text using the loaded TF-IDF vectorizer
            # Important: vectorizer.transform expects an iterable (like a list of strings)
            vectorized_text = vectorizer.transform([cleaned_text]) # Pass as a list

            # 3. Make prediction using the loaded model
            prediction = model.predict(vectorized_text)
            prediction_proba = model.predict_proba(vectorized_text) # Get probabilities

            # Display result
            st.subheader("Classification Result:")
            if prediction[0] == 1: # Assuming 1 is Spam
                st.error("🚨 This email is classified as: **Spam**")
                st.write(f"Confidence (Spam): {prediction_proba[0][1]*100:.2f}%")
            else: # Assuming 0 is Ham
                st.success("✅ This email is classified as: **Ham (Not Spam)**")
                st.write(f"Confidence (Ham): {prediction_proba[0][0]*100:.2f}%")

            with st.expander("Show Processed Text and Probabilities"):
                st.write("Cleaned (Preprocessed) Text:", cleaned_text if cleaned_text else "No meaningful text after cleaning.")
                st.write("Prediction Probabilities (Ham, Spam):", prediction_proba[0])
else:
    st.info("Enter email text and click 'Classify Email'.")

st.sidebar.header("About")
st.sidebar.info(
    "This is a demo Spam Email Detector built with Streamlit. "
    "It uses an SVM model trained on a public dataset."
    # Add your GitHub link or name if you like
)
