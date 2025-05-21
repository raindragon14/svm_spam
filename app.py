# app.py
import streamlit as st
import joblib
import os
import re
import string
import nltk

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Spam Email Detector", layout="wide", initial_sidebar_state="expanded")

# --- Function to Check and Download NLTK Resources ---
@st.cache_data # Cache the fact that downloads have been attempted/completed
def download_nltk_resources():
    resources_to_check = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4" # Open Multilingual Wordnet, often a WordNetLemmatizer dependency
    }
    download_needed = False
    for resource_name, resource_path in resources_to_check.items():
        try:
            nltk.data.find(resource_path)
        except nltk.downloader.DownloadError: # More specific error for NLTK downloads
            st.sidebar.info(f"NLTK resource '{resource_name}' not found. Attempting download.")
            nltk.download(resource_name, quiet=True)
            download_needed = True
        except LookupError: # General lookup error
            st.sidebar.info(f"NLTK resource '{resource_name}' not found (LookupError). Attempting download.")
            nltk.download(resource_name, quiet=True)
            download_needed = True
        except Exception as e: # Catch any other exception during find
            st.sidebar.warning(f"Could not verify NLTK resource '{resource_name}': {e}. Attempting download anyway.")
            try:
                nltk.download(resource_name, quiet=True)
                download_needed = True
            except Exception as download_e:
                st.sidebar.error(f"Failed to download '{resource_name}': {download_e}")
                return False # Indicate failure

    if download_needed:
        st.sidebar.success("NLTK resources download attempt complete. Please refresh if it's the first run.")
    else:
        st.sidebar.success("NLTK resources appear to be available.")
    return True # Indicate success or that no download was needed

# Call the download function once
NLTK_READY = download_nltk_resources()

# --- Initialize NLTK components AFTER ensuring resources are available ---
if NLTK_READY:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    stop_words_english = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
else:
    st.error("Critical NLTK resources could not be initialized. App functionality will be severely limited.")
    # Define dummy fallbacks to prevent crashes, though results will be poor
    stop_words_english = set()
    class DummyLemmatizer:
        def lemmatize(self, word, pos='n'): return word # Added pos for compatibility
    wordnet_lemmatizer = DummyLemmatizer()
    def word_tokenize(text): return text.split()


# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    model_path = "svm_model.joblib"
    vectorizer_path = "tfidf_vectorizer.joblib"

    # Check if files exist before attempting to load
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
        st.error(f"Error loading model/vectorizer: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer()


# --- Text Preprocessing Function ---
def preprocess_text(text):
    if not NLTK_READY: # If NLTK didn't initialize, use basic split
        return text.lower() if isinstance(text, str) else ""

    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' <NUM> ', text) # Replace numbers
    
    tokens = word_tokenize(text) # Uses the globally defined word_tokenize
    
    tokens = [word for word in tokens if word not in stop_words_english and len(word) > 1]
    # Lemmatize each word. For WordNetLemmatizer, providing POS tag can improve results,
    # but for simplicity, we'll lemmatize without POS tagging here.
    tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# --- Streamlit App UI ---
st.title("📧 Spam Email Detector")
st.markdown("""
    Welcome to the Spam Email Detector!
    Enter an email text in the box below and click "Classify Email" to see if it's predicted as Spam or Ham (Not Spam).
    This application uses a Support Vector Machine (SVM) model.
""")
st.write("---")

email_text_input = st.text_area(
    "Enter Email Text Here:",
    height=250,
    placeholder="Dear User, congratulations! You've won a prize..."
)

if st.button("🔎 Classify Email", type="primary"):
    if not NLTK_READY:
        st.error("NLTK resources are not properly initialized. Cannot perform classification.")
    elif model is None or vectorizer is None:
        st.warning("The model or vectorizer could not be loaded. Please check the application setup and logs.")
    elif not email_text_input.strip():
        st.warning("⚠️ Please enter some email text to classify.")
    else:
        with st.spinner("🔍 Analyzing email..."):
            cleaned_text = preprocess_text(email_text_input)

            if not cleaned_text.strip() and email_text_input.strip():
                 st.info("The input text resulted in no meaningful content after preprocessing (e.g., only stopwords, punctuation, or numbers).")

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
                st.write("**Cleaned (Preprocessed) Text:**")
                st.text(cleaned_text if cleaned_text.strip() else "No meaningful text after cleaning.")
                st.write("**Prediction Probabilities:**")
                st.write(f"Ham (Class 0): {prediction_proba[0][0]:.4f}")
                st.write(f"Spam (Class 1): {prediction_proba[0][1]:.4f}")
else:
    st.info("☝️ Enter email text above and click the 'Classify Email' button.")

# Sidebar
st.sidebar.header("About This App")
st.sidebar.info(
    "This Spam Email Detector is a demonstration of a machine learning model "
    "in action. It uses an SVM classifier trained on text features (TF-IDF) "
    "extracted from a dataset of emails."
)
st.sidebar.markdown("---")
st.sidebar.subheader("How it works:")
st.sidebar.markdown("""
- You provide the email text.
- The text is preprocessed (lowercase, remove punctuation & numbers, remove stopwords, lemmatize).
- The cleaned text is converted into numerical features using a pre-trained TF-IDF vectorizer.
- A pre-trained SVM model predicts whether the email is Spam or Ham.
""")
# GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY"
# st.sidebar.markdown(f"[View Source Code on GitHub]({GITHUB_REPO_URL})")

st.markdown("---")
st.caption("Built with Streamlit by [Your Name/Handle Here]")
