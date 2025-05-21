# app.py
import streamlit as st
import joblib
import os
import re
import string
import nltk # Import base NLTK first

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Spam Email Detector", layout="wide", initial_sidebar_state="expanded")

# --- NLTK Setup Function ---
@st.cache_resource # Cache the setup process so it runs effectively once per session
def setup_nltk():
    """
    Checks for NLTK resources and downloads them if missing.
    Initializes global NLTK components.
    Returns True if setup is successful, False otherwise.
    """
    st.sidebar.info("Initializing NLTK resources...")
    # Print NLTK's default search paths for debugging
    st.sidebar.caption(f"NLTK search paths: {nltk.data.path}")

    resources_to_ensure = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4"
    }
    all_resources_found_or_downloaded = True

    for resource_name, resource_internal_path in resources_to_ensure.items():
        try:
            nltk.data.find(resource_internal_path)
            st.sidebar.text(f"NLTK resource '{resource_name}' found.")
        except LookupError:
            st.sidebar.warning(f"NLTK resource '{resource_name}' not found. Attempting download...")
            try:
                nltk.download(resource_name, quiet=False) # quiet=False for more log output
                # Verify after download
                nltk.data.find(resource_internal_path)
                st.sidebar.success(f"NLTK resource '{resource_name}' downloaded successfully.")
            except Exception as e:
                st.sidebar.error(f"Failed to download or verify NLTK resource '{resource_name}': {e}")
                all_resources_found_or_downloaded = False
        except Exception as e: # Catch other errors during find
            st.sidebar.error(f"Unexpected error checking NLTK resource '{resource_name}': {e}")
            all_resources_found_or_downloaded = False
            
    if all_resources_found_or_downloaded:
        st.sidebar.success("All necessary NLTK resources are available.")
        # Initialize global NLTK components here, after successful resource check/download
        global stop_words_english, wordnet_lemmatizer, word_tokenize_nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize as nltk_word_tokenize_func

        stop_words_english = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        word_tokenize_nltk = nltk_word_tokenize_func # Assign to a global variable
        return True
    else:
        st.sidebar.error("Failed to set up all NLTK resources. Preprocessing will be basic.")
        return False

# Run NLTK setup
NLTK_INITIALIZED_SUCCESSFULLY = setup_nltk()

# Define fallbacks if NLTK setup failed, to prevent app from crashing
if not NLTK_INITIALIZED_SUCCESSFULLY:
    st.error("NLTK initialization failed. Using basic text processing.")
    stop_words_english = set() # Empty set
    class DummyLemmatizer:
        def lemmatize(self, word, pos='n'): return word
    wordnet_lemmatizer = DummyLemmatizer()
    def word_tokenize_nltk(text): # Fallback tokenizer
        return text.split()


# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    # IMPORTANT: Use the model and vectorizer trained WITH NLTK preprocessing
    model_path = "svm_model.joblib" # Your original NLTK-trained model
    vectorizer_path = "tfidf_vectorizer.joblib" # Your original NLTK-trained vectorizer

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


# --- Text Preprocessing Function (using NLTK components if available) ---
def preprocess_text_with_nltk(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' <NUM> ', text) # Replace numbers
    
    # Use the globally initialized (or dummy) NLTK components
    tokens = word_tokenize_nltk(text)
    
    tokens = [word for word in tokens if word not in stop_words_english and len(word) > 1]
    tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens] # Simple lemmatization
    return " ".join(tokens)

# --- Streamlit App UI ---
st.title("📧 Spam Email Detector (with NLTK)")
st.markdown("""
    Welcome! Enter an email text below to classify it as **Spam** or **Ham**.
    This version uses NLTK for text preprocessing.
""")
st.write("---")

email_text_input = st.text_area(
    "Enter Email Text Here:",
    height=250,
    placeholder="Dear User, congratulations! You've won a prize..."
)

if st.button("🔎 Classify Email", type="primary"):
    if not NLTK_INITIALIZED_SUCCESSFULLY:
        st.error("NLTK components are not ready. Cannot classify accurately. Please check sidebar logs.")
    elif model is None or vectorizer is None:
        st.warning("The model or vectorizer could not be loaded. Please check the application setup and logs.")
    elif not email_text_input.strip():
        st.warning("⚠️ Please enter some email text to classify.")
    else:
        with st.spinner("🔍 Analyzing email..."):
            # 1. Preprocess using the NLTK-enabled function
            cleaned_text = preprocess_text_with_nltk(email_text_input)

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
                st.write("**Cleaned (Preprocessed) Text (with NLTK):**")
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
    "text preprocessing using the NLTK library."
)
st.sidebar.markdown("---")
st.sidebar.subheader("How it works (with NLTK):")
st.sidebar.markdown("""
- You provide the email text.
- NLTK resources (punkt, stopwords, wordnet) are checked/downloaded.
- The text is preprocessed using NLTK (tokenize, remove stopwords, lemmatize).
- The cleaned text is converted into numerical features using a pre-trained TF-IDF vectorizer.
- A pre-trained SVM model predicts whether the email is Spam or Ham.
""")
# GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY"
# st.sidebar.markdown(f"[View Source Code on GitHub]({GITHUB_REPO_URL})")

st.markdown("---")
st.caption("Built with Streamlit by [Your Name/Handle Here]")
