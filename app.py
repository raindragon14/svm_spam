# app.py
import streamlit as st
import joblib
import os
import re
import string
import nltk

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Spam Email Detector", layout="wide", initial_sidebar_state="expanded")

# --- Function to Attempt NLTK Resource Downloads ---
# No caching for now to ensure it runs every time for debugging
def attempt_nltk_downloads():
    st.sidebar.info("Attempting to ensure NLTK resources are available...")
    nltk_data_path_info = f"NLTK data paths: {nltk.data.path}"
    st.sidebar.text(nltk_data_path_info) # Log this to see where NLTK looks

    resources_to_download = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    all_successful = True

    for resource_name in resources_to_download:
        try:
            st.sidebar.text(f"Checking/Downloading NLTK resource: {resource_name}...")
            # Attempt to download. If it's already there, it usually doesn't re-download.
            nltk.download(resource_name, quiet=False) # Set quiet=False for more verbose output
            # Try a minimal operation to confirm
            if resource_name == 'punkt':
                nltk.word_tokenize("test")
            elif resource_name == 'stopwords':
                nltk.corpus.stopwords.words('english')
            elif resource_name == 'wordnet' or resource_name == 'omw-1.4':
                nltk.stem.WordNetLemmatizer().lemmatize("test")
            st.sidebar.text(f"Successfully verified/downloaded {resource_name}.")
        except Exception as e:
            st.sidebar.error(f"Error with NLTK resource '{resource_name}': {e}")
            all_successful = False
    
    if all_successful:
        st.sidebar.success("NLTK resource download/verification process complete.")
    else:
        st.sidebar.error("One or more NLTK resources failed to download/verify. Check logs.")
    return all_successful

# Call the download function once at the start
NLTK_READY = attempt_nltk_downloads()

# --- Initialize NLTK components AFTER ensuring resources are available ---
if NLTK_READY:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    try:
        stop_words_english = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        # Test word_tokenize again here to be sure
        word_tokenize("confirming tokenizer")
        st.sidebar.text("NLTK components (stopwords, lemmatizer, tokenizer) initialized.")
    except Exception as e:
        st.error(f"Failed to initialize NLTK components after download: {e}")
        NLTK_READY = False # Set to false if initialization fails
else:
    st.error("Critical NLTK resources could not be initialized due to download/verification issues. App functionality will be severely limited.")

# Define fallbacks if NLTK is not ready
if not NLTK_READY:
    stop_words_english = set()
    class DummyLemmatizer:
        def lemmatize(self, word, pos='n'): return word
    wordnet_lemmatizer = DummyLemmatizer()
    def word_tokenize(text): # Dummy tokenizer
        st.warning("Using fallback tokenizer due to NLTK issues.")
        return text.split()


# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
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
        st.error(f"Error loading model/vectorizer: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer()


# --- Text Preprocessing Function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' <NUM> ', text)
    
    tokens = word_tokenize(text) # Uses the globally defined or dummy word_tokenize
    
    tokens = [word for word in tokens if word not in stop_words_english and len(word) > 1]
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
        st.error("NLTK resources are not properly initialized. Cannot perform classification. Please check sidebar logs.")
    elif model is None or vectorizer is None:
        st.warning("The model or vectorizer could not be loaded. Please check the application setup and logs.")
    elif not email_text_input.strip():
        st.warning("⚠️ Please enter some email text to classify.")
    else:
        with st.spinner("🔍 Analyzing email..."):
            cleaned_text = preprocess_text(email_text_input)

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
            if prediction[0] == 1:
                st.error("🚨 This email is classified as: **Spam**")
                spam_probability = prediction_proba[0][1] * 100
                st.progress(int(spam_probability))
                st.markdown(f"Confidence (Spam): **{spam_probability:.2f}%**")
            else:
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
