# app.py
import streamlit as st
import joblib
import os
import re
import string
import nltk
# It's often better to import submodules after ensuring downloads if they cause issues on import
# from nltk.corpus import stopwords -> Will do this later
# from nltk.stem import WordNetLemmatizer -> Will do this later
# from nltk.tokenize import word_tokenize -> Will do this later

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Spam Email Detector", layout="wide", initial_sidebar_state="expanded")

# --- NLTK Resource Download (Revised Approach) ---
NLTK_RESOURCES_DOWNLOADED_SUCCESSFULLY = False
try:
    # Try a very basic NLTK operation to see if the base data path is working
    # This doesn't guarantee all resources, but checks if NLTK can find its data dir
    nltk.data.find("corpora/wordnet") # A common resource
    from nltk.corpus import stopwords # Now try importing
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    stopwords.words('english') # And try using them
    word_tokenize("test")
    WordNetLemmatizer().lemmatize("test")
    NLTK_RESOURCES_DOWNLOADED_SUCCESSFULLY = True
    st.sidebar.success("NLTK resources appear to be available.") # Subtle success message
except LookupError:
    st.sidebar.info("NLTK resources not found. Attempting to download...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True) # Open Multilingual Wordnet, dependency for WordNetLemmatizer

        # After download, re-attempt imports and usage
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        stopwords.words('english')
        word_tokenize("test")
        WordNetLemmatizer().lemmatize("test")

        NLTK_RESOURCES_DOWNLOADED_SUCCESSFULLY = True
        st.sidebar.success("NLTK resources downloaded successfully! Please refresh if it's the first run.")
    except Exception as e:
        st.sidebar.error(f"Failed to download or verify NLTK resources: {e}")
        st.error("Critical NLTK resources could not be loaded. App functionality will be limited. Please check the logs.")
except Exception as e:
    st.sidebar.error(f"An unexpected error occurred during NLTK setup: {e}")
    st.error("An error occurred with NLTK setup. App functionality may be limited.")


# --- Initialize NLTK components AFTER potential download ---
# These are now defined globally if NLTK_RESOURCES_DOWNLOADED_SUCCESSFULLY is True
if NLTK_RESOURCES_DOWNLOADED_SUCCESSFULLY:
    stop_words_english = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
else:
    # Fallback if NLTK is critically broken, to prevent app from crashing entirely
    st.warning("NLTK components could not be initialized. Preprocessing quality will be affected.")
    stop_words_english = set()
    class DummyLemmatizer:
        def lemmatize(self, word): return word
    wordnet_lemmatizer = DummyLemmatizer()
    def word_tokenize(text): # Dummy tokenizer
        return text.split()


# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    # ... (same as before)
    model_path = "svm_model.joblib"
    vectorizer_path = "tfidf_vectorizer.joblib"

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {os.path.abspath(model_path)}")
        return None, None
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer file not found at: {os.path.abspath(vectorizer_path)}")
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
    
    # Use the globally defined (or dummy) word_tokenize
    tokens = word_tokenize(text)
    
    tokens = [word for word in tokens if word not in stop_words_english and len(word) > 1]
    tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# --- Streamlit App UI ---
# ... (same as the previous "full code" version: st.title, st.markdown, text_area, button, etc.) ...
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
    if not NLTK_RESOURCES_DOWNLOADED_SUCCESSFULLY:
        st.error("NLTK resources are not available. Cannot preprocess text. Please check the sidebar messages and logs.")
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
