# app.py
import streamlit as st
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Spam Email Detector", layout="wide", initial_sidebar_state="expanded")

# --- NLTK Resource Download ---
# This section tries to download NLTK resources if they are not found.
# In a deployed environment (like Streamlit Cloud), this often runs once.
try:
    # Test if resources are available by trying to use them
    stopwords.words('english')
    word_tokenize("test string")
    WordNetLemmatizer().lemmatize("running")
except LookupError as e:
    st.info(f"NLTK resource not found ({e}). Attempting to download...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True) # Often needed by WordNetLemmatizer
        st.success("NLTK resources downloaded successfully. Please refresh the page if issues persist.")
        # Test again after download
        stopwords.words('english')
        word_tokenize("test string")
        WordNetLemmatizer().lemmatize("running")
    except Exception as download_exc:
        st.error(f"Failed to download NLTK resources: {download_exc}. App functionality might be limited.")


# --- Load Model and Vectorizer ---
# Use st.cache_resource to load these only once and speed up the app.
# This decorator ensures that the function's result is cached, so the
# model and vectorizer are not reloaded on every interaction.
@st.cache_resource
def load_model_and_vectorizer():
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

# --- Text Preprocessing Function (MUST BE THE SAME AS USED IN TRAINING) ---
# Ensure NLTK components are initialized globally if used here
stop_words_english = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str): # Handle potential non-string values
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Example: Replace numbers with a placeholder. Adjust if you handled them differently.
    text = re.sub(r'\d+', ' <NUM> ', text)
    tokens = word_tokenize(text)
    # Remove stopwords and short tokens
    tokens = [word for word in tokens if word not in stop_words_english and len(word) > 1]
    # Lemmatization
    tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# --- Streamlit App UI ---
st.title("📧 Spam Email Detector")
st.markdown("""
    Welcome to the Spam Email Detector!
    Enter an email text in the box below and click "Classify Email" to see if it's predicted as Spam or Ham (Not Spam).
    This application uses a Support Vector Machine (SVM) model.
""")
st.write("---") # Separator

# Input text area for user to paste email content
email_text_input = st.text_area(
    "Enter Email Text Here:",
    height=250,
    placeholder="Dear User, congratulations! You've won a prize..."
)

# Prediction button
if st.button("🔎 Classify Email", type="primary"):
    if model is None or vectorizer is None:
        st.warning("The model or vectorizer could not be loaded. Please check the application setup.")
    elif not email_text_input.strip(): # Check if input is not just whitespace
        st.warning("⚠️ Please enter some email text to classify.")
    else:
        with st.spinner("🔍 Analyzing email..."):
            # 1. Preprocess the input text
            cleaned_text = preprocess_text(email_text_input)

            # 2. Vectorize the preprocessed text
            # vectorizer.transform expects an iterable (e.g., a list of strings)
            try:
                vectorized_text = vectorizer.transform([cleaned_text])
            except Exception as e:
                st.error(f"Error during text vectorization: {e}")
                st.stop() # Stop further execution in this block

            # 3. Make prediction
            try:
                prediction = model.predict(vectorized_text)
                prediction_proba = model.predict_proba(vectorized_text)
            except Exception as e:
                st.error(f"Error during model prediction: {e}")
                st.stop() # Stop further execution in this block


            # Display result with more flair
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

            # Expander for more details
            with st.expander("🔬 Show Details"):
                st.write("**Original Input:**")
                st.text(email_text_input[:500] + "..." if len(email_text_input) > 500 else email_text_input) # Show snippet
                st.write("**Cleaned (Preprocessed) Text:**")
                st.text(cleaned_text if cleaned_text.strip() else "No meaningful text after cleaning.")
                st.write("**Prediction Probabilities:**")
                st.write(f"Ham (Class 0): {prediction_proba[0][0]:.4f}")
                st.write(f"Spam (Class 1): {prediction_proba[0][1]:.4f}")
else:
    st.info("☝️ Enter email text above and click the 'Classify Email' button.")

# Sidebar for additional information
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
- The text is preprocessed (lowercase, remove punctuation, remove stopwords, lemmatize).
- The cleaned text is converted into numerical features using a pre-trained TF-IDF vectorizer.
- A pre-trained SVM model predicts whether the email is Spam or Ham.
""")
# Add your GitHub link or name if you like
# st.sidebar.markdown("[View Source Code on GitHub](YOUR_GITHUB_REPO_LINK_HERE)")

st.markdown("---")
st.caption("Built with Streamlit by [Your Name/Handle]")
