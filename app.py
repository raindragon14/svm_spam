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

# --- NLTK Resource Download (Corrected Section) ---
# List of NLTK resources needed by the app
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
resources_to_download_explicitly = []

# Check each resource and add to download list if missing
for resource_name in nltk_resources:
    try:
        # A simple way to check for resource availability is to try a minimal operation
        if resource_name == 'punkt':
            nltk.word_tokenize("test")
        elif resource_name == 'stopwords':
            stopwords.words('english')
        elif resource_name == 'wordnet': # omw-1.4 is a dependency for wordnet, so checking wordnet often covers both
            WordNetLemmatizer().lemmatize("tests")
        elif resource_name == 'omw-1.4': # Explicit check if needed, though wordnet check might suffice
            # No direct small operation for omw-1.4 itself, usually implicitly used by WordNetLemmatizer
            # We can assume if wordnet works, omw-1.4 (its dependency) is likely okay or will be handled by wordnet download
            pass # Covered by wordnet check or wordnet download
    except LookupError:
        resources_to_download_explicitly.append(resource_name)
    except Exception: # Catch other potential errors during the check, be safe
        resources_to_download_explicitly.append(resource_name)

# Remove duplicates just in case
resources_to_download_explicitly = list(set(resources_to_download_explicitly))

if resources_to_download_explicitly:
    st.info(f"Attempting to download missing NLTK resources: {', '.join(resources_to_download_explicitly)}")
    all_downloaded_successfully = True
    for resource_to_download in resources_to_download_explicitly:
        try:
            nltk.download(resource_to_download, quiet=True)
        except Exception as e:
            st.error(f"Failed to download NLTK resource '{resource_to_download}': {e}")
            all_downloaded_successfully = False
    if all_downloaded_successfully:
        st.success("NLTK resources checked/downloaded. The app should now function correctly. You might need to refresh if this was the first run.")
    else:
        st.error("Some NLTK resources could not be downloaded. App functionality might be limited.")


# --- Load Model and Vectorizer ---
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
# These NLTK components should be initialized after ensuring resources are downloaded
try:
    stop_words_english = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
except LookupError as e:
    st.error(f"NLTK resource for preprocessing (stopwords/wordnet) not available even after download attempt: {e}. Please check logs.")
    # Fallback to empty sets/dummy lemmatizer if critical, or stop the app
    stop_words_english = set()
    class DummyLemmatizer:
        def lemmatize(self, word): return word
    wordnet_lemmatizer = DummyLemmatizer()


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' <NUM> ', text) # Replace numbers with a placeholder
    try:
        tokens = word_tokenize(text)
    except LookupError: # Fallback if punkt still fails
        st.warning("Tokenizer (punkt) might not be fully available. Splitting by space as a fallback.")
        tokens = text.split()

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
    if model is None or vectorizer is None:
        st.warning("The model or vectorizer could not be loaded. Please check the application setup and logs.")
    elif not email_text_input.strip():
        st.warning("⚠️ Please enter some email text to classify.")
    else:
        with st.spinner("🔍 Analyzing email..."):
            cleaned_text = preprocess_text(email_text_input)

            if not cleaned_text.strip() and email_text_input.strip(): # If input had content but cleaning removed it all
                 st.info("The input text resulted in no meaningful content after preprocessing (e.g., only stopwords, punctuation, or numbers).")
                 # Optionally, you could classify the empty string or skip classification
                 # For now, let's try to classify anyway, vectorizer might handle empty string
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
# To add your GitHub link:
# GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY" # Replace with your actual link
# st.sidebar.markdown(f"[View Source Code on GitHub]({GITHUB_REPO_URL})")

st.markdown("---")
st.caption("Built with Streamlit by [Your Name/Handle Here]") # Replace with your name/handle
