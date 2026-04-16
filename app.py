
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Download NLTK resources silently ─────────────────────────
for pkg in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

lemmatizer  = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

# ── Load the saved model (do NOT retrain) ────────────────────
@st.cache_resource
def load_model():
    return joblib.load('spam_classifier.pkl')

model = load_model()

# ── Preprocessing pipeline (mirrors training pipeline) ───────
def preprocess(text: str) -> tuple[str, list[str]]:
    """Returns (cleaned_string, token_list)"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # HTML tags
    text = re.sub(r'\S+@\S+', ' ', text)           # emails
    text = re.sub(r'http\S+|www\S+', ' ', text)    # URLs
    text = re.sub(r'\d+', ' ', text)               # numbers
    text = re.sub(r'[^a-z\s]', ' ', text)           # punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens), tokens

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Classifier")
st.markdown("Paste the raw email text below and click **Classify** to detect spam.")
st.divider()

# ── Input ─────────────────────────────────────────────────────
raw_email = st.text_area(
    "✉️ Raw Email Text",
    height=200,
    placeholder="Paste your email content here..."
)

if st.button("🔍 Classify", use_container_width=True):
    if not raw_email.strip():
        st.warning("Please enter some email text first.")
    else:
        # ── Preprocessing Display ─────────────────────────────
        cleaned, tokens = preprocess(raw_email)

        st.subheader("🧹 Preprocessing Pipeline")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Input**")
            st.text_area("", value=raw_email[:500], height=150, disabled=True, key="raw")
        with col2:
            st.markdown("**Cleaned / Tokenised**")
            st.text_area("", value=cleaned[:500], height=150, disabled=True, key="cleaned")

        st.markdown(f"**Tokens ({len(tokens)}):** `{tokens[:20]}`")
        st.divider()

        # ── Prediction ────────────────────────────────────────
        prediction   = model.predict([cleaned])[0]
        confidence   = model.predict_proba([cleaned])[0].max() * 100

        label = "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
        colour = "red" if prediction == 1 else "green"

        st.subheader("🎯 Prediction Result")
        st.markdown(
            f"<h2 style='color:{colour}; text-align:center'>{label}</h2>",
            unsafe_allow_html=True
        )
        st.metric("Confidence", f"{confidence:.1f}%")
        st.progress(confidence / 100)

        if prediction == 1:
            st.error("⚠️ This email has been classified as **Spam**.")
        else:
            st.success("✔️ This email appears to be **Legitimate**.")

st.divider()
st.caption("Built with Scikit-learn & Streamlit | Ling-Spam Dataset | NLP Assignment 3")
