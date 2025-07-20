import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Streamlit page config
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# Set full-page background and text styles
st.markdown("""
    <style>
        .stApp {
            background-color: #FFC1CC;
            color: #3B0A25;
        }
        h1, h2, h3 {
            color: #8B0000;
        }
        .stTextArea textarea {
            background-color: #fff5f7;
            color: #3B0A25;
        }
        .css-1cpxqw2 edgvbvh3 {  /* Optional: adjust button colors if needed */
            background-color: #8B0000;
        }
    </style>
""", unsafe_allow_html=True)
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = text.split()

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Title and subtitle
st.title("📩 Email/SMS Spam Classifier")
st.markdown("### 💡 Enter your message below to find out if it's SPAM or Not.")

# Input area
text_sms = st.text_area("✉️ Enter your message below:", height=150)

# Custom styled button
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #8B0000;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

# Predict button
if st.button("🔍 Predict"):
    transformed_sms = transform_text(text_sms)
    vector_input = tfidf.transform([transformed_sms])
    prediction = model.predict(vector_input)

    if prediction == 1:
        st.markdown("<h2 style='color: red;'>🚫 This is SPAM!</h2>", unsafe_allow_html=True)
        st.warning("⚠️ Be careful! This looks like a spam message.")
    else:
        st.markdown("<h2 style='color: green;'>✅ This is NOT SPAM!</h2>", unsafe_allow_html=True)
        st.success("🎉 Safe message! Looks clean.")
        st.balloons()  # 🎈🎈🎈 Balloons animation

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ by @sonal</p>", unsafe_allow_html=True)
