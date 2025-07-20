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

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stTextArea > label {
            font-size: 18px;
            color: #1f4e79;
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
st.markdown("<h1 style='text-align: center; color: #6a1b9a;'>📧 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Instantly classify messages using Machine Learning 🚀</p>", unsafe_allow_html=True)

# Input area
text_sms = st.text_area("✉️ Enter your message below:", height=150)

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
