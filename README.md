# 📩 Email/SMS Spam Classifier

This is a machine learning web application built using **Streamlit** that classifies messages (Email or SMS) as **Spam** or **Not Spam** using **Naive Bayes classifier** and **TF-IDF vectorization**.  

---

## 🔍 Features

- 🧠 **Naive Bayes Classifier**: Trained on labeled SMS/email dataset to detect spam.
- 📊 **TF-IDF Vectorization**: Converts text into numerical feature vectors for model input.
- 🧹 **Text Preprocessing**: Includes lowercasing, punctuation removal, stopword removal, stemming.
- 🗃️ **Pickle Model Loading**: Pre-trained model and vectorizer are loaded using `.pkl` files.
- ⚡ **Real-time Prediction**: Instant classification on user input without page reload.

---

## 🧰 Tech Stack

- **Frontend**: Streamlit, HTML, CSS
- **Backend**: Python
- **ML Model**: Naive Bayes
- **NLP**: TF-IDF, NLTK
- **Dataset Used**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---
## 🧠 Machine Learning

- **Vectorizer**: `TfidfVectorizer` (Term Frequency–Inverse Document Frequency)
- **Classifier**: `Multinomial Naive Bayes` from `scikit-learn`
- **Text Processing**:
  - Stopword removal using NLTK
  - Word stemming using `PorterStemmer`
  - Tokenization using simple Python split (you can use `nltk.word_tokenize` as well)

---
## 🌐 Live Demo
👉 [link](https://email-spam-detection-99.streamlit.app/)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sonall99/Email-Spam-Detection
cd spam-classifier
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3.Run the app
```bash
streamlit run spam-app.py
```
