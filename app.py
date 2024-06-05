import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download stopwords if not already present
nltk.download('stopwords')

# Load the stopwords
stop_words = set(stopwords.words('english'))

# Load the model and vectorizer
model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
lemmatizer = WordNetLemmatizer()

# Define a function for preprocessing a single text
def preprocess_text(text):
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Streamlit app
st.title("Text Classification App")

st.write("Enter some text and the model will predict the tag:")

input_text = st.text_area("Enter your text here")

if st.button("Predict"):
    if input_text:
        processed_text = preprocess_text(input_text)
        text_vector = tfidf_vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)
        st.header(f"Predicted Tag: {prediction[0]}")
    else:
        st.warning("Please enter some text to get a prediction.")
