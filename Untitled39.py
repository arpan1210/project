
import streamlit as st
import pickle

# Load pre-trained model and vectorizer
model = pickle.load(open("randomforestmodel.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Streamlit App
st.title("Fake News Detection")

# Input text
news_text = st.text_area("Enter the news article:")

if st.button("Predict"):
    if news_text:
        # Transform and predict
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)
        result = "Real" if prediction[0] == 1 else "Fake"
        st.success(f"The news is: {result}")
    else:
        st.error("Please enter text to predict.")






