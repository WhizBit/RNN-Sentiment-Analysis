import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reverse = { value:key for key,value in word_index.items()}

model = load_model('Simple_RNN.h5')

#Helper Function

def decode_review(encoded_review):
    return " ".join(reverse(i-3,'?') for i in encoded_review)

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padding_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padding_review

# Prediction Function 
def predict_sentiment(review):
    preprocessed  = preprocess_text(review)
    prediction = model.predict(preprocessed)
    sentiment = 'Postive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment, prediction[0][0]

#Streamlit App

st.title('RNN Predictor')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie_Review')

if st.button('Classify'):
    preprocessed = preprocess_text(user_input)
    prediction = model.predict(preprocessed)
    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'

    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction Score : {prediction[0][0]}')

else:
    st.write('Please Enter a Movie Review')


