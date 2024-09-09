# src/predict_sentiment.py
import pandas as pd
from joblib import load
from src.clean_data import clean_comment

# Load the pre-trained model and vectorizer
model = load(
    r"C:\Users\ashis\Documents\001_Programming_related\My_Projects\InstaSentiment\models\sentiment_model.pkl")
vectorizer = load(
    r"C:\Users\ashis\Documents\001_Programming_related\My_Projects\InstaSentiment\models\tfidf_vectorizer.pkl")


def predict_sentiment(comment):
    # Clean the comments
    cleaned_comments = [clean_comment(comment) for comment in comment]
    null_val = cleaned_comments.count('')
    # Convert the comments into a vectorized form using the loaded vectorizer
    X_test = vectorizer.transform(cleaned_comments)
    # Predict sentiments using the pre-trained model
    predictions = model.predict(X_test)
    return predictions, null_val
