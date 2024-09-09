import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from joblib import dump
from clean_data import clean_comment

# Load the IMDB dataset
data = pd.read_csv(
    r"C:\Users\ashis\Documents\001_Programming_related\My_Projects\InstaSentiment\data\IMDB_Data.csv")

# Clean the reviews
data['review'] = data['review'].apply(clean_comment)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['review']).toarray()
y = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the trained model and vectorizer
dump(model, r"C:\Users\ashis\Documents\001_Programming_related\My_Projects\InstaSentiment\models\sentiment_model.pkl")
dump(vectorizer, r"C:\Users\ashis\Documents\001_Programming_related\My_Projects\InstaSentiment\models\tfidf_vectorizer.pkl")
