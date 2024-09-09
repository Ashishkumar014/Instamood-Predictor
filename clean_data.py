import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_comment(comment):
    comment = comment.lower()  # Lowercase text
    comment = re.sub(r'http\S+|www\S+', '', comment)  # Remove URLs
    comment = re.sub(r'[^a-zA-Z\s]', '', comment)  # Remove non-alphabet characters
    tokens = comment.split()  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize & remove stopwords
    return ' '.join(tokens)
