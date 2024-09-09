# main.py
import pandas as pd
from src.predict_sentiment import predict_sentiment

# Read the Instagram comments from the CSV file
df = pd.read_csv("path\to\your\csv\file")

# Extract the comments from the DataFrame
comments = df['comment'].tolist()

# Predict sentiment for each comment
sentiments = predict_sentiment(comments)[0]

# Add the sentiments back to the DataFrame
df['sentiment'] = sentiments

# Save the DataFrame with sentiments to a new CSV file
df.to_csv('instagram_comments_with_sentiment.csv', index=False)

new_df = pd.read_csv(
    r"C:\Users\ashis\Documents\001_Programming_related\My_Projects\InstaSentiment\instagram_comments_with_sentiment.csv")
new_df_row = len(new_df)
positive_reviews = new_df['sentiment'].eq(1).sum()
null_val = predict_sentiment(comments)[1]
total_reviews_excluding_emoji_reviews = len(new_df) - null_val
percent = round(
    (positive_reviews/total_reviews_excluding_emoji_reviews) * 100, 2)

print(f"The positive comments is {percent} %")

print("Sentiment prediction complete. Results saved to 'instagram_comments_with_sentiment.csv'.")
