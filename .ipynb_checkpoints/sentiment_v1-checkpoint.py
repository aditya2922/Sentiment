from textblob import TextBlob

# Map sentiment labels to categories
sentiment_to_category = {
    "Positive": ["education", "transport", "health", "hygiene", "neutral"],
    "Negative": ["crime"],
    "Neutral": []
}

# Take user input for a sentence
sentence = input("Enter a sentence: ")

# Create a TextBlob object from the input sentence
blob = TextBlob(sentence)

# Determine sentiment polarity
if blob.sentiment.polarity > 0:
    sentiment = "Positive"
elif blob.sentiment.polarity < 0:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

# Map sentiment label to categories
categories = sentiment_to_category[sentiment]

# Print predicted categories
if len(categories) == 0:
    print("The sentence is neutral.")
else:
    print("The sentence is related to the following categories:")
    for category in categories:
        print("- " + category)
