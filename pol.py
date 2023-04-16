from textblob import TextBlob

sentence = "Fuck off"
blob = TextBlob(sentence)

# Get the sentiment polarity and subjectivity
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

# Print the results
print(f"Polarity: {polarity:.2f}")
print(f"Subjectivity: {subjectivity:.2f}")
