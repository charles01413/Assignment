import spacy
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon reviews
reviews = [
    "The Apple iPhone 13 is amazing, but the battery doesn't last long.",
    "I love my Samsung Galaxy—it’s super fast and sleek.",
    "Sony headphones are good value for the price."
]

# NER and sentiment analysis
for review in reviews:
    doc = nlp(review)
    print("Review:", review)

    # Named Entity Recognition
    print("Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")

    # Sentiment Analysis
    sentiment = TextBlob(review).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    print("Sentiment:", sentiment_label)
    print("—" * 40)
