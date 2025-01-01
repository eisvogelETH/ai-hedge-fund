# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Example financial text (can be news, stock reports, etc.)
text = "the ceo is selling stocks"
# Tokenize the input
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class (sentiment)
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

# Map the predicted class to the sentiment labels
sentiment_labels = ['negative', 'neutral', 'positive']
sentiment = sentiment_labels[predicted_class]

print(f"Sentiment: {sentiment}")