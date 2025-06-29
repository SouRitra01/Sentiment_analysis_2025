import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

MODEL_DIR = "sagemaker_trained/extracted"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=1)
        pred = probs.argmax(dim=1).item()
    return pred

# Load CSV (adjust path if needed)
df = pd.read_csv("input/Amazon_Product_Reviews_US.csv")
df = df.dropna(subset=['review_body'])
# Ensure 'review_body' is the correct column name for text reviews
if 'review_body' not in df.columns:
    raise ValueError("Column 'review_body' not found in the DataFrame. Please check the CSV file.")

# Check column names if you're not sure!
print("Columns found:", df.columns.tolist())

# Predict using review_body column
predictions = []
for idx, review in enumerate(df['review_body']):
    # Handle missing or non-string values gracefully
    if not isinstance(review, str):
        predictions.append(-1)   # or any default class/indicator for missing input
        continue
    pred = predict(review)
    predictions.append(pred)
    if idx % 100 == 0:
        print(f"Processed {idx} reviews")
df["predicted_class"] = predictions


#df["predicted_class"] = df["review_body"].astype(str).apply(predict)

# Save new CSV
output_csv = "input/Amazon_Product_Reviews_US_with_predictions.csv"
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")

# Visualize predicted sentiment distribution
sentiment_counts = df["predicted_class"].value_counts().sort_index()
labels = [f"Class {i}" for i in sentiment_counts.index]

plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Predicted Sentiment Distribution (Pie Chart)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(labels, sentiment_counts)
plt.title("Predicted Sentiment Distribution (Bar Chart)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
