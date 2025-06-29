import tarfile
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ----- 1. Extract model.tar.gz -----
MODEL_TAR_PATH = os.path.join('sagemaker_trained', 'model.tar.gz')
EXTRACTED_MODEL_DIR = os.path.join('sagemaker_trained', 'extracted')

# Only extract if not already done
if not os.path.isdir(EXTRACTED_MODEL_DIR):
    os.makedirs(EXTRACTED_MODEL_DIR, exist_ok=True)
    print("Extracting model.tar.gz...")
    with tarfile.open(MODEL_TAR_PATH, 'r:gz') as tar:
        tar.extractall(path=EXTRACTED_MODEL_DIR)
    print("Extraction complete.")

# ----- 2. Load model and tokenizer -----
print("Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(EXTRACTED_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(EXTRACTED_MODEL_DIR)
print("Model loaded.")

# ----- 3. Load dataset -----
DATASET_PATH = os.path.join('input', 'Amazon_Product_Reviews_US.csv')
df = pd.read_csv(DATASET_PATH)
# Change this column name if your text column is named differently!
TEXT_COLUMN = 'review_body'  

# ----- 4. Run inference on first 5 rows -----
print("\nSample Predictions:")
for idx, text in enumerate(df[TEXT_COLUMN].head(5)):
    inputs = tokenizer(str(text), return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    print(f"{idx+1}. Review: {text[:80]}...  -->  Predicted Class: {predicted_class_id}")

print("\nDone!")

