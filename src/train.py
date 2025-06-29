import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

# SageMaker passes environment variables for input/output
input_path = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
output_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

# Detect if running on SageMaker or locally
if os.path.exists('/opt/ml/input/data/train'):
    input_path = '/opt/ml/input/data/train'
else:
    # Use your local path for local runs
    input_path = 'C:/Users/sobanerj/Documents/Personal_Projects/sentiment-sagemaker/input'
output_dir = os.environ.get('SM_MODEL_DIR', './model')


# Locate your data file in input_path (SageMaker will copy it there)
csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
if len(csv_files) == 0:
    raise FileNotFoundError("No CSV file found in the input directory")
csv_file = os.path.join(input_path, csv_files[0])



# Load data
df = pd.read_csv(csv_file)
df = df[['review_body', 'sentiment']].dropna()
df['review_body'] = df['review_body'].astype(str)
df['sentiment'] = df['sentiment'].astype(int)

# Optional: downsample for speed (remove for full training)
# df = df.sample(n=5000, random_state=42)

# Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review_body'].tolist(), df['sentiment'].tolist(), test_size=0.2, random_state=42, stratify=df['sentiment']
)

# Load tokenizer and tokenize
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=128), batched=True)
val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=128), batched=True)

# Model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# Train!
trainer.train()
trainer.save_model(output_dir)
print("Training complete. Model saved to:", output_dir)

