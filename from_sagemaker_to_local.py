from transformers import AutoTokenizer

model_name = "bert-base-uncased"  # Change if you used a different model!
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("sagemaker_trained/extracted")
