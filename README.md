# Sentiment Analysis 2025

## End-to-End Sentiment Analysis on Amazon Reviews using AWS SageMaker and HuggingFace Transformers

---

### **Project Overview**

This project demonstrates a full machine learning workflow for sentiment analysis on Amazon product reviews, including:

- **Cloud-based training** on AWS SageMaker with HuggingFace Transformers
- **IAM and resource management** for real-world deployment
- **Model artifact management** and **local inference** (batch and single predictions)
- **Visualization** of sentiment predictions

---

### **Project Structure**
sentiment-sagemaker/
├── input/ # Input dataset (CSV)

├── sagemaker_trained/ # Trained model artifact from SageMaker

├── src/ # Source scripts

├── Scripts/ # Utility scripts

├── local_inference.py # Script for extracting and running local predictions

├── batch_prediction.py # Batch inference & visualization script

├── .gitignore

└── README.md

---

### **How to Run Locally**

1. **Clone the repo:**
    ```sh
    git clone https://github.com/<your-username>/Sentiment_analysis_2025.git
    cd Sentiment_analysis_2025
    ```

2. **Install requirements (use virtualenv!):**
    ```sh
    pip install -r requirements.txt
    ```
    *(Add your `requirements.txt` using `pip freeze > requirements.txt` if needed)*

3. **Run local inference:**
    ```sh
    python local_inference.py
    ```
    - Extracts the SageMaker model and predicts sample reviews.

4. **Batch prediction & visualization:**
    ```sh
    python batch_prediction.py
    ```
    - Processes the whole dataset and shows result charts.

---

### **Screenshots**

*(Add training output, model extraction, predictions, and charts!)*

---

### **Key Learnings**
- Real-world AWS resource, quota, and IAM troubleshooting
- How to train and deploy Transformer models at scale
- Automating ML pipelines for reproducibility
- Visualizing and interpreting results

---

### **Tech Stack**
- Python 3.11+
- HuggingFace Transformers & PyTorch
- Pandas, Matplotlib
- AWS SageMaker (training), S3 (storage)
- IAM roles and permissions

---

### **Author**
- Souritra Banerjee

---

*For questions, open an issue or contact me on [LinkedIn](https://www.linkedin.com/)*
