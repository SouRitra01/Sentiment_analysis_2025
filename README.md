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
![Project_image_1](https://github.com/user-attachments/assets/55a2ade4-fd5f-4c10-8024-a1964554cac3)

![Project_image_2](https://github.com/user-attachments/assets/87070db6-5e3f-4582-9ec8-638fc8aa5446)

![Project_image_3](https://github.com/user-attachments/assets/3f67810a-d899-4959-a822-97bc8b93b13d)

![Project_image_4](https://github.com/user-attachments/assets/ee39e837-68ea-49c5-a24b-163758dee2a6)

![Project_image_5](https://github.com/user-attachments/assets/bf7d72ba-f08f-4577-8dbb-f4e72c29450c)

![Project_image_6](https://github.com/user-attachments/assets/f1d67b46-04df-4e4c-bc66-62de79d999a9)

*Output Vizualizations*

![output2](https://github.com/user-attachments/assets/d4edcd26-6c01-4cb3-8a76-07448d0def54)

![output1](https://github.com/user-attachments/assets/d0604f88-0838-4529-810a-1f545db51809)


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

*For questions, open an issue or contact me on [LinkedIn]([https://www.linkedin.com/](https://www.linkedin.com/in/souritra1/))*

Let me know once you’re done pushi
