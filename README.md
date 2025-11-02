

# 🧠 AWS MLOps Demo — Diabetes Prediction

This project demonstrates a **complete MLOps workflow on AWS** — from model training to deployment — using **SageMaker, S3, and Gradio**.  
It includes model versioning, endpoint deployment, and a simple user interface for real-time predictions.

---

## 🚀 Features
- Train a **Linear Regression model** on the Diabetes dataset (from `scikit-learn`)
- Store and manage the model in **AWS S3**
- Deploy the model as an **API endpoint** using **Amazon SageMaker**
- Build an interactive **Gradio UI** for predictions
- Use `.env` to keep all AWS secrets safe (no hardcoding credentials)

---

## ⚙️ Steps

### 1️⃣ Train & Upload Model

Diabetes_Model.py

This script:

Trains the Linear Regression model on the diabetes dataset
Saves it as diabetes_model.pkl
Uploads it to S3 bucket

### 2️⃣ Deploy to SageMaker


Diabetes_Deploy.py


Deploys the model as a SageMaker endpoint.

### 3️⃣ Test Locally

Use Gradio for easy predictions

## 🧩 Tools

* Python + scikit-learn
* AWS S3 + SageMaker
* Boto3 + Gradio
* dotenv for security

---

**By Razan Alkhamisi**


