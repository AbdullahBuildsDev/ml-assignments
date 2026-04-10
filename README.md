# 🤖 Machine Learning Assignments

**Student:** Abdullah Malik  
**GitHub:** AbdullahBuildsDev  
**Subject:** Machine Learning  
**Weight:** 70% of final grade

---

## 📁 Assignment 1 — Logistic Regression
- 📊 **Tabular Dataset:** Gold Price Trends (Kaggle) → Predicts price UP or DOWN
- 🧠 **Image Dataset:** Brain Tumor MRI → Detects Tumor or No Tumor
- ⚙️ **Method:** Logistic Regression (Binary Classification)
- 🌐 **Interface:** Flask Web App

## 📁 Assignment 2 — Model Comparison
- 🧠 **Dataset:** Brain Tumor MRI Images
- ⚙️ **Methods:** KNN, Decision Tree, Naive Bayes
- 📊 **Accuracy:** All models 78.43%
- 📉 **Confusion Matrix:** Generated for all 3 models
- 🌐 **Interface:** Flask Web App

## 📁 Assignment 3 — HuggingFace Pretrained Model
- 🤗 **Model:** DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- ⚙️ **Task:** Sentiment Analysis (Positive / Negative)
- 💡 **Concept:** Transfer Learning
- 🌐 **Interface:** Flask Web App

---

## 🛠️ Tech Stack
- Python 3.9
- Flask
- Scikit-learn
- HuggingFace Transformers
- PyTorch
- Pandas, NumPy
- Matplotlib, Seaborn

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/AbdullahBuildsDev/ml-assignments.git
cd ml-assignments

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install flask scikit-learn pandas numpy matplotlib seaborn pillow transformers torch

# Run Assignment 1
cd assignment1 && python3 app.py

# Run Assignment 2
cd assignment2 && flask run --port 5001

# Run Assignment 3
cd assignment3 && python3 app.py
```
