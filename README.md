# 🫀 Heart Disease Prediction System

An end-to-end **Machine Learning web application** that predicts the **risk of heart disease** based on clinical patient data. The project demonstrates the complete data science workflow — from preprocessing and model training to deployment using Streamlit.

🌐 **Live App:** [https://heartdiseaseprediction-vlqukuwef9mw4ntipbqlbf.streamlit.app/](https://heartdiseaseprediction-vlqukuwef9mw4ntipbqlbf.streamlit.app/)

---

## 🎯 Project Motivation

Heart disease is one of the leading causes of death worldwide. Early prediction can help in timely medical intervention. This project aims to provide a simple, interactive tool that leverages machine learning to estimate heart disease risk using commonly available medical parameters.

---

## 🧠 Machine Learning Workflow

1. Data Collection (Clinical heart disease dataset)
2. Data Cleaning & Preprocessing
3. One-Hot Encoding of Categorical Features
4. Feature Scaling using StandardScaler
5. Model Training
6. Model Evaluation
7. Deployment using Streamlit

---

## 📊 Dataset Information

* Type: Clinical tabular dataset
* Records: 300+ patient entries
* Features: Age, sex, chest pain type, cholesterol, blood pressure, etc.
* Target Variable:

  * `0` → No Heart Disease
  * `1` → Presence of Heart Disease

---

## 🤖 Machine Learning Model

* Algorithm: **K-Nearest Neighbors (KNN)**
* Feature Scaling: **StandardScaler**
* Output: Binary classification with prediction confidence

---

## 📈 Key Features

* Interactive Streamlit-based UI
* Real-time heart disease risk prediction
* Prediction confidence score
* Explanation of medical terms for better user understanding
* Clean and beginner-friendly interface

---

## 🖥️ Application Preview

Below are screenshots showcasing different sections of the application:

### 🏠 Home Page

![Home Page](screenshots/1_home.png)

### 🧮 Prediction Page

![Prediction Page](screenshots/2_predict.png)

### 📘 Medical Terms & Abbreviations (Part 1)

![Medical Terms 1](screenshots/3_Medical_term_used.png)

### 📘 Medical Terms & Abbreviations (Part 2)

![Medical Terms 2](screenshots/4_Medical_term_used.png)

### ℹ️ About Page

![About Page](screenshots/5_about.png)

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Model Persistence:** Joblib

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

The application is deployed using **Streamlit Cloud** and can be accessed here:

🔗 [https://heartdiseaseprediction-vlqukuwef9mw4ntipbqlbf.streamlit.app/](https://heartdiseaseprediction-vlqukuwef9mw4ntipbqlbf.streamlit.app/)

---

## 👤 Author

**Shivam Maurya**
📊 Data Science Student








