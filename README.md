# AI-ML-TASK-NO-03-
# Week 3 Internship Task
## Dimensionality Reduction (PCA) and FastAPI Dashboard

### Student Information
- Name: Taha Adnan
- University: Abasyn University Islamabad
- Internship: AI/ML Internship

---

## Project Description

This project focuses on reducing the number of features using Principal Component Analysis (PCA) and deploying the machine learning model using FastAPI.

The goal is to maintain model performance while reducing dimensionality and providing real-time predictions through a web dashboard.

---

## Dataset

- Dataset: Steel Industry Energy Consumption Dataset
- Target Variable: Usage_kWh

---

## Part 1: PCA (Dimensionality Reduction)

### Tasks Completed

- Data Loading and Preprocessing
- Feature Engineering
- Data Scaling using StandardScaler
- PCA Implementation
- Scree Plot Generation
- Cumulative Variance Analysis
- PCA with 3 Components
- PCA with 95% Variance Components
- Model Performance Comparison
- PCA Loading Heatmap

### Evaluation Metrics

- RMSE (Root Mean Squared Error)
- R² Score

---

## Part 2: FastAPI Dashboard

### Features

- Home Page
- Dashboard Page
- Prediction Page
- Real-Time Energy Consumption Prediction

### Dashboard Visualizations

- Energy Consumption by Hour
- Energy Consumption by Load Type
- Correlation Heatmap

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- PCA
- FastAPI
- Jinja2
- Joblib

---

## Project Structure

```text
Week3_Project/
│
├── main.py
├── model.joblib
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── week3_pca.ipynb
│
├── static/
│   ├── energy_by_hour.png
│   ├── load_type.png
│   └── correlation_heatmap.png
│
└── templates/
    ├── home.html
    ├── dashboard.html
    └── predict.html
```

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload

Open your browser:
http://127.0.0.1:8000
PCA reduced the number of features while preserving most of the important information.
The FastAPI dashboard allows users to view visualizations and generate real-time energy consumption predictions.
