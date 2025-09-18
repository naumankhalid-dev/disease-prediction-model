# 🏥 Disease Prediction from Medical Data (Diabetes Dataset)

## 📌 Overview
This project focuses on predicting the likelihood of diabetes in patients using structured medical data. We applied multiple machine learning classification algorithms and identified **Random Forest** as the best-performing model. The project was developed as part of my **Machine Learning Internship Tasks**.

Google Colab Notebook 👉 [View Project](https://colab.research.google.com/drive/1T9NOkF7jzwKkjU2py3HYAS9pZltuXPW8?usp=sharing)

---

## 🎯 Objectives
- Predict the probability of a patient having diabetes
- Compare multiple classification algorithms
- Evaluate the models with accuracy, confusion matrix, and classification reports
- Provide sample patient predictions with probabilities

---

## 🗂 Dataset
- **Source**: [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target Variable**: Diabetes outcome (`0 = No Diabetes, 1 = Diabetes`)

---

## 🛠️ Technologies Used
- Python
- Google Colab
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

---

## 📊 Approach
1. **Data Loading & Preprocessing**
   - Handled missing/invalid values
   - Normalized features for fair comparison

2. **Exploratory Data Analysis (EDA)**
   - Statistical summaries
   - Feature correlations & distributions

3. **Model Training & Comparison**
   - Algorithms used:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest 🌟 (Best Model)
     - XGBoost

4. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix

5. **Final Model Selection**
   - Random Forest gave the highest accuracy and balanced performance

6. **Predictions**
   - Generated predictions for test patients
   - Displayed probabilities of having diabetes vs not

---

## 🏆 Results
- **Best Model**: Random Forest
- **Performance Summary**:
  - Achieved highest accuracy among all models
  - Balanced precision and recall
- **Visual Results**:
  - Accuracy Comparison Bar Chart
  - Confusion Matrix Visualization
  - Sample Predictions with Probabilities

---

## 🚀 How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-username/disease-prediction-randomforest.git
cd disease-prediction-randomforest
```
2.Open the notebook in Google Colab or Jupyter Notebook:
```bash
disease_prediction.ipynb
```
3.Upload the dataset (diabetes.csv) into your environment
4.Run all cells to reproduce the results.

---
## 📌 Future Work
Extend predictions to other diseases (Heart Disease, Breast Cancer, etc.)

Deploy the best model using Flask/Django as a web application

Implement feature importance visualization for interpretability

---
## 👨‍💻 Author

**Nauman Khalid**

🌐[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/nauman-khalid-58a2692a0)

