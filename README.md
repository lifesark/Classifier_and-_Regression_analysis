# 💼 Loan Default Risk Analysis

This project explores loan default prediction and amount forecasting using a comprehensive machine learning pipeline for both **classification** and **regression** tasks. The dataset is preprocessed, scaled, and used to train various models with hyperparameter tuning, PCA analysis, and performance evaluation.

---

## 📁 Project Structure

- `classifier_and_regression_analysis.ipynb`: Classification models, scaling, PCA, grid search,Regression models, performance comparison, Cleaned dataset with feature-engineered variables
- `loan_data.csv`: Cleaned dataset with feature-engineered variables
- `README.md`: Project overview, methodology, and results

---

## 📊 Classification: Will the applicant default?

### ✅ Models Used
- Logistic Regression
- Naive Bayes
- k-Nearest Neighbors (kNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting

---

### 📈 Results Summary

#### **1. Base Classifier Performance (No Scaling)**
| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9132  | 0.9121    | 0.9146 | 0.9133   |
| Naive Bayes         | 0.8313  | 0.8302    | 0.8317 | 0.8308   |
| kNN                 | 0.8981  | 0.8951    | 0.9022 | 0.8986   |
| SVM                 | 0.9188  | 0.9183    | 0.9200 | 0.9191   |
| Decision Tree       | 0.8997  | 0.8982    | 0.9019 | 0.9000   |
| Random Forest       | **0.9207** | **0.9202** | **0.9214** | **0.9208** |
| Gradient Boosting   | 0.9191  | 0.9185    | 0.9200 | 0.9192   |

---

#### **2. Impact of Feature Scaling**
| Model              | MinMax | Standard | MaxAbs | Robust | L1 Norm | L2 Norm |
|-------------------|--------|----------|--------|--------|---------|---------|
| Logistic Regression | 0.9122 | **0.9162** | 0.9135 | 0.9135 | 0.7282 | 0.7261 |
| Naive Bayes         | 0.8310 | 0.8310   | 0.8310 | 0.8310 | 0.8310 | 0.8310 |
| kNN                 | 0.9057 | **0.9086** | 0.9057 | 0.9072 | 0.8811 | 0.8793 |
| SVM                 | 0.9179 | **0.9207** | 0.9175 | 0.9183 | 0.9100 | 0.9107 |
| Decision Tree       | 0.9011 | 0.9017   | 0.9017 | **0.9024** | 0.8940 | 0.8940 |
| Random Forest       | 0.9186 | **0.9217** | 0.9193 | 0.9200 | 0.9193 | 0.9193 |
| Gradient Boosting   | 0.9191 | **0.9230** | 0.9197 | 0.9197 | 0.9176 | 0.9172 |

> 🔎 **Insight:** StandardScaler consistently yielded the best performance across most models.

---

#### **3. Grid Search + Cross Validation**
| Model              | Best Estimator                        | Test Accuracy | F1 Score |
|-------------------|----------------------------------------|---------------|----------|
| Logistic Regression | `C=0.01, solver='liblinear'`         | 0.9142        | 0.9136   |
| kNN                 | `n_neighbors=5, metric='manhattan'`  | 0.9086        | 0.9085   |
| SVM                 | `C=1, kernel='rbf'`                   | 0.9201        | 0.9194   |
| Decision Tree       | `max_depth=5, min_samples_split=2`   | 0.9014        | 0.9013   |
| Random Forest       | `n_estimators=200, max_depth=10`     | 0.9215        | 0.9204   |
| Gradient Boosting   | `n_estimators=200, learning_rate=0.1, max_depth=3` | **0.9226** | **0.9210** |

---

#### **4. PCA (10 components capturing 97.05% variance)**

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9049  | 0.9040    | 0.9060 | 0.9050   |
| kNN                 | 0.8943  | 0.8909    | 0.8998 | 0.8953   |
| SVM                 | 0.9123  | 0.9113    | 0.9143 | 0.9128   |
| Decision Tree       | 0.8861  | 0.8823    | 0.8910 | 0.8866   |
| Random Forest       | **0.9145** | **0.9139** | **0.9150** | **0.9144** |
| Gradient Boosting   | 0.9128  | 0.9118    | 0.9144 | 0.9131   |

---

### 🎯 Classification Highlights
- **Top performer:** Gradient Boosting (with tuning and standard scaling)
- **Feature Importance:** `loan_int_rate`, `loan_percent_income`, `loan_amnt`, and `previous_loan_defaults_on_file`
- **Scaling impact:** StandardScaler > MinMax > others

---

## 📉 Regression: What amount might they default on?

### ✅ Models Used
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- Decision Tree
- Random Forest

---

### 📈 Regression Results

| Model              | RMSE   | MAE    | R² Score |
|-------------------|--------|--------|----------|
| Linear Regression | 9.37   | 6.55   | 0.375    |
| Ridge             | 9.37   | 6.55   | 0.375    |
| Lasso             | 9.36   | 6.54   | 0.376    |
| ElasticNet        | 9.35   | 6.53   | 0.377    |
| Decision Tree     | 10.89  | 7.87   | 0.176    |
| Random Forest     | **8.96** | **6.10** | **0.468** |

---

### 🧠 Regression Highlights
- **Best model:** Random Forest
- **Ensemble models** clearly outperform linear models
- **Features like** `loan_amnt`, `loan_percent_income`, and `loan_int_rate` have high predictive power

---

## 📌 Key Takeaways

- **Gradient Boosting (classification)** and **Random Forest (regression)** consistently deliver top results.
- **Standard scaling** is the most effective preprocessing step.
- **Feature engineering and PCA** play important roles in performance.
- Grid search and cross-validation significantly boost model accuracy.

---

## 📈 Next Steps

- Explore **stacking ensemble models**
- Apply **SHAP or LIME** for interpretability
- Introduce more domain-specific features
- Deploy best models via a simple web app (e.g., Streamlit or Flask)

---

## 🧠 Author

**Machine Learning Project by Vibhanshu Vaibhav

> This work was done as part of a personal project to explore predictive modeling techniques in finance and credit risk assessment.

---

## 🛠️ Libraries Used

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

---

## 📬 Contact

Feel free to reach out via [GitHub Issues](https://github.com/your-username/loan-default-risk/issues) or contribute to the project!

---
