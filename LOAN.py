import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm  # For progress bar
import time

# Load the data
data = pd.read_csv('loan_data.csv')

# Data Cleaning and Preprocessing

## Remove rows with extreme values in 'person_age' and 'person_emp_exp'
data = data[(data['person_age'] > 0) & (data['person_age'] < 100)]
data = data[data['person_emp_exp'] <= 60]

# Handle missing values
## For numerical columns, impute with the mean
numerical_cols_with_nulls = ['person_emp_exp', 'loan_int_rate']
for col in numerical_cols_with_nulls:
    data[col] = data[col].fillna(data[col].mean())

# Encode categorical features
categorical_cols = [
    'person_gender', 'person_education', 'person_home_ownership', 'loan_intent',
    'previous_loan_defaults_on_file'
]
for col in categorical_cols:
    data[col] = data[col].astype('category')
    data[col] = data[col].cat.codes

# Define features (X) and target (y)
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True), # Probability needed for some metrics
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Evaluation metrics
metrics = {
    'Accuracy': accuracy_score,
    'MSE': mean_squared_error,
    'MAE': mean_absolute_error,
    'R2': r2_score
}

# Number of cross-validation folds
n_folds = 5

results = {}

# Model Evaluation Loop with Progress Bar
for name, model in tqdm(models.items(), desc='Evaluating Models', unit='model'):
    start_time = time.time()
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])

    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = {}
    for metric_name, metric_func in metrics.items():
        cv_scores[metric_name] = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error' if metric_name in ['MSE', 'MAE'] else 'accuracy') # Corrected scoring for CV
        cv_scores[metric_name] = -cv_scores[metric_name] if metric_name in ['MSE', 'MAE'] else cv_scores[metric_name]  # Revert the sign for MSE/MAE

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model on test set
    test_scores = {}
    for metric_name, metric_func in metrics.items():
        test_scores[metric_name] = metric_func(y_test, y_pred)

    end_time = time.time()
    training_time = end_time - start_time

    # Store results
    results[name] = {
        'cv_scores': cv_scores,
        'test_scores': test_scores,
        'classification_report': classification_report(y_test, y_pred),
        'training_time': training_time
    }

# Find the best model based on accuracy
best_model_name = max(results, key=lambda k: np.mean(results[k]['cv_scores']['Accuracy']))  #Use Cross Validation Accuracy to find best model.
best_model = models[best_model_name]

print("\n"+"="*50)
print("Model Evaluation Results:")
print("="*50)

for name, result in results.items():
    print(f"\nModel: {name}")
    print(f"Training Time: {result['training_time']:.2f} seconds")

    print("\nCross-Validation Scores:")
    for metric_name, scores in result['cv_scores'].items():
        print(f"  {metric_name}: Mean = {np.mean(scores):.4f}, Std = {np.std(scores):.4f}")

    print("\nTest Set Scores:")
    for metric_name, score in result['test_scores'].items():
        print(f"  {metric_name}: {score:.4f}")

    print("\nClassification Report:")
    print(result['classification_report'])

print("\n"+"="*50)
print(f"Best Model: {best_model_name}")
print("="*50)
