#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
print("Loading training dataset...")
df = pd.read_csv('cs-training.csv')
print("Dataset loaded. Shape:", df.shape)

# Drop Unnamed column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Fill missing values with mean
print("Handling missing values...")
df.fillna(df.mean(), inplace=True)

# Feature Engineering
print("Performing feature engineering...")
df['income_to_debt'] = df['MonthlyIncome'] / (df['DebtRatio'] + 1e-6)
df['credit_to_income'] = df['NumberOfOpenCreditLinesAndLoans'] / (df['MonthlyIncome'] + 1e-6)

# Separate features and target
X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

# Split the dataset for training and evaluation (80-20)
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

# Apply SMOTE for class imbalance
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("After SMOTE: Resampled Train shape:", X_train_resampled.shape)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
xgb_model = XGBClassifier(random_state=42)

# Train models
print("Training models...")
rf_model.fit(X_train_resampled, y_train_resampled)
gb_model.fit(X_train_resampled, y_train_resampled)
xgb_model.fit(X_train_resampled, y_train_resampled)
print("Models trained!")

# Make predictions
print("Making predictions...")
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\nEvaluation for {name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Evaluate models
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("Gradient Boosting", y_test, y_pred_gb)
evaluate_model("XGBoost", y_test, y_pred_xgb)

# Confusion Matrix Plotting
def plot_conf_matrix(y_true, y_pred, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Plot for each model
plot_conf_matrix(y_test, y_pred_rf, "Random Forest")
plot_conf_matrix(y_test, y_pred_gb, "Gradient Boosting")
plot_conf_matrix(y_test, y_pred_xgb, "XGBoost")

print("\nAll steps completed successfully!")


# In[ ]:




