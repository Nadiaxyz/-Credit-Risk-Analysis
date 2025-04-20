# -Credit-Risk-Analysis

This project aims to predict the likelihood of serious delinquency within two years using financial and credit-related features. It leverages machine learning models with SMOTE to address class imbalance and includes performance evaluation.

## Features

• Handles missing values and performs feature engineering  
• Applies SMOTE for class balancing  
• Trains multiple classifiers: Random Forest, Gradient Boosting, XGBoost  
• Evaluates with accuracy, ROC AUC, classification reports, and confusion matrices  

## Dataset

• **Source:** `cs-training.csv`  
• **Target variable:** `SeriousDlqin2yrs`  
• **Features:** Credit history, monthly income, debt ratio, open credit lines, etc.  
• **New features:** `income_to_debt`, `credit_to_income`

## Steps

• Load and clean dataset  
• Feature engineering  
• Train-test split (80-20)  
• Apply SMOTE for balancing  
• Train models (Random Forest, Gradient Boosting, XGBoost)  
• Evaluate performance using metrics and confusion matrices  

## Requirements

• pandas  
• numpy  
• matplotlib  
• seaborn  
• scikit-learn  
• imbalanced-learn  
• xgboost  

Install all with:

```bash
pip install -r requirements.txt
```

## Results

Each model's performance is evaluated using:

• Accuracy  
• ROC AUC Score  
• Classification Report  
• Confusion Matrix (plotted via Seaborn)

## Usage

Run the script:

```bash
python credit_risk_modeling.py
```

**License**

This project is for educational puproses.
