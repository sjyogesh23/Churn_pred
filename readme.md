# Telco Customer Churn Prediction

This project demonstrates how to predict customer churn in a telecommunications company using machine learning models. By leveraging customer data, the goal is to identify customers likely to churn, enabling businesses to take proactive actions to retain valuable customers.

## Project Overview

The project includes:
- Data preprocessing and exploration.
- Machine learning model development using Decision Tree, Gradient Boosting, Random Forest, and XGBoost.
- Model evaluation using various performance metrics (accuracy, precision, recall).
- Deployment of the best-performing model via a Streamlit web application for real-time churn prediction.

### Key Steps in the Project:

1. **Data Preprocessing**:
   - Loaded customer data and cleaned the dataset (e.g., handling missing values and converting the `TotalCharges` column to numeric).
   - Performed exploratory data analysis (EDA) to identify key trends and correlations within the data.
   
2. **Feature Engineering**:
   - Converted categorical columns (e.g., "SeniorCitizen", "Contract") using Label Encoding.
   
3. **Model Training**:
   - Split the data into training and testing sets (80%/20%).
   - Resampled the training data using SMOTE to handle class imbalance.
   - Trained four different models: Decision Tree, Gradient Boosting, XGBoost, and Random Forest.
   
4. **Model Evaluation**:
   - Evaluated the models using cross-validation and selected the best model based on its performance.
   - Used metrics like accuracy, precision, recall, and confusion matrix for evaluation.
   
5. **Model Deployment**:
   - Saved the trained model and encoder using `pickle`.
   - Created a Streamlit web application where users can input customer details and predict the likelihood of churn.

## Project Details

- **Notebook:** [Telco Customer Churn Prediction - Colab](https://colab.research.google.com/drive/13NohKopanWAXnT02RhsgOsGy3X96T7jf?authuser=0#scrollTo=JfXv4iefNPGq)
- **Streamlit Web Application:** A deployed app that predicts customer churn based on input features.

## Dataset

The dataset used for this project is the **Telco Customer Churn** dataset, which is available on Kaggle. You can download the dataset using the link below:

- **Dataset Link:** [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset includes the following columns:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, etc.

For a complete list of columns and data information, refer to the dataset itself.

## Requirements

To run the project locally, install the required dependencies using:

```bash
pip install -r requirements.txt
