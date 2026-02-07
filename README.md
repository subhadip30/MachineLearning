# Employee Attrition Prediction using Machine Learning (Binary Classification)

## 1. Problem Statement

Employee attrition is a critical issue faced by organizations as it impacts productivity, recruitment costs, and overall workforce stability. 
The objective of this project is to build and evaluate multiple binary classification machine learning models to predict whether an 
employee is likely to leave the organization (Attrition: Yes/No).

The project aims to:

- Develop predictive models to identify employees at risk of attrition.
- Compare the performance of multiple classification algorithms.
- Evaluate models using standard machine learning performance metrics.
- Deploy the solution through an interactive Streamlit application.

The following models are implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN) Classifier  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

## 2. Dataset Description

The dataset used in this project contains employee-related information used to predict attrition status. It includes demographic details, 
job characteristics, performance indicators, and work environment factors.

### Dataset Overview

- Dataset Type: Employee Workforce Data  
- Total Records: 10,000 rows  
- Total Features: 22 columns  
- Target Variable: `attrition` (Yes/No)

### Feature Categories

#### a. Employee Demographics
- employee_id – Unique employee identifier  
- age – Employee age  
- gender – Gender of employee  
- education – Educational qualification  

#### b. Job Information
- department – Department of work  
- job_role – Role designation  
- monthly_salary – Monthly compensation  
- years_at_company – Total years at organization  
- years_in_current_role – Years in current role  

#### c. Performance & Work Metrics
- performance_score – Employee performance rating  
- job_satisfaction – Satisfaction score  
- work_life_balance – Work-life balance rating  
- overtime – Whether employee works overtime  

#### d. Organizational & Work Environment Metrics
- distance_from_home_km – Distance between home and workplace  
- num_projects_last_year – Number of projects handled  
- training_hours_last_year – Training hours completed  
- num_promotions – Number of promotions received  
- last_promotion_years_ago – Years since last promotion  
- stock_option_level – Stock option benefits  
- relationship_satisfaction – Satisfaction with colleagues  
- environment_satisfaction – Work environment satisfaction  

#### e. Target Variable
- attrition – Employee leaving status  
  - Yes → Employee left organization  
  - No → Employee retained

---

## 3. Machine Learning Workflow

The following steps were followed:

1. Dataset upload through Streamlit application.
2. Data preprocessing:
   - Missing value handling
   - Encoding categorical variables
   - Feature scaling
3. Train-test split.
4. Model training using selected algorithm.
5. Model evaluation using:
   - Accuracy
   - AUC Score
   - Precision
   - Recall
   - F1 Score
   - Matthews Correlation Coefficient (MCC)
6. Visualization through:
   - Confusion Matrix
   - Classification Report

---

## 4. Streamlit Application Features

The application includes:

- CSV dataset upload option
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix / classification report visualization

---

## 5. Tools & Technologies

- Python
- Scikit-learn
- XGBoost
- Pandas & NumPy
- Streamlit
- Matplotlib

---

## 6. Expected Outcome

The system provides:

- Comparative analysis of classification models
- Predictive insights into employee attrition risk
- Interactive deployment using Streamlit
- Model performance visualization for academic evaluation

---



