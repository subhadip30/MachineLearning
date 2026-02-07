
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from xgboost import XGBClassifier

# ------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------
df = pd.read_csv("emp_data.csv")
df.drop(columns=['employee_id'],inplace=True)


TARGET_COLUMN = "attrition"   

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Convert categorical target to binary if needed
if y.dtype == "object":
    y = y.map({"Yes": 1, "No": 0})

# ------------------------------------------------------
# 2. Feature Types
# ------------------------------------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

# ------------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------
# 4. Models
# ------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
}

# ------------------------------------------------------
# 5. Training & Evaluation
# ------------------------------------------------------
results = []

for name, model in models.items():

    # Special handling for GaussianNB
    if name == "Naive Bayes (Gaussian)":
        X_train_prep = preprocessor.fit_transform(X_train).toarray()
        X_test_prep = preprocessor.transform(X_test).toarray()

        model.fit(X_train_prep, y_train)

        y_pred = model.predict(X_test_prep)
        y_prob = model.predict_proba(X_test_prep)[:, 1]
    else:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # For AUC
        if hasattr(model, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

# ------------------------------------------------------
# 6. Results Summary
# ------------------------------------------------------
results_df = pd.DataFrame(results)
print("\nModel Evaluation Summary\n")
print(results_df.sort_values(by="AUC", ascending=False))
