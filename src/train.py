import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
import os

# Load dataset
df = pd.read_csv("data/winequality-red.csv", sep=';')

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")

# Convert to classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

print("Class distribution:\n", df['quality'].value_counts())

# Features
X = df.drop('quality', axis=1)
y = df['quality']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",   
    random_state=42
)

# MLflow
mlflow.set_experiment("2022BCS0092_experiment")

with mlflow.start_run():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("Accuracy:", acc)
    print("F1 Score:", f1)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # MLflow logging
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    # Save metrics JSON (MANDATORY)
    os.makedirs("outputs", exist_ok=True)
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "name": "Mohammed Aslam",
        "roll_no": "2022BCS0092"
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f)

print("Training complete")