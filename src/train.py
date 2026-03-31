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

# Preprocessing
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MLflow setup
mlflow.set_experiment("2022BCS0092_experiment")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

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