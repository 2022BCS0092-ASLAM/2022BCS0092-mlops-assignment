import argparse
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train wine quality classifier")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "lr"])
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--feature_subset", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    # Load dataset
    df = pd.read_csv("data/winequality-red.csv", sep=";")
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")

    # Convert to binary classification
    df["quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)
    print("Class distribution:\n", df["quality"].value_counts())

    # Mandatory feature-selection run support
    if args.feature_subset == 1:
        selected_features = ["alcohol", "pH", "sulphates"]
        X = df[selected_features]
    else:
        selected_features = [col for col in df.columns if col != "quality"]
        X = df.drop("quality", axis=1)

    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    if args.model == "rf":
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=args.random_state,
        )
    else:
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=args.random_state,
        )

    mlflow.set_experiment("2022BCS0092_experiment")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print("Accuracy:", acc)
        print("F1 Score:", f1)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        mlflow.log_param("model", args.model)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("feature_subset", args.feature_subset)
        mlflow.log_param("selected_features", ",".join(selected_features))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "name": "Mohammed Aslam",
            "roll_no": "2022BCS0092",
            "model": args.model,
            "feature_subset": args.feature_subset,
            "selected_features": selected_features,
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print("Training complete")


if __name__ == "__main__":
    main()