import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.models
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def normalize_landmarks(df):
    logging.info("starting normalization...")
    df = df.copy()
    x_wrist, y_wrist = df["x1"], df["y1"]
    x_tip, y_tip = df["x13"], df["y13"]
    scale = np.sqrt((x_tip - x_wrist)**2 + (y_tip - y_wrist)**2)
    scale = np.maximum(scale, 1e-6)
    for j in range(1, 22):
        df[f"x{j}"] = (df[f"x{j}"] - x_wrist) / scale
        df[f"y{j}"] = (df[f"y{j}"] - y_wrist) / scale
    return df

def main():
    logging.info("Loading dataset...")
    url = "https://media.githubusercontent.com/media/alaa-alt/ML1_FinalProject/refs/heads/main/hand_landmarks_data.csv"
    df = pd.read_csv(url)

    logging.info("Normalizing landmarks...")
    df = normalize_landmarks(df)

    logging.info("Encoding labels...")
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["label"])
    label_mapping = {str(label): int(idx) for idx, label in enumerate(encoder.classes_)}
    with open("label_mapping.json", "w") as f:
        json.dump(label_mapping, f)

    features = df.drop("label", axis=1)
    labels = df["label"]

    logging.info("Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    mlflow.set_experiment("MLOps-Maze-Navigator")

    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred, average="weighted")
            recall = recall_score(y_val, y_val_pred, average="weighted")
            f1 = f1_score(y_val, y_val_pred, average="weighted")

            logging.info(f"{model_name} - Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")

            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            input_example = X_train.iloc[:1]
            signature = mlflow.models.infer_signature(input_example, model.predict(input_example))

            mlflow.sklearn.log_model(
                model,
                "model",
                input_example=input_example,
                signature=signature
            )

            if model_name == "RandomForest":
                joblib.dump(model, "RF_model.pkl")
                mlflow.log_artifact("RF_model.pkl")
                mlflow.log_artifact("label_mapping.json")

if __name__ == "__main__":
    main()