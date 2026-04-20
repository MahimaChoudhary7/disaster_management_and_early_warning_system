import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_PATH = "model/model.pkl"
RANDOM_STATE = 42


def generate_synthetic_dataset(n_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(RANDOM_STATE)

    temperature = np.random.uniform(10, 50, n_samples)
    humidity = np.random.uniform(20, 100, n_samples)
    rainfall = np.random.uniform(0, 300, n_samples)
    wind_speed = np.random.uniform(0, 150, n_samples)
    regions = np.random.choice(["North", "South", "East", "West"], n_samples)

    disaster_types = []
    for temp, hum, rain, wind in zip(temperature, humidity, rainfall, wind_speed):
        if rain > 200 and wind < 80:
            disaster_types.append("Flood")
        elif wind > 100 and rain > 150:
            disaster_types.append("Cyclone")
        elif hum > 80 and temp > 35:
            disaster_types.append("Heatwave")
        else:
            disaster_types.append("No Disaster")

    disaster_types = [
        label
        if np.random.rand() > 0.05
        else np.random.choice(["Flood", "Cyclone", "Heatwave", "No Disaster"])
        for label in disaster_types
    ]

    return pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "wind_speed": wind_speed,
            "region": regions,
            "disaster_type": disaster_types,
        }
    )


def build_pipeline() -> Pipeline:
    numeric_features = ["temperature", "humidity", "rainfall", "wind_speed"]
    categorical_features = ["region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_save_model() -> dict:
    data = generate_synthetic_dataset()
    X = data.drop(columns="disaster_type")
    y = data["disaster_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels = sorted(y.unique())

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    confusion_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=labels,
        columns=labels,
    )

    artifact = {
        "model": pipeline,
        "metrics": metrics,
        "classification_report": report_df,
        "confusion_matrix": confusion_df,
        "labels": labels,
        "dataset_size": len(data),
        "features": list(X.columns),
        "target_name": "disaster_type",
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    return artifact


if __name__ == "__main__":
    artifact = train_and_save_model()
    print("Model training complete.")
    print(f"Accuracy: {artifact['metrics']['accuracy'] * 100:.2f}%")
    print("\nClassification report:")
    print(artifact["classification_report"].round(4).to_string())
    print(f"\nModel saved successfully at: {MODEL_PATH}")
