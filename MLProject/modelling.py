import pandas as pd
import logging
import time
import argparse
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import mlflow
import mlflow.sklearn


# ===== RESET RUN ID (ANTI CI CONFLICT) =====
os.environ.pop("MLFLOW_RUN_ID", None)

# ===== LOGGING CONFIG =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ===== DATA LOADER =====
def load_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded | shape={df.shape}")
    return df


# ===== DATA PREP =====
def prepare_data(df, test_size, random_state):
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    logger.info(f"Target distribution:\n{y.value_counts()}")

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ===== TRAINING =====
def train_model(X_train, X_test, y_train, y_test, params):
    logger.info("=" * 60)
    logger.info("TRAINING GRADIENT BOOSTING")
    logger.info("=" * 60)

    with mlflow.start_run(run_name="GradientBoosting") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info(f"Experiment ID: {run.info.experiment_id}")

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                random_state=params["random_state"]
            ))
        ])

        start = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start

        y_pred = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "training_time": training_time
        }

        logger.info("Evaluation Metrics:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))

        logger.info("Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(y_test, y_pred)))

        # ===== MLflow LOGGING =====
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        # SAVE RUN ID FOR CI
        with open("mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        logger.info("✓ Model logged at artifacts/model")
        logger.info("✓ Run ID saved")

        return run.info.run_id


# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("diabetes_prediction")

    logger.info(f"MLFLOW_TRACKING_URI = {tracking_uri}")

    data_path = Path("diabetes_prediction_dataset/data_clean.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = load_data(data_path)

    X_train, X_test, y_train, y_test = prepare_data(
        df, args.test_size, args.random_state
    )

    params = vars(args)
    run_id = train_model(
        X_train, X_test, y_train, y_test, params
    )

    logger.info("=" * 60)
    logger.info("TRAINING FINISHED SUCCESSFULLY")
    logger.info(f"Run ID: {run_id}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
