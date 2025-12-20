

import pandas as pd
import numpy as np
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from pathlib import Path
import os

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modelling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Konfigurasi MLflow Tracking URI
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

logger.info(f"MLFLOW_TRACKING_URI: {tracking_uri}")


# Set experiment name
EXPERIMENT_NAME = "Diabetes_Prediction_Experiment"
mlflow.set_experiment(EXPERIMENT_NAME)
MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
logger.info(f"MLflow Experiment: {EXPERIMENT_NAME}")


def load_data(file_path):
    """
    Load dataset dari file CSV
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def prepare_data(df):
    """
    Prepare data untuk training dengan scaling
    """
    try:
        logger.info("Preparing data for training...")
        
        # Pisahkan features dan target
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Testing set: {X_test.shape}")
        
        # Scaling data
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Data scaling completed")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise


def train_gradient_boosting(X_train, X_test, y_train, y_test, scaler):
    """
    Train Gradient Boosting Classifier dengan MLflow autolog
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Gradient Boosting Classifier")
        logger.info(f"{'='*60}")
        
        # Enable autolog 
        mlflow.sklearn.autolog()
        
        # Mulai MLflow run
        with mlflow.start_run(run_name="Gradient_Boosting_Classifier"):
            
            # Initialize model dengan parameter seperti di notebook
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            
            # Train model dengan timing
            logger.info("Training Gradient Boosting Classifier...")
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            logger.info(f"Training selesai dalam {training_time:.2f} detik")
            
            # Predict untuk evaluasi (autolog akan handle metrics otomatis)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics untuk logging ke console
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, average='weighted')
            test_recall = recall_score(y_test, y_pred_test, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            
            # Log hasil ke console
            logger.info(f"\n=== Gradient Boosting Performance ===")
            logger.info(f"Training Time: {training_time:.2f} seconds")
            logger.info(f"Train Accuracy: {train_accuracy:.4f}")
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Test Precision: {test_precision:.4f}")
            logger.info(f"Test Recall: {test_recall:.4f}")
            logger.info(f"Test F1-Score: {test_f1:.4f}")
            
            # Classification Report
            logger.info(f"\nClassification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred_test)}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_test)
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"\n{cm}")
            
            # Log scaler secara manual (model sudah otomatis di-log oleh autolog)
            mlflow.sklearn.log_model(scaler, "scaler")
            logger.info("Model dan scaler logged to MLflow successfully (via autolog)")
            
            # Get run info
            run = mlflow.active_run()
            logger.info(f"Run ID: {run.info.run_id}")
            logger.info(f"Artifact URI: {run.info.artifact_uri}")
            
            return {
                'model_name': 'Gradient Boosting',
                'training_time': training_time,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'run_id': run.info.run_id
            }
            
    except Exception as e:
        logger.error(f"Error training Gradient Boosting: {str(e)}")
        raise


def main():
    """
    Main function untuk menjalankan pipeline ML
    """
    try:
        logger.info("="*60)
        logger.info("MEMULAI TRAINING GRADIENT BOOSTING MODEL")
        logger.info("="*60)
        
        # Load data
        data_path = Path("diabetes_prediction_dataset/data_clean.csv")
        if not data_path.exists():
            logger.error(f"File {data_path} tidak ditemukan!")
            return
        
        df = load_data(data_path)
        
        # Prepare data dengan scaling
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        
        # Train Gradient Boosting model
        result = train_gradient_boosting(X_train, X_test, y_train, y_test, scaler)
        
        # Summary results
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Model: {result['model_name']}")
        logger.info(f"Training Time: {result['training_time']:.2f} seconds")
        logger.info(f"Train Accuracy: {result['train_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {result['test_accuracy']:.4f}")
        logger.info(f"Test Precision: {result['test_precision']:.4f}")
        logger.info(f"Test Recall: {result['test_recall']:.4f}")
        logger.info(f"Test F1-Score: {result['test_f1']:.4f}")
        logger.info(f"Run ID: {result['run_id']}")
        logger.info("="*60)
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"Check MLflow UI at: {MLFLOW_TRACKING_URI}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
