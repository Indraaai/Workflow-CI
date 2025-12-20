import pandas as pd
import numpy as np
import logging
import time
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from pathlib import Path

# Clear any existing MLflow run ID to avoid conflicts
if 'MLFLOW_RUN_ID' in os.environ:
    del os.environ['MLFLOW_RUN_ID']

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


def prepare_data(df, test_size=0.2, random_state=42):
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
            X, y, test_size=test_size, random_state=random_state, stratify=y
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


def train_gradient_boosting(X_train, X_test, y_train, y_test, scaler, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """
    Train Gradient Boosting Classifier dengan MLflow logging
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Gradient Boosting Classifier")
        logger.info(f"{'='*60}")
        
        # Check if there's already an active run (from mlflow run command)
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"Using existing MLflow run: {active_run.info.run_id}")
            run_context = None  # Don't create new run
        else:
            logger.info("Creating new MLflow run...")
            run_context = mlflow.start_run(run_name="Gradient_Boosting_Classifier")
        
        try:
            # Initialize model dengan parameter dari CLI
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
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
            
            # Log model secara eksplisit (jangan rely on autolog saja)
            logger.info("Logging model to MLflow...")
            mlflow.sklearn.log_model(model, "model")
            logger.info("✓ Model logged successfully")
            
            # Log scaler juga
            mlflow.sklearn.log_model(scaler, "scaler")
            logger.info("✓ Scaler logged successfully")
            
            # Log additional metrics secara eksplisit
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("training_time", training_time)
            logger.info("✓ Metrics logged successfully")
            
            # Get run info
            run = mlflow.active_run()
            run_id = run.info.run_id
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Artifact URI: {run.info.artifact_uri}")
            
            # Save run_id to file for CI/CD pipeline
            with open('mlflow_run_id.txt', 'w') as f:
                f.write(run_id)
            logger.info(f"Run ID saved to mlflow_run_id.txt")
            
            return {
                'model_name': 'Gradient Boosting',
                'training_time': training_time,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'run_id': run_id
            }
        
        finally:
            # End run only if we created it
            if run_context is not None:
                mlflow.end_run()
                logger.info("MLflow run ended")
            
    except Exception as e:
        logger.error(f"Error training Gradient Boosting: {str(e)}")
        raise


def main():
    """
    Main function untuk menjalankan pipeline ML
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Gradient Boosting model for diabetes prediction')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state (default: 42)')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--max_depth', type=int, default=5, help='Max depth (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Note: When running via `mlflow run .`, MLflow handles tracking URI automatically
        # Only set manually if running standalone
        if not mlflow.active_run():
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("diabetes_prediction")
            logger.info("Set tracking URI manually (standalone mode)")
        else:
            logger.info(f"Using existing MLflow run context: {mlflow.active_run().info.run_id}")
        
        logger.info("="*60)
        logger.info("MEMULAI TRAINING GRADIENT BOOSTING MODEL")
        logger.info("="*60)
        logger.info(f"Parameters:")
        logger.info(f"  test_size: {args.test_size}")
        logger.info(f"  random_state: {args.random_state}")
        logger.info(f"  n_estimators: {args.n_estimators}")
        logger.info(f"  learning_rate: {args.learning_rate}")
        logger.info(f"  max_depth: {args.max_depth}")
        
        # Load data
        data_path = Path("diabetes_prediction_dataset/data_clean.csv")
        if not data_path.exists():
            logger.error(f"File {data_path} tidak ditemukan!")
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        df = load_data(data_path)
        
        # Prepare data dengan scaling
        X_train, X_test, y_train, y_test, scaler = prepare_data(df, args.test_size, args.random_state)
        
        # Train Gradient Boosting model
        result = train_gradient_boosting(
            X_train, X_test, y_train, y_test, scaler,
            args.n_estimators, args.learning_rate, args.max_depth, args.random_state
        )
        
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
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
