#!/usr/bin/env python3
"""
CUDA Kernel Execution Time Prediction Model Training Script

This script trains an XGBoost model to predict CUDA kernel execution times
based on kernel launch parameters and hardware characteristics.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import onnx
import onnxmltools
from onnxmltools.utils import save_model
import os
import argparse
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KernelExecutionPredictor:
    """ML model for predicting CUDA kernel execution times."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'grid_x', 'grid_y', 'grid_z',
            'block_x', 'block_y', 'block_z',
            'shared_mem_kb',
            'input_tensor_volume',
            'operation_complexity_score',
            'arithmetic_intensity',
            'parallelism_degree',
            'memory_access_pattern_score',
            'recent_avg_execution_time_ms',
            'queue_depth_penalty',
            'compute_capability_major',
            'compute_capability_minor',
            'total_global_memory_gb',
            'multiprocessor_count'
        ]
    
    def generate_synthetic_data(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for kernel execution times."""
        logger.info(f"Generating {num_samples} synthetic training samples...")
        
        # Generate random kernel parameters
        data = {
            'grid_x': np.random.randint(1, 1000, num_samples),
            'grid_y': np.random.randint(1, 100, num_samples),
            'grid_z': np.random.randint(1, 100, num_samples),
            'block_x': np.random.randint(32, 1025, num_samples),
            'block_y': np.random.randint(1, 33, num_samples),
            'block_z': np.random.randint(1, 33, num_samples),
            'shared_mem_kb': np.random.randint(0, 49, num_samples),  # 0-48KB
            'input_tensor_volume': np.random.randint(1000, 10000000, num_samples),
            'operation_complexity_score': np.random.uniform(0.1, 1.0, num_samples),
            'arithmetic_intensity': np.random.uniform(0.1, 10.0, num_samples),
            'parallelism_degree': np.random.uniform(0.1, 5.0, num_samples),
            'memory_access_pattern_score': np.random.uniform(0.1, 1.0, num_samples),
            'recent_avg_execution_time_ms': np.random.uniform(0.1, 100.0, num_samples),
            'queue_depth_penalty': np.random.uniform(0.0, 10.0, num_samples),
            'compute_capability_major': np.random.choice([6, 7, 8], num_samples),
            'compute_capability_minor': np.random.choice([0, 1, 2, 5], num_samples),
            'total_global_memory_gb': np.random.choice([4, 8, 16, 24, 32], num_samples),
            'multiprocessor_count': np.random.choice([20, 40, 60, 80, 100], num_samples)
        }
        
        # Create feature matrix
        X = np.column_stack([data[name] for name in self.feature_names])
        
        # Generate synthetic execution times based on features
        # This is a simplified model - in practice, you'd use real execution data
        y = self._generate_execution_times(X)
        
        return X, y
    
    def _generate_execution_times(self, X: np.ndarray) -> np.ndarray:
        """Generate synthetic execution times based on kernel features."""
        # Extract features
        grid_volume = X[:, 0] * X[:, 1] * X[:, 2]
        block_volume = X[:, 3] * X[:, 4] * X[:, 5]
        total_threads = grid_volume * block_volume
        shared_mem = X[:, 6]
        tensor_volume = X[:, 7]
        complexity = X[:, 8]
        arithmetic_intensity = X[:, 9]
        
        # Base execution time model
        base_time = (total_threads / 1000000.0) * complexity
        
        # Memory access penalty
        memory_penalty = (tensor_volume / 1000000.0) * 0.1
        
        # Shared memory benefit
        shared_mem_benefit = np.maximum(0, (48 - shared_mem) / 48.0) * 0.2
        
        # Arithmetic intensity factor
        arithmetic_factor = np.maximum(0.1, arithmetic_intensity / 5.0)
        
        # Add some randomness
        noise = np.random.normal(0, 0.1, len(X))
        
        # Final execution time in milliseconds
        execution_time = (base_time + memory_penalty - shared_mem_benefit) * arithmetic_factor + noise
        execution_time = np.maximum(0.1, execution_time)  # Ensure positive times
        
        return execution_time
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model."""
        logger.info("Training XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        logger.info(f"Training MAE: {train_mae:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")
        logger.info(f"Training R²: {train_r2:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def save_model(self, model_path: str, onnx_path: str) -> None:
        """Save the trained model in multiple formats."""
        logger.info(f"Saving model to {model_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save XGBoost model
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Convert to ONNX format
        logger.info(f"Converting to ONNX format: {onnx_path}")
        try:
            # Create ONNX model
            initial_type = [('float_input', onnx.FloatTensorType([None, len(self.feature_names)]))]
            onx = onnxmltools.convert_xgboost(self.model, initial_types=initial_type)
            
            # Save ONNX model
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            with open(onnx_path, "wb") as f:
                f.write(onx.SerializeToString())
            
            logger.info("Model saved successfully in both XGBoost and ONNX formats")
            
        except Exception as e:
            logger.warning(f"Failed to convert to ONNX format: {e}")
            logger.info("Model saved in XGBoost format only")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        logger.info(f"Loading model from {model_path}")
        
        self.model = joblib.load(model_path)
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def main():
    parser = argparse.ArgumentParser(description='Train CUDA kernel execution time prediction model')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--model-path', type=str, default='models/kernel_predictor.pkl', help='Path to save XGBoost model')
    parser.add_argument('--onnx-path', type=str, default='models/kernel_predictor.onnx', help='Path to save ONNX model')
    parser.add_argument('--load-existing', action='store_true', help='Load existing model instead of training')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = KernelExecutionPredictor()
    
    if args.load_existing:
        # Load existing model
        predictor.load_model(args.model_path)
        logger.info("Model loaded successfully")
    else:
        # Generate training data
        X, y = predictor.generate_synthetic_data(args.num_samples)
        
        # Train model
        predictor.train(X, y)
        
        # Save model
        predictor.save_model(args.model_path, args.onnx_path)
    
    # Test prediction
    logger.info("Testing model prediction...")
    test_features = np.random.rand(5, len(predictor.feature_names))
    predictions = predictor.predict(test_features)
    
    logger.info("Sample predictions:")
    for i, pred in enumerate(predictions):
        logger.info(f"  Sample {i+1}: {pred:.4f} ms")

if __name__ == "__main__":
    main() 