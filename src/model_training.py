"""
Model Training Script for Career Prediction System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import TECHNICAL_SKILLS, PERSONALITY_TRAITS, CAREER_CATEGORIES, MODEL_PATH
from src.logging_config import get_logger

logger = get_logger('model_training')

class ModelTrainer:
    """
    Class to handle model training for career prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [skill[0] for skill in TECHNICAL_SKILLS] + [trait[0] for trait in PERSONALITY_TRAITS]
        
    def create_sample_data(self, n_samples=1000):
        """
        Create sample data for testing (replace with real data loading)
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Sample data
        """
        logger.info(f"Creating sample data with {n_samples} samples")
        
        np.random.seed(42)
        
        # Generate technical skills (1-7 scale)
        technical_data = {}
        for skill, _ in TECHNICAL_SKILLS:
            technical_data[skill] = np.random.randint(1, 8, n_samples)
        
        # Generate personality traits (0-1 scale)
        personality_data = {}
        for trait, _ in PERSONALITY_TRAITS:
            personality_data[trait] = np.random.uniform(0, 1, n_samples)
        
        # Generate career labels
        career_labels = np.random.choice(CAREER_CATEGORIES, n_samples)
        
        # Combine all data
        data = {**technical_data, **personality_data, 'career': career_labels}
        
        return pd.DataFrame(data)
    
    def prepare_features(self, data):
        """
        Prepare features for training
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            tuple: (X, y) features and targets
        """
        logger.info("Preparing features for training")
        
        # Normalize technical skills to 0-1 range
        for skill, _ in TECHNICAL_SKILLS:
            if skill in data.columns:
                data[skill] = data[skill] / 7.0
        
        # Extract features and target
        X = data[self.feature_names].values
        y = data['career'].values
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def train_model(self, X, y):
        """
        Train the career prediction model
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            dict: Training results
        """
        logger.info("Starting model training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = self.model.score(X_test_scaled, y_test_encoded)
        
        logger.info(f"Model training completed with accuracy: {accuracy:.3f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test_encoded, y_pred),
            'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
        }
    
    def save_model(self, model_path=MODEL_PATH):
        """
        Save the trained model
        
        Args:
            model_path (str): Path to save the model
        """
        logger.info(f"Saving model to: {model_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_classes': list(self.label_encoder.classes_)
        }
        
        # Save model
        joblib.dump(model_data, model_path)
        logger.info("Model saved successfully")
    
    def train_and_save(self, data=None):
        """
        Complete training pipeline
        
        Args:
            data (pd.DataFrame, optional): Training data
        """
        try:
            # Load or create data
            if data is None:
                data = self.create_sample_data()
            
            # Prepare features
            X, y = self.prepare_features(data)
            
            # Train model
            results = self.train_model(X, y)
            
            # Save model
            self.save_model()
            
            logger.info("Training pipeline completed successfully")
            logger.info(f"Model accuracy: {results['accuracy']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    """
    Main function to run model training
    """
    logger.info("Starting model training script")
    
    trainer = ModelTrainer()
    results = trainer.train_and_save()
    
    print("Model Training Results:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    logger.info("Model training script completed")

if __name__ == "__main__":
    main()
