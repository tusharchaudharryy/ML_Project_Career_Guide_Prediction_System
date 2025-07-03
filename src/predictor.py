"""
Career Prediction Model Handler
Handles model loading, validation, and prediction
"""

import os
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from src.config import MODEL_PATH, TECHNICAL_SKILLS, PERSONALITY_TRAITS, CAREER_CATEGORIES

from src.logging_config import get_logger

logger = get_logger('predictor')

class CareerPredictor:
    """
    Career Prediction Model Handler
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the Career Predictor
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = []
        self.target_classes = []
        self.is_loaded = False
        
        # Expected feature count
        self.expected_features = len(TECHNICAL_SKILLS) + len(PERSONALITY_TRAITS)
        
        logger.info(f"CareerPredictor initialized with model path: {model_path}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the trained model from file
        
        Args:
            model_path (str, optional): Path to model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            path = model_path or self.model_path
            
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False

            logger.info(f"Loading model from: {path}")
            model_data = joblib.load(path)
            
            # Validate model structure
            required_keys = ['model', 'feature_names', 'target_classes']
            missing_keys = [key for key in required_keys if key not in model_data]
            
            if missing_keys:
                logger.error(f"Model file missing required keys: {missing_keys}")
                return False
            
            # Validate model object
            if not hasattr(model_data['model'], 'predict'):
                logger.error("Loaded object is not a valid model")
                return False
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.target_classes = model_data.get('target_classes', CAREER_CATEGORIES)
            
            # Validate feature consistency
            if hasattr(self.model, 'n_features_in_'):
                expected_features = self.model.n_features_in_
                if len(self.feature_names) != expected_features:
                    logger.warning(f"Feature count mismatch: expected {expected_features}, got {len(self.feature_names)}")
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully - Features: {len(self.feature_names)}, Classes: {len(self.target_classes)}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def validate_features(self, features: List[float]) -> np.ndarray:
        """
        Validate and convert features to numpy array
        
        Args:
            features (List[float]): List of feature values
            
        Returns:
            np.ndarray: Validated feature array
            
        Raises:
            ValueError: If features are invalid
        """
        if not isinstance(features, (list, np.ndarray)):
            raise ValueError("Features must be a list or numpy array")
        
        if len(features) != self.expected_features:
            raise ValueError(f"Expected {self.expected_features} features, got {len(features)}")
        
        # Convert to numpy array and validate values
        features_array = np.array(features, dtype=float)
        
        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            raise ValueError("Features contain invalid values (NaN or Inf)")
        
        return features_array.reshape(1, -1)
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Make career prediction based on input features
        
        Args:
            features (List[float]): List of feature values
            
        Returns:
            Dict[str, Any]: Prediction results
            
        Raises:
            RuntimeError: If model is not loaded or prediction fails
        """
        try:
            if not self.is_loaded:
                if not self.load_model():
                    raise RuntimeError("Failed to load model")
            
            logger.info(f"Making prediction with {len(features)} features")
            
            # Validate features
            features_array = self.validate_features(features)
            
            # Make prediction
            prediction = self.model.predict(features_array)
            
            # Get probability predictions if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_array)[0]
                logger.info("Probability predictions available")
            else:
                # Create uniform probabilities if not available
                probabilities = np.ones(len(self.target_classes)) / len(self.target_classes)
                logger.warning("Model doesn't support probability predictions, using uniform distribution")
            
            pred_idx = int(prediction[0])
            
            # Validate prediction index
            if pred_idx >= len(self.target_classes):
                logger.error(f"Prediction index {pred_idx} out of range for {len(self.target_classes)} classes")
                pred_idx = 0
            
            predicted_class = self.target_classes[pred_idx]
            confidence = probabilities[pred_idx]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_predictions = [
                (self.target_classes[i], float(probabilities[i]))
                for i in top_indices
            ]
            
            # Get all predictions sorted by probability
            all_indices = np.argsort(probabilities)[::-1]
            all_predictions = [
                (self.target_classes[i], float(probabilities[i]))
                for i in all_indices
            ]
            
            result = {
                'primary_prediction': predicted_class,
                'confidence': float(confidence),
                'top_3_predictions': top_3_predictions,
                'all_predictions': all_predictions,
                'raw_prediction': pred_idx,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction successful: {predicted_class} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.is_loaded:
            return {'status': 'Model not loaded'}
        
        info = {
            'status': 'Model loaded',
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names),
            'class_count': len(self.target_classes),
            'feature_names': self.feature_names,
            'target_classes': self.target_classes
        }
        
        if hasattr(self.model, 'n_features_in_'):
            info['model_features'] = self.model.n_features_in_
        
        return info
