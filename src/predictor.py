import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

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
        self.feature_names: List[str] = []
        self.target_classes: List[str] = []
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

            # Validate core model structure
            required_keys = ['model', 'feature_names']
            missing_keys = [key for key in required_keys if key not in model_data]
            if missing_keys:
                logger.error(f"Model file missing required keys: {missing_keys}")
                return False

            # Recover or assign target_classes
            if 'target_classes' in model_data:
                self.target_classes = model_data['target_classes']
            elif 'label_encoder' in model_data:
                # legacy files: reconstruct from saved LabelEncoder
                self.target_classes = list(model_data['label_encoder'].classes_)
            else:
                # fallback to default categories
                self.target_classes = CAREER_CATEGORIES

            # Validate model object
            if not hasattr(model_data['model'], 'predict'):
                logger.error("Loaded object is not a valid model")
                return False

            self.model = model_data['model']
            self.feature_names = model_data['feature_names']

            # Validate feature consistency
            if hasattr(self.model, 'n_features_in_'):
                expected = self.model.n_features_in_
                if len(self.feature_names) != expected:
                    logger.warning(
                        f"Feature count mismatch: expected {expected}, got {len(self.feature_names)}"
                    )

            self.is_loaded = True
            logger.info(
                f"Model loaded successfully – Features: {len(self.feature_names)}, "
                f"Classes: {len(self.target_classes)}"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
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

        arr = np.array(features, dtype=float)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("Features contain invalid values (NaN or Inf)")

        return arr.reshape(1, -1)

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
            X = self.validate_features(features)

            preds = self.model.predict(X)
            pred_idx = int(preds[0])

            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[0]
            else:
                probs = np.ones(len(self.target_classes)) / len(self.target_classes)
                logger.warning("Model doesn't support predict_proba; using uniform probabilities")

            # Bound-check index
            if pred_idx < 0 or pred_idx >= len(self.target_classes):
                logger.error(f"Prediction index {pred_idx} out of range")
                pred_idx = 0

            primary = self.target_classes[pred_idx]
            confidence = float(probs[pred_idx])

            # Top-3 and all predictions
            idx_sorted = np.argsort(probs)[::-1]
            top_3 = [(self.target_classes[i], float(probs[i])) for i in idx_sorted[:3]]
            all_preds = [(self.target_classes[i], float(probs[i])) for i in idx_sorted]

            return {
                'primary_prediction': primary,
                'confidence': confidence,
                'top_3_predictions': top_3,
                'all_predictions': all_preds,
                'raw_prediction': pred_idx,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

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
            info['model_features_in'] = self.model.n_features_in_
        return info
