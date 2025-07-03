import unittest
import numpy as np
import tempfile
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predictor import CareerPredictor

from unittest.mock import patch, MagicMock
from unittest.mock import patch

from predictor import CareerPredictor
from config import TECHNICAL_SKILLS, PERSONALITY_TRAITS


class TestCareerPredictor(unittest.TestCase):
    """
    Test cases for CareerPredictor class
    """
    
    def setUp(self):
        """Set up test cases"""
        self.predictor = CareerPredictor()
        self.test_features = [0.5] * (len(TECHNICAL_SKILLS) + len(PERSONALITY_TRAITS))
    
    def test_init(self):
        """Test predictor initialization"""
        self.assertIsNotNone(self.predictor)
        self.assertEqual(len(self.predictor.feature_names), 0)
        self.assertEqual(len(self.predictor.target_classes), 0)
        self.assertFalse(self.predictor.is_loaded)
    
    def test_validate_features_valid(self):
        """Test feature validation with valid features"""
        features_array = self.predictor.validate_features(self.test_features)
        self.assertEqual(features_array.shape, (1, len(self.test_features)))
    
    def test_validate_features_invalid_count(self):
        """Test feature validation with invalid feature count"""
        with self.assertRaises(ValueError):
            self.predictor.validate_features([0.5, 0.5])  # Too few features
    
    def test_validate_features_invalid_values(self):
        """Test feature validation with invalid values"""
        invalid_features = [np.nan] * len(self.test_features)
        with self.assertRaises(ValueError):
            self.predictor.validate_features(invalid_features)
    
    @patch('joblib.load')
    def test_load_model_success(self, mock_load):
        """Test successful model loading"""
        # Mock model data
        mock_model = MagicMock()
        mock_model.predict = MagicMock()
        
        mock_load.return_value = {
            'model': mock_model,
            'feature_names': ['test_feature'],
            'target_classes': ['test_class']
        }
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            result = self.predictor.load_model()
        
        self.assertTrue(result)
        self.assertTrue(self.predictor.is_loaded)
    
    @patch('os.path.exists', return_value=False)
    def test_load_model_file_not_found(self, mock_exists):
        """Test model loading with missing file"""
        result = self.predictor.load_model()
        self.assertFalse(result)
        self.assertFalse(self.predictor.is_loaded)
    
    def test_predict_model_not_loaded(self):
        """Test prediction when model is not loaded and model loading fails"""
        with patch.object(self.predictor, 'load_model', return_value=False):
            with self.assertRaises(RuntimeError):
                self.predictor.predict([0.5] * (len(TECHNICAL_SKILLS) + len(PERSONALITY_TRAITS)))
    
    @patch('joblib.load')
    def test_predict_success(self, mock_load):
        """Test successful prediction"""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        
        mock_load.return_value = {
            'model': mock_model,
            'feature_names': ['test_feature'] * len(self.test_features),
            'target_classes': ['Software Developer', 'Data Scientist']
        }
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            self.predictor.load_model()
        
        result = self.predictor.predict(self.test_features)
        
        self.assertIsInstance(result, dict)
        self.assertIn('primary_prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('top_3_predictions', result)
    
    def test_get_model_info_not_loaded(self):
        """Test getting model info when model is not loaded"""
        info = self.predictor.get_model_info()
        self.assertEqual(info['status'], 'Model not loaded')
    
    @patch('joblib.load')
    def test_get_model_info_loaded(self, mock_load):
        """Test getting model info when model is loaded"""
        mock_model = MagicMock()
        mock_model.predict = MagicMock()
        
        mock_load.return_value = {
            'model': mock_model,
            'feature_names': ['test_feature'],
            'target_classes': ['test_class']
        }
        
        with patch('os.path.exists', return_value=True):
            self.predictor.load_model()
        
        info = self.predictor.get_model_info()
        
        self.assertEqual(info['status'], 'Model loaded')
        self.assertIn('model_type', info)
        self.assertIn('feature_count', info)

if __name__ == '__main__':
    unittest.main()
