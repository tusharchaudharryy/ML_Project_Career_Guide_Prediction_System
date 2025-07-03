"""
Utility functions for the Career Prediction System
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
from typing import Dict, List, Tuple, Any
from datetime import datetime

from src.config import TECHNICAL_SKILLS, PERSONALITY_TRAITS
from src.logging_config import get_logger

logger = get_logger('utils')

def sanitize_form_data(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize form data to prevent injection attacks
    
    Args:
        form_data (Dict[str, Any]): Raw form data
        
    Returns:
        Dict[str, Any]: Sanitized form data
    """
    sanitized = {}
    
    for key, value in form_data.items():
        if isinstance(value, str):
            # Remove potentially dangerous characters
            sanitized[key] = re.sub(r'[<>"\';()&+]', '', value.strip())
        else:
            sanitized[key] = value
    
    logger.debug(f"Sanitized {len(form_data)} form fields")
    return sanitized

def validate_input(form_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate form input data
    
    Args:
        form_data (Dict[str, Any]): Form data to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Check for required technical skills
        for skill_key, skill_name in TECHNICAL_SKILLS:
            if skill_key not in form_data:
                return False, f"Missing required field: {skill_name}"
            
            value = form_data[skill_key]
            if value is None or value == '':
                return False, f"Empty value for: {skill_name}"
            
            try:
                float_val = float(value)
                if not (1 <= float_val <= 7):
                    return False, f"Invalid range for {skill_name}: must be between 1 and 7"
            except (ValueError, TypeError):
                return False, f"Invalid value for {skill_name}: must be a number"
        
        # Check for required personality traits
        for trait_key, trait_name in PERSONALITY_TRAITS:
            if trait_key not in form_data:
                return False, f"Missing required field: {trait_name}"
            
            value = form_data[trait_key]
            if value is None or value == '':
                return False, f"Empty value for: {trait_name}"
            
            try:
                float_val = float(value)
                if not (0 <= float_val <= 1):
                    return False, f"Invalid range for {trait_name}: must be between 0 and 1"
            except (ValueError, TypeError):
                return False, f"Invalid value for {trait_name}: must be a number"
        
        logger.info("Form validation successful")
        return True, "Validation successful"
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False, f"Validation failed: {str(e)}"

def get_confidence_level(confidence: float) -> str:
    """
    Get confidence level description
    
    Args:
        confidence (float): Confidence value between 0 and 1
        
    Returns:
        str: Confidence level description
    """
    if confidence >= 0.8:
        return "Very High"
    elif confidence >= 0.6:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    elif confidence >= 0.2:
        return "Low"
    else:
        return "Very Low"

def format_predictions(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format prediction results for display
    
    Args:
        prediction_result (Dict[str, Any]): Raw prediction results
        
    Returns:
        Dict[str, Any]: Formatted prediction results
    """
    try:
        # Validate input
        if not isinstance(prediction_result, dict):
            raise ValueError("prediction_result must be a dictionary")
        
        required_keys = ['primary_prediction', 'confidence', 'top_3_predictions', 'all_predictions']
        missing_keys = [key for key in required_keys if key not in prediction_result]
        
        if missing_keys:
            logger.error(f"Missing required keys: {missing_keys}")
            return get_default_formatted_result()
        
        # Format the results
        formatted = {
            'primary_career': prediction_result['primary_prediction'],
            'confidence_percentage': round(prediction_result['confidence'] * 100, 1),
            'confidence_level': get_confidence_level(prediction_result['confidence']),
            'alternative_careers': [],
            'top_matches': [],
            'timestamp': prediction_result.get('timestamp', datetime.now().isoformat())
        }
        
        # Format top 3 predictions
        if prediction_result['top_3_predictions']:
            formatted['top_matches'] = [
                {
                    'career': career,
                    'probability': round(prob * 100, 1),
                    'confidence_level': get_confidence_level(prob)
                }
                for career, prob in prediction_result['top_3_predictions']
            ]
        
        # Format alternative careers (excluding the primary prediction)
        if prediction_result['all_predictions']:
            formatted['alternative_careers'] = [
                {
                    'career': career,
                    'probability': round(prob * 100, 1),
                    'confidence_level': get_confidence_level(prob)
                }
                for career, prob in prediction_result['all_predictions'][1:6]  # Top 5 alternatives
            ]
        
        logger.info("Predictions formatted successfully")
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting predictions: {str(e)}")
        return get_default_formatted_result()

def get_default_formatted_result() -> Dict[str, Any]:
    """
    Return a default formatted result when errors occur
    
    Returns:
        Dict[str, Any]: Default formatted result
    """
    return {
        'primary_career': 'Unable to determine',
        'confidence_percentage': 0,
        'confidence_level': 'Low',
        'alternative_careers': [],
        'top_matches': [],
        'timestamp': datetime.now().isoformat()
    }

def validate_model_predictions(predictions: Dict[str, Any]) -> bool:
    """
    Validate model predictions before returning
    
    Args:
        predictions (Dict[str, Any]): Prediction results to validate
        
    Returns:
        bool: True if predictions are valid
        
    Raises:
        ValueError: If predictions are invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must be a dictionary")
    
    required_keys = ['primary_prediction', 'confidence', 'top_3_predictions']
    for key in required_keys:
        if key not in predictions:
            raise ValueError(f"Missing required prediction key: {key}")
    
    if not isinstance(predictions['confidence'], (int, float)):
        raise ValueError("Confidence must be a number")
    
    if not (0 <= predictions['confidence'] <= 1):
        raise ValueError("Confidence must be between 0 and 1")
    
    if not isinstance(predictions['top_3_predictions'], list):
        raise ValueError("top_3_predictions must be a list")
    
    logger.debug("Model predictions validated successfully")
    return True

def prepare_features_from_form(form_data: Dict[str, Any]) -> List[float]:
    """
    Prepare features list from form data
    
    Args:
        form_data (Dict[str, Any]): Form data
        
    Returns:
        List[float]: Prepared features list
    """
    features = []
    
    # Process technical skills
    for skill_key, _ in TECHNICAL_SKILLS:
        raw_value = form_data.get(skill_key)
        if raw_value is None or raw_value == '':
            logger.warning(f"Missing value for {skill_key}, using default")
            value = 1.0 / 7.0  # Default value
        else:
            try:
                value = float(raw_value) / 7.0  # Normalize to 0-1 range
            except (ValueError, TypeError):
                logger.error(f"Invalid value for {skill_key}: {raw_value}")
                value = 1.0 / 7.0
        features.append(value)
    
    # Process personality traits
    for trait_key, _ in PERSONALITY_TRAITS:
        raw_value = form_data.get(trait_key)
        if raw_value is None or raw_value == '':
            logger.warning(f"Missing value for {trait_key}, using default")
            value = 0.5  # Default value
        else:
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                logger.error(f"Invalid value for {trait_key}: {raw_value}")
                value = 0.5
        features.append(value)
    
    logger.info(f"Prepared {len(features)} features from form data")
    return features
