import os
import traceback
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from dotenv import load_dotenv

from src.config import (
    MODEL_PATH, TECHNICAL_SKILLS, PERSONALITY_TRAITS, CAREER_CATEGORIES, SECRET_KEY, DEBUG
)
from src.predictor import CareerPredictor
from src.utils import (
    sanitize_form_data, validate_input, format_predictions, 
    validate_model_predictions, prepare_features_from_form
)
from src.logging_config import setup_logging, get_logger


# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger('app')

def ensure_directories():
    """Ensure required directories exist"""
    directories = ['models', 'data', 'logs', 'static/css', 'static/js']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Required directories ensured")

# Ensure directories exist
ensure_directories()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['DEBUG'] = DEBUG

# Initialize predictor
predictor = CareerPredictor()

# --- Flask 3.x-compatible model initialization ---
try:
    if predictor.load_model():
        logger.info("Model loaded successfully during app initialization")
    else:
        logger.warning("Model could not be loaded during app initialization")
except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}")
# --------------------------------------------------

@app.route('/')
def index():
    """
    Render the main prediction form
    """
    try:
        logger.info("Rendering index page")
        return render_template('index.html', 
                             technical_skills=TECHNICAL_SKILLS,
                             personality_traits=PERSONALITY_TRAITS)
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return render_template('error.html', error="Page could not be loaded"), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    """
    try:
        logger.info("Prediction request received")
        
        # Get and sanitize form data
        form_data = sanitize_form_data(request.form.to_dict())
        logger.debug(f"Form data received: {len(form_data)} fields")
        
        # Validate input
        is_valid, validation_message = validate_input(form_data)
        if not is_valid:
            logger.warning(f"Validation failed: {validation_message}")
            flash(f'Validation Error: {validation_message}', 'error')
            return redirect(url_for('index'))
        
        # Prepare features
        features = prepare_features_from_form(form_data)
        logger.info(f"Features prepared: {len(features)} features")
        
        # Make prediction
        prediction_result = predictor.predict(features)
        
        # Validate prediction result
        validate_model_predictions(prediction_result)
        
        # Format results for display
        formatted_result = format_predictions(prediction_result)
        
        logger.info(f"Prediction successful: {formatted_result['primary_career']}")
        
        return render_template('result.html', 
                             prediction=formatted_result,
                             form_data=form_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for prediction requests
    """
    try:
        logger.info("API prediction request received")
        
        # Get JSON data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        # Validate JSON structure
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON format'}), 400
        
        # Sanitize data
        sanitized_data = sanitize_form_data(data)
        
        # Validate input
        is_valid, validation_message = validate_input(sanitized_data)
        if not is_valid:
            logger.warning(f"API validation failed: {validation_message}")
            return jsonify({'error': validation_message}), 400
        
        # Prepare features
        features = prepare_features_from_form(sanitized_data)
        
        # Make prediction
        prediction_result = predictor.predict(features)
        
        # Validate prediction result
        validate_model_predictions(prediction_result)
        
        # Format results
        formatted_result = format_predictions(prediction_result)
        
        logger.info(f"API prediction successful: {formatted_result['primary_career']}")
        
        return jsonify({
            'success': True,
            'prediction': formatted_result
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    try:
        model_info = predictor.get_model_info()
        return jsonify({
            'status': 'healthy',
            'model_status': model_info['status'],
            'timestamp': predictor.get_model_info().get('timestamp', 'unknown')
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/model-info')
def model_info():
    """
    Get model information
    """
    try:
        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 error: {str(error)}")
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    logger.info("Starting Career Prediction System")
    app.run(debug=DEBUG, host='0.0.0.0', port=5000)
