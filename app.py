import os
import sys
import traceback
from datetime import datetime

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from dotenv import load_dotenv

# Project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import (
    MODEL_PATH,
    TECHNICAL_SKILLS,
    PERSONALITY_TRAITS,
    CAREER_CATEGORIES,
    SECRET_KEY,
    DEBUG,
)
from src.predictor import CareerPredictor
from src.utils import (
    sanitize_form_data,
    validate_input,
    format_predictions,
    validate_model_predictions,
    prepare_features_from_form,
)
from src.logging_config import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger('app')

def ensure_directories():
    """Ensure required directories exist."""
    directories = ['models', 'data', 'logs', 'static/css', 'static/js', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Required directories ensured")

ensure_directories()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['DEBUG'] = DEBUG

# Inject global template variables
@app.context_processor
def inject_globals():
    return dict(
        technical_skills=TECHNICAL_SKILLS,
        personality_traits=PERSONALITY_TRAITS
    )

# Initialize predictor
predictor = CareerPredictor()

# Model initialization
try:
    if predictor.load_model():
        logger.info("Model loaded successfully during app initialization")
    else:
        logger.warning("Model could not be loaded during app initialization")
except Exception as e:
    logger.error(f"Error during model initialization: {e}")

@app.route('/')
def index():
    """Render the main prediction form."""
    try:
        logger.info("Rendering index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return render_template('error.html', error="Page could not be loaded"), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the web form."""
    try:
        logger.info("Prediction request received")
        form_data = sanitize_form_data(request.form.to_dict())
        logger.debug(f"Form data received: {len(form_data)} fields")

        is_valid, validation_message = validate_input(form_data)
        if not is_valid:
            logger.warning(f"Validation failed: {validation_message}")
            flash(f'Validation Error: {validation_message}', 'error')
            return redirect(url_for('index'))

        features = prepare_features_from_form(form_data)
        logger.info(f"Features prepared: {len(features)} features")

        prediction_result = predictor.predict(features)
        validate_model_predictions(prediction_result)
        formatted_result = format_predictions(prediction_result)

        logger.info(f"Prediction successful: {formatted_result['primary_career']}")
        return render_template(
            'result.html',
            prediction=formatted_result,
            form_data=form_data
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        flash(f'An error occurred during prediction: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction requests."""
    try:
        logger.info("API prediction request received")
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON format'}), 400

        sanitized_data = sanitize_form_data(data)
        is_valid, validation_message = validate_input(sanitized_data)
        if not is_valid:
            logger.warning(f"API validation failed: {validation_message}")
            return jsonify({'error': validation_message}), 400

        features = prepare_features_from_form(sanitized_data)
        prediction_result = predictor.predict(features)
        validate_model_predictions(prediction_result)
        formatted_result = format_predictions(prediction_result)

        logger.info(f"API prediction successful: {formatted_result['primary_career']}")
        return jsonify({
            'success': True,
            'prediction': formatted_result
        }), 200
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        model_info = predictor.get_model_info()
        now = datetime.now().isoformat()
        return jsonify({
            'status': 'healthy',
            'model_status': model_info.get('status', 'unknown'),
            'model_loaded_at': model_info.get('loaded_at', 'unknown'),
            'timestamp': now
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/model-info')
def model_info():
    """Get model information."""
    try:
        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    logger.warning(f"404 error: {request.url}")
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"500 error: {error}")
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    logger.info("Starting Career Prediction System")
    app.run(debug=DEBUG, host='0.0.0.0', port=3000)
