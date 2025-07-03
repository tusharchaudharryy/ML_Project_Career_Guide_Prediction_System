"""
Logging configuration for the Career Prediction System
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(log_level='INFO'):
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    # Create formatters
    formatter = logging.Formatter(log_format)
    
    # Create handlers
    file_handler = logging.FileHandler(
        log_dir / f'app_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger('career_prediction')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

def get_logger(name):
    """
    Get a logger instance with the specified name
    
    Args:
        name (str): Logger name
        
    Returns:
        logger: Logger instance
    """
    return logging.getLogger(f'career_prediction.{name}')
