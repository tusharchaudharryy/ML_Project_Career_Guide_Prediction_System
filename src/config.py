"""
Configuration file for Career Prediction System
Contains all constants and feature definitions
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'career_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'career_data.csv')

# Technical Skills (consistent naming)
TECHNICAL_SKILLS = [
    ('database_fundamentals', 'Database Fundamentals'),
    ('computer_architecture', 'Computer Architecture'),
    ('distributed_computing', 'Distributed Computing'),
    ('cyber_security', 'Cyber Security'),
    ('networking', 'Networking'),
    ('development', 'Software Development'),
    ('programming_skills', 'Programming Skills'),
    ('project_management', 'Project Management'),
    ('computer_forensics', 'Computer Forensics'),
    ('technical_communication', 'Technical Communication'),
    ('ai_ml', 'AI/ML'),
    ('software_engineering', 'Software Engineering'),
    ('business_analysis', 'Business Analysis'),
    ('communication_skills', 'Communication Skills'),
    ('data_science', 'Data Science'),
    ('troubleshooting', 'Troubleshooting'),
    ('graphics_designing', 'Graphics Designing')
]

# Personality Traits (fixed typo)
PERSONALITY_TRAITS = [
    ('openness', 'Openness'),
    ('conscientiousness', 'Conscientiousness'),
    ('extraversion', 'Extraversion'),
    ('agreeableness', 'Agreeableness'),
    ('emotional_range', 'Emotional Range'),
    ('conversation', 'Conversation'),
    ('openness_to_change', 'Openness to Change'),
    ('hedonism', 'Hedonism'),
    ('self_enhancement', 'Self Enhancement'),
    ('self_transcendence', 'Self Transcendence'),
    ('conservation', 'Conservation')
]

# Career categories
CAREER_CATEGORIES = [
    'Software Developer',
    'Data Scientist',
    'System Administrator',
    'Network Engineer',
    'Cybersecurity Analyst',
    'Database Administrator',
    'AI/ML Engineer',
    'Project Manager',
    'Business Analyst',
    'Technical Writer',
    'DevOps Engineer',
    'Quality Assurance',
    'UI/UX Designer',
    'Cloud Architect',
    'Research Scientist'
]

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Flask configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
