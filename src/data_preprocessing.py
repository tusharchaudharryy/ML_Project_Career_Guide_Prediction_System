"""
Data Preprocessing for Career Prediction System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

from src.config import MODEL_PATH, TECHNICAL_SKILLS, PERSONALITY_TRAITS, CAREER_CATEGORIES, DATA_PATH

from src.logging_config import get_logger

logger = get_logger('data_preprocessing')

class DataPreprocessor:
    """
    Class to handle data preprocessing for career prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self, file_path=DATA_PATH):
        """
        Load data from file
        
        Args:
            file_path (str): Path to data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading data from: {file_path}")
            
            if not os.path.exists(file_path):
                logger.warning(f"Data file not found: {file_path}")
                return None
            
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully: {data.shape}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def clean_data(self, data):
        """
        Clean and validate data
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Cleaning data")
        
        # Make a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Remove duplicates
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_data)
        
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        missing_counts = cleaned_data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Fill missing values
            for column in cleaned_data.columns:
                if cleaned_data[column].dtype in ['int64', 'float64']:
                    cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].mean())
                else:
                    cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].mode()[0])
        
        # Validate data ranges
        self.validate_data_ranges(cleaned_data)
        
        logger.info(f"Data cleaning completed: {cleaned_data.shape}")
        return cleaned_data
    
    def validate_data_ranges(self, data):
        """
        Validate data ranges for technical skills and personality traits
        
        Args:
            data (pd.DataFrame): Data to validate
        """
        logger.info("Validating data ranges")
        
        # Check technical skills (should be 1-7)
        for skill_key, skill_name in TECHNICAL_SKILLS:
            if skill_key in data.columns:
                min_val = data[skill_key].min()
                max_val = data[skill_key].max()
                
                if min_val < 1 or max_val > 7:
                    logger.warning(f"Invalid range for {skill_name}: {min_val}-{max_val} (expected 1-7)")
                    # Clip values to valid range
                    data[skill_key] = data[skill_key].clip(1, 7)
        
        # Check personality traits (should be 0-1)
        for trait_key, trait_name in PERSONALITY_TRAITS:
            if trait_key in data.columns:
                min_val = data[trait_key].min()
                max_val = data[trait_key].max()
                
                if min_val < 0 or max_val > 1:
                    logger.warning(f"Invalid range for {trait_name}: {min_val}-{max_val} (expected 0-1)")
                    # Clip values to valid range
                    data[trait_key] = data[trait_key].clip(0, 1)
    
    def normalize_features(self, data):
        """
        Normalize technical skills to 0-1 range
        
        Args:
            data (pd.DataFrame): Data to normalize
            
        Returns:
            pd.DataFrame: Normalized data
        """
        logger.info("Normalizing features")
        
        normalized_data = data.copy()
        
        # Normalize technical skills from 1-7 to 0-1
        for skill_key, _ in TECHNICAL_SKILLS:
            if skill_key in normalized_data.columns:
                normalized_data[skill_key] = (normalized_data[skill_key] - 1) / 6
        
        return normalized_data
    
    def encode_labels(self, data, target_column='career'):
        """
        Encode categorical labels
        
        Args:
            data (pd.DataFrame): Data with categorical labels
            target_column (str): Name of target column
            
        Returns:
            tuple: (data, encoded_labels)
        """
        logger.info(f"Encoding labels for column: {target_column}")
        
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return data, None
        
        encoded_labels = self.label_encoder.fit_transform(data[target_column])
        
        logger.info(f"Encoded {len(set(encoded_labels))} unique labels")
        
        return data, encoded_labels
    
    def prepare_features(self, data, target_column='career'):
        """
        Prepare features for model training
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Name of target column
            
        Returns:
            tuple: (X, y) features and targets
        """
        logger.info("Preparing features for model training")
        
        # Get feature columns
        feature_columns = [skill[0] for skill in TECHNICAL_SKILLS] + [trait[0] for trait in PERSONALITY_TRAITS]
        
        # Check if all feature columns exist
        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing feature columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Extract features
        X = data[feature_columns].values
        
        # Extract target
        if target_column in data.columns:
            y = data[target_column].values
        else:
            logger.warning(f"Target column '{target_column}' not found")
            y = None
        
        logger.info(f"Features prepared: X shape {X.shape}, y shape {y.shape if y is not None else 'None'}")
        
        return X, y
    
    def preprocess_pipeline(self, data_path=None, target_column='career'):
        """
        Complete preprocessing pipeline
        
        Args:
            data_path (str, optional): Path to data file
            target_column (str): Name of target column
            
        Returns:
            tuple: (X, y) preprocessed features and targets
        """
        try:
            logger.info("Starting preprocessing pipeline")
            
            # Load data
            data = self.load_data(data_path or DATA_PATH)
            if data is None:
                raise ValueError("Could not load data")
            
            # Clean data
            cleaned_data = self.clean_data(data)
            
            # Normalize features
            normalized_data = self.normalize_features(cleaned_data)
            
            # Prepare features
            X, y = self.prepare_features(normalized_data, target_column)
            
            logger.info("Preprocessing pipeline completed successfully")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise

def main():
    """
    Main function to run data preprocessing
    """
    logger.info("Starting data preprocessing script")
    
    preprocessor = DataPreprocessor()
    
    try:
        X, y = preprocessor.preprocess_pipeline()
        
        print("Data Preprocessing Results:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape if y is not None else 'None'}")
        
        if y is not None:
            unique_labels = len(set(y))
            print(f"Unique labels: {unique_labels}")
        
        logger.info("Data preprocessing script completed successfully")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
