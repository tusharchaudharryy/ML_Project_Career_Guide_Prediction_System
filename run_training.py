# run_training.py
from src.model_training import ModelTrainer
from src.logging_config import setup_logging

def main():
    setup_logging()
    trainer = ModelTrainer()
    
    # Train with sample data (or load real data)
    results = trainer.train_and_save()
    print(f"Model trained successfully with accuracy: {results['accuracy']:.3f}")

if __name__ == "__main__":
    main()
