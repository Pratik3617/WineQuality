# pipeline
from ML_PROJECT.components.data_transformation import DataTransformation
from ML_PROJECT.components.model_trainer import ModelTrainer
from ML_PROJECT.config.configuration import ConfigurationManager
from pathlib import Path

class ModelTrainerPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer_config = ModelTrainer(config=model_trainer_config)
            model_trainer_config.train()
        except Exception as e:
            raise e

