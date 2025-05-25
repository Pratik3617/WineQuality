# pipeline
from ML_PROJECT.components.model_evaluation import ModelEvaluation
from ML_PROJECT.config.configuration import ConfigurationManager

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
            model_evaluation_config.log_into_mlflow()
        except Exception as e:
            raise e

