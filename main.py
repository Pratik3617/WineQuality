from ML_PROJECT import logger
from ML_PROJECT.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ML_PROJECT.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from ML_PROJECT.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from ML_PROJECT.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from ML_PROJECT.pipeline.stage_05_data_evaluation import ModelEvaluationPipeline

STAGE_01_NAME = "Data Ingestion stage"
STAGE_02_NAME = "Data Validation stage"
STAGE_03_NAME = "Data Transformation stage"
STAGE_04_NAME = "Model Trainer Stage"
STAGE_05_NAME = "Model Evaluation Stage"

import mlflow
print("Tracking URI:", mlflow.get_tracking_uri())


if __name__ == '__main__':
    try:
        logger.info(f">>>>> {STAGE_01_NAME} started <<<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> {STAGE_01_NAME} completed <<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f">>>>> {STAGE_02_NAME} started <<<<<<")
        data_validation = DataValidationTrainingPipeline()
        data_validation.main()
        logger.info(f">>>>>> {STAGE_02_NAME} completed <<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f">>>>> {STAGE_03_NAME} started <<<<<<")
        data_transformation = DataTransformationTrainingPipeline()
        data_transformation.main()
        logger.info(f">>>>>> {STAGE_03_NAME} completed <<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f">>>>> {STAGE_04_NAME} started <<<<<<")
        model_trainer = ModelTrainerPipeline()
        model_trainer.main()
        logger.info(f">>>>>> {STAGE_04_NAME} completed <<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f">>>>> {STAGE_05_NAME} started <<<<<<")
        model_evaulation = ModelEvaluationPipeline()
        model_evaulation.main()
        logger.info(f">>>>>> {STAGE_05_NAME} completed <<<<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise e

    


