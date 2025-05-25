from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlflow.models.signature import infer_signature
from ML_PROJECT.entity.config_entity import ModelEvaluationConfig
from ML_PROJECT.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Prepare MLflow tracking
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Predict
            predicted_qualities = model.predict(test_x)

            # Calculate metrics
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log parameters and metrics to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Prepare input example and signature
            input_example = test_x.iloc[:1]
            signature = infer_signature(test_x, predicted_qualities)

            # Log the model with input example and signature
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name="ElasticnetModel",
                    input_example=input_example,
                    signature=signature
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    input_example=input_example,
                    signature=signature
                )
