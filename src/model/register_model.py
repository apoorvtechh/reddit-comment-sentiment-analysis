import os
import json
import logging
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# -------------------------------------------------
# Load environment variables (.env should have MLFLOW_TRACKING_URI)
# -------------------------------------------------
load_dotenv()

# 🔑 Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("dvc-pipeline-runs-Reddit-sentiments-analysis")

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

if not logger.handlers:  # Prevent duplicate logs if re-run
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_registration_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def load_model_info(file_path: str) -> dict:
    """
    Load the model info (run_id and model_path) from a JSON file.
    This file is typically created after training (e.g., experiment_info.json).
    """
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("Model info file not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading model info: %s", e)
        raise

# -------------------------------------------------
# Model Registration
# -------------------------------------------------
def register_model(model_name: str, model_info: dict):
    """
    Register the model in MLflow Model Registry and transition it to 'Staging'.
    """
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug("Model URI to register: %s", model_uri)

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"✅ Model '{model_name}' registered with version {model_version.version}")

        # Transition model to 'Staging'
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f"🚀 Model '{model_name}' version {model_version.version} transitioned to STAGING")

    except Exception as e:
        logger.error("Error during model registration: %s", e)
        raise

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    try:
        # Path to the model info file generated during training
        model_info_path = "experiment_info.json"
        model_info = load_model_info(model_info_path)

        # Name for the model in MLflow Model Registry
        model_name = "reddit_chrome_plugin_model"  # 🔸 Change this if needed

        # Register and transition the model
        register_model(model_name, model_info)

    except Exception as e:
        logger.error("Failed to complete model registration: %s", e)
        print(f"Error: {e}")

# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    main()
