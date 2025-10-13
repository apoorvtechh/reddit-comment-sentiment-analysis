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
    Register the model in MLflow Model Registry.
    (✅ Removed stage transition — only register the model version now)
    """
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug("Model URI to register: %s", model_uri)

        # ✅ Register model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"✅ Model '{model_name}' registered with version {model_version.version}")

        # ❌ Removed: Stage transition to 'Staging'
        # Originally used:
        # client = MlflowClient()
        # client.transition_model_version_stage(
        #     name=model_name,
        #     version=model_version.version,
        #     stage="Staging"
        # )
        # logger.info(f"🚀 Model '{model_name}' version {model_version.version} transitioned to STAGING")
        #
        # 📝 Reason for removal:
        # Stage transition API was failing with repeated 500 errors on your MLflow server,
        # and is also deprecated as of MLflow 2.9.0. We now only register the model version.

    except Exception as e:
        logger.error("Error during model registration: %s", e)# added
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
        model_name = "reddit_chrome_plugin_model"  
        # Register the model (without stage transition)
        register_model(model_name, model_info)

    except Exception as e:
        logger.error("Failed to complete model registration: %s", e)
        print(f"Error: {e}")

# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    main()
