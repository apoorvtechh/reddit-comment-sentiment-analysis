# test_model_loading.py
import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv
import time
load_dotenv()

# Set remote/local MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

@pytest.mark.parametrize("model_name", [
    "reddit_chrome_plugin_model",  # ✅ Only model name, no stage
])
def test_load_latest_model(model_name):
    client = MlflowClient()
    # ⏸️ Wait a few seconds to allow model registration to reflect
    time.sleep(15)

    # ✅ Get all versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    assert versions, f"No versions found for model '{model_name}'"

    # ✅ Pick the latest version by number
    latest_version = max(int(v.version) for v in versions)

    # ✅ Load the latest version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # ✅ Validate
    assert model is not None, f"Failed to load model '{model_name}' version {latest_version}"
    print(f"✅ Model '{model_name}' version {latest_version} loaded successfully.")
