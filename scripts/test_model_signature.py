import mlflow
import pytest
import pandas as pd
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os

# ✅ Load environment variables (MLFLOW_TRACKING_URI, etc.)
load_dotenv()

# ✅ Set MLflow Tracking URI
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

@pytest.mark.parametrize("model_name", [
    "reddit_chrome_plugin_model",  # 🔸 Registered model name
])
def test_model_signature(model_name):
    """
    ✅ Test that the latest registered MLflow model can handle input 
    that matches its saved signature (e.g., 'clean_comment').
    """

    client = MlflowClient()

    # ✅ Get latest model version
    latest_versions = client.get_latest_versions(model_name)
    assert latest_versions, f"No versions found for model '{model_name}'"
    latest_version = latest_versions[0].version

    # ✅ Load model from MLflow Registry
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # ✅ Match the expected schema (based on error logs: ['clean_comment'])
    test_input = pd.DataFrame({"clean_comment": ["This is awesome!", "This is terrible."]})

    try:
        predictions = model.predict(test_input)
        assert len(predictions) == len(test_input), (
            f"Expected {len(test_input)} predictions, got {len(predictions)}"
        )
        print(f"✅ Model '{model_name}' v{latest_version} successfully handled signature input.")

    except Exception as e:
        pytest.fail(f"❌ Model signature test failed for '{model_name}' v{latest_version}: {e}")
