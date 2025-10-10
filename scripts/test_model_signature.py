import mlflow
import pytest
import pandas as pd
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()
import os

# ✅ Set your MLflow Tracking URI
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

@pytest.mark.parametrize("model_name", [
    "reddit_chrome_plugin_model",   # ✅ Use your actual registered model name
])
def test_model_signature(model_name):
    """
    ✅ Test that the latest registered MLflow model can successfully handle 
    input data that matches its saved signature (no manual vectorizer needed).
    """

    client = MlflowClient()

    # ✅ Get the latest version of the model (instead of using deprecated 'stage')
    latest_versions = client.get_latest_versions(model_name)
    assert latest_versions, f"No versions found for model '{model_name}'"
    latest_version = latest_versions[0].version

    # ✅ Load the model by version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # ✅ Create dummy input DataFrame that matches the signature
    # If your signature was inferred from a 'comment' column:
    test_input = pd.DataFrame({"comment": ["This is awesome!", "This is terrible."]})

    try:
        # Run inference
        predictions = model.predict(test_input)

        # ✅ Check basic shape alignment
        assert len(predictions) == len(test_input), (
            f"Expected {len(test_input)} predictions, got {len(predictions)}"
        )

        print(f"✅ Model '{model_name}' version {latest_version} successfully handled signature input.")

    except Exception as e:
        pytest.fail(f"❌ Model signature test failed for '{model_name}' v{latest_version}: {e}")
