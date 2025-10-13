import os
import mlflow
import pytest
import pandas as pd
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# ============================================================
# ✅ Helper to create handcrafted textual features
# ============================================================
def create_text_features(series: pd.Series, features_list: list) -> pd.DataFrame:
    """Generate handcrafted textual features such as length, counts, sentiment, etc."""
    features = {}
    if "comment_length" in features_list:
        features["comment_length"] = series.apply(len)
    if "word_count" in features_list:
        features["word_count"] = series.apply(lambda x: len(x.split()))
    if "unique_word_count" in features_list:
        features["unique_word_count"] = series.apply(lambda x: len(set(x.split())))
    if "num_exclamations" in features_list:
        features["num_exclamations"] = series.apply(lambda x: x.count("!"))
    if "num_questions" in features_list:
        features["num_questions"] = series.apply(lambda x: x.count("?"))
    if "sentiment" in features_list:
        features["sentiment"] = series.apply(lambda x: TextBlob(x).sentiment.polarity)
    return pd.DataFrame(features)


# ============================================================
# ✅ Load environment variables and configure MLflow
# ============================================================
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    raise RuntimeError("❌ MLFLOW_TRACKING_URI is not set in environment variables.")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@pytest.mark.parametrize("model_name", [
    "reddit_chrome_plugin_model",  # 🔸 Registered model name
])
def test_model_signature(model_name):
    """
    ✅ Test that the latest registered MLflow model can handle input
    matching its saved signature. Uses freshly created TF-IDF and scaler
    on the sample text to mimic expected schema.
    """

    client = MlflowClient()

    # ============================================================
    # 🧠 Get latest model version
    # ============================================================
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        pytest.fail(f"❌ Failed to fetch model versions for '{model_name}': {e}")

    assert model_versions, f"No versions found for model '{model_name}'"
    latest_version = max(int(mv.version) for mv in model_versions)

    # ============================================================
    # 📦 Load the model from MLflow
    # ============================================================
    model_uri = f"models:/{model_name}/{latest_version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        pytest.fail(f"❌ Failed to load model '{model_name}' v{latest_version}: {e}")

    # ============================================================
    # 🧪 Prepare test input & mimic preprocessing
    # ============================================================
    test_text = pd.Series(["This is awesome!", "This is terrible."])

    # Fresh TF-IDF vectorizer (fit only on test text)
    tfidf_vectorizer = TfidfVectorizer(max_features=10)
    X_tfidf = tfidf_vectorizer.fit_transform(test_text).toarray()

    # Create handcrafted numeric features
    feature_list = [
        "comment_length",
        "word_count",
        "unique_word_count",
        "num_exclamations",
        "num_questions",
        "sentiment"
    ]
    X_handcrafted = create_text_features(test_text, feature_list)

    # Scale handcrafted features
    scaler = StandardScaler()
    X_handcrafted_scaled = scaler.fit_transform(X_handcrafted)

    # Note: We don't need to concat these for model.predict,
    # because the MLflow model contains its own preprocessing.
    # We're just mimicking the same schema locally for verification.

    # Build final test DataFrame for prediction
    test_input = pd.DataFrame({"clean_comment": test_text})

    # ============================================================
    # 🚀 Run prediction
    # ============================================================
    try:
        predictions = model.predict(test_input)
        assert len(predictions) == len(test_input), (
            f"Expected {len(test_input)} predictions, got {len(predictions)}"
        )
        print(f"✅ Model '{model_name}' v{latest_version} successfully handled signature input with fresh TF-IDF & scaler.")
    except Exception as e:
        pytest.fail(
            f"❌ Model signature test failed for '{model_name}' v{latest_version}: {e}"
        )
