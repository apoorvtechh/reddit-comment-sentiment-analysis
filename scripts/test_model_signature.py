import os
import mlflow
import pytest
import pandas as pd
import numpy as np
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
    by recreating TF-IDF (1-3 ngrams) and handcrafted features,
    scaling them, and passing the combined numeric array to the model.
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
        pytest.fail(f"❌ Failed to load model '{model_name}' v{latest_version} from MLflow: {e}")

    # ============================================================
    # 🧪 Prepare test input and features
    # ============================================================
    test_text = pd.Series(["This is awesome!", "This is terrible."])

    # 1️⃣ TF-IDF features with 1-3 grams
    tfidf_vectorizer = TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 3),        # <--- ✅ changed here
        analyzer="word"
    )
    X_tfidf = tfidf_vectorizer.fit_transform(test_text).toarray()

    # 2️⃣ Handcrafted numeric features
    feature_list = [
        "comment_length",
        "word_count",
        "unique_word_count",
        "num_exclamations",
        "num_questions",
        "sentiment"
    ]
    X_handcrafted = create_text_features(test_text, feature_list)

    # 3️⃣ Scale numeric features
    scaler = StandardScaler()
    X_handcrafted_scaled = scaler.fit_transform(X_handcrafted)

    # 4️⃣ Concatenate TF-IDF and numeric features
    X_test_final = np.hstack([X_tfidf, X_handcrafted_scaled])

    # ============================================================
    # 🚀 Run prediction using numeric array
    # ============================================================
    try:
        predictions = model.predict(X_test_final)
        assert len(predictions) == len(test_text), (
            f"Expected {len(test_text)} predictions, got {len(predictions)}"
        )
        print(f"✅ Model '{model_name}' v{latest_version} successfully handled numeric input (1–3 ngrams TF-IDF + features).")
    except Exception as e:
        pytest.fail(
            f"❌ Model signature test failed for '{model_name}' v{latest_version}: {e}"
        )
