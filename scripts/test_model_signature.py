import os
import pytest
import mlflow
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from dotenv import load_dotenv

# ============================================================
# ✅ Helper to create handcrafted textual features
# ============================================================
def create_text_features(series: pd.Series) -> pd.DataFrame:
    """Generate handcrafted textual features (length, counts, sentiment, etc.)."""
    features = {
        "comment_length": series.apply(len),
        "word_count": series.apply(lambda x: len(x.split())),
        "unique_word_count": series.apply(lambda x: len(set(x.split()))),
        "num_exclamations": series.apply(lambda x: x.count("!")),
        "num_questions": series.apply(lambda x: x.count("?")),
        "sentiment": series.apply(lambda x: TextBlob(x).sentiment.polarity),
    }
    return pd.DataFrame(features)

# ============================================================
# ✅ Load environment variables and configure MLflow
# ============================================================
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

@pytest.mark.parametrize("model_name", [
    "reddit_chrome_plugin_model",  # your registered model name
])
def test_model_signature(model_name):
    """
    ✅ For bare Logistic Regression model:
    - Recreate TF-IDF + handcrafted features
    - Scale them
    - Pass numeric DataFrame to model.predict()
    """

    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    assert model_versions, f"No versions found for model '{model_name}'"
    latest_version = max(int(mv.version) for mv in model_versions)

    # Load the registered model
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # ============================================================
    # 🧪 Create sample text & recreate features exactly like training
    # ============================================================
    test_text = pd.Series(["This is awesome!", "This is terrible."])

    # 1️⃣ TF-IDF with 1–3 ngrams (to match training config)
    tfidf_vectorizer = TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 3)
    )
    X_tfidf = tfidf_vectorizer.fit_transform(test_text).toarray()

    # 2️⃣ Handcrafted features
    X_handcrafted = create_text_features(test_text)

    # 3️⃣ Standard scaling for handcrafted features
    scaler = StandardScaler()
    X_handcrafted_scaled = scaler.fit_transform(X_handcrafted)

    # 4️⃣ Concatenate TF-IDF + handcrafted
    X_final = np.hstack([X_tfidf, X_handcrafted_scaled])

    # 5️⃣ Convert to DataFrame with numeric column names
    num_features = X_final.shape[1]
    X_final_df = pd.DataFrame(X_final, columns=[str(i) for i in range(num_features)])

    # ============================================================
    # 🚀 Run prediction
    # ============================================================
    try:
        predictions = model.predict(X_final_df)
        assert len(predictions) == len(test_text), (
            f"Expected {len(test_text)} predictions, got {len(predictions)}"
        )
        print(f"✅ Bare model '{model_name}' v{latest_version} handled numeric input correctly.")
    except Exception as e:
        pytest.fail(f"❌ Model signature test failed for bare model '{model_name}' v{latest_version}: {e}")
