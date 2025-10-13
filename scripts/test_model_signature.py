import os
import mlflow
import pytest
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# ============================================================
# Helper to recreate handcrafted textual features
# ============================================================
def create_text_features(series: pd.Series) -> pd.DataFrame:
    """Generate handcrafted textual features (length, counts, sentiment, etc.)."""
    return pd.DataFrame({
        "comment_length": series.apply(len),
        "word_count": series.apply(lambda x: len(x.split())),
        "unique_word_count": series.apply(lambda x: len(set(x.split()))),
        "num_exclamations": series.apply(lambda x: x.count("!")),
        "num_questions": series.apply(lambda x: x.count("?")),
        "sentiment": series.apply(lambda x: TextBlob(x).sentiment.polarity),
    })

# ============================================================
# ✅ Load environment variables and set MLflow Tracking URI
# ============================================================
load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise RuntimeError("❌ MLFLOW_TRACKING_URI is not set in your .env file.")
mlflow.set_tracking_uri(tracking_uri)

@pytest.mark.parametrize(
    "model_name",
    [
        "reddit_chrome_plugin_model",  # ✅ your actual model name
    ],
)
def test_model_with_fresh_vectorizer_and_scaler(model_name):
    """
    ✅ Test that the latest registered model can handle inputs preprocessed
    with a freshly created TF-IDF vectorizer (1–3 ngrams) and StandardScaler,
    mimicking the training preprocessing without loading pickled artifacts.
    """

    # 1️⃣ Load latest registered model from MLflow
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    assert versions, f"No versions found for model '{model_name}'"
    latest_version = max(int(v.version) for v in versions)
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # 2️⃣ Dummy text samples
    test_series = pd.Series([
        "hi how are you",
        "this is amazing!",
        "what??? seriously!"
    ])

    # 3️⃣ Fresh TF-IDF vectorizer (same ngram range as training)
    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 3))
    X_tfidf = vectorizer.fit_transform(test_series).toarray()

    # 4️⃣ Fresh handcrafted features + scale
    X_handcrafted = create_text_features(test_series)
    scaler = StandardScaler()
    X_handcrafted_scaled = scaler.fit_transform(X_handcrafted)

    # 5️⃣ Combine TF-IDF + handcrafted features
    X_final = np.hstack([X_tfidf, X_handcrafted_scaled])
    X_final_df = pd.DataFrame(
        X_final,
        columns=[str(i) for i in range(X_final.shape[1])]
    )

    # 6️⃣ Make predictions
    prediction = model.predict(X_final_df)

    # 7️⃣ ✅ Assert output shape matches input rows
    assert len(prediction) == X_final_df.shape[0], (
        f"Output row count mismatch: expected {X_final_df.shape[0]}, got {len(prediction)}"
    )

    print(f"✅ Model '{model_name}' v{latest_version} processed {len(test_series)} fresh inputs successfully.")
