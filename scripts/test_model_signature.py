import os
import pickle
import pytest
import pandas as pd
import numpy as np
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

@pytest.mark.parametrize(
    "model_path, vectorizer_path, scaler_path",
    [
        ("logreg_model.pkl", "tfidf_vectorizer.pkl", "scaler.pkl"),
    ],
)
def test_model_with_local_artifacts(model_path, vectorizer_path, scaler_path):
    """
    ✅ Test the trained model locally using pickled model, vectorizer, and scaler.
    This bypasses MLflow schema enforcement and directly checks if the raw model
    works with the same preprocessing as training.
    """

    # 1️⃣ Load model and artifacts from pickle files
    assert os.path.exists(model_path), f"❌ Model file not found: {model_path}"
    assert os.path.exists(vectorizer_path), f"❌ Vectorizer file not found: {vectorizer_path}"
    assert os.path.exists(scaler_path), f"❌ Scaler file not found: {scaler_path}"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # 2️⃣ Dummy text samples
    test_series = pd.Series([
        "hi how are you",
        "this is amazing!",
        "what??? seriously!"
    ])

    # 3️⃣ Apply TF-IDF exactly like training
    X_tfidf = vectorizer.transform(test_series).toarray()

    # 4️⃣ Handcrafted features + scale
    X_handcrafted = create_text_features(test_series)
    X_handcrafted_scaled = scaler.transform(X_handcrafted)

    # 5️⃣ Combine TF-IDF + handcrafted features
    X_final = np.hstack([X_tfidf, X_handcrafted_scaled])

    # 6️⃣ Run prediction using the raw sklearn model
    prediction = model.predict(X_final)

    # 7️⃣ Assert output matches number of inputs
    assert len(prediction) == len(test_series), (
        f"Output row count mismatch: expected {len(test_series)}, got {len(prediction)}"
    )

    print(f"✅ Local model successfully processed {len(test_series)} inputs.")
    print(f"Predictions: {prediction}")
