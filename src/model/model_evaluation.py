import os
import pickle
import yaml
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from textblob import TextBlob
from scipy.sparse import hstack

# -------------------------------------------------
# Logging configuration
# -------------------------------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def get_root_directory() -> str:
    """Return the project root directory (two levels up)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def load_params(params_path: str) -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(file_path: str) -> pd.DataFrame:
    """Load preprocessed data (train or test)."""
    df = pd.read_csv(file_path)
    df.dropna(subset=['clean_comment', 'category'], inplace=True)
    df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})
    logger.debug("Data loaded: %s rows from %s", df.shape[0], file_path)
    return df

def create_text_features(series: pd.Series, features_list: list) -> pd.DataFrame:
    """Generate handcrafted features."""
    features = {}
    if "comment_length" in features_list:
        features["comment_length"] = series.apply(len)
    if "word_count" in features_list:
        features["word_count"] = series.apply(lambda x: len(x.split()))
    if "unique_word_count" in features_list:
        features["unique_word_count"] = series.apply(lambda x: len(set(x.split())))
    if "num_exclamations" in features_list:
        features["num_exclamations"] = series.apply(lambda x: x.count('!'))
    if "num_questions" in features_list:
        features["num_questions"] = series.apply(lambda x: x.count('?'))
    if "sentiment" in features_list:
        features["sentiment"] = series.apply(lambda x: TextBlob(x).sentiment.polarity)
    return pd.DataFrame(features)

# -------------------------------------------------
# Evaluation pipeline
# -------------------------------------------------
def evaluate_split(model, tfidf, data, params, split_name=""):
    """Evaluate model on given dataset split (train/test)."""
    X_raw = data['clean_comment']
    y_true = data['category']

    # Handcrafted features
    if params.get("extra_features", True):
        X_extra = create_text_features(X_raw, params["features_list"])
        scaler = StandardScaler()
        X_extra_scaled = scaler.fit_transform(X_extra)  # ⚠️ For consistency, better to load same scaler used in training
    else:
        X_extra_scaled = None

    # TF-IDF
    X_tfidf = tfidf.transform(X_raw)

    # Combine
    if X_extra_scaled is not None:
        X_combined = hstack([X_tfidf, X_extra_scaled])
    else:
        X_combined = X_tfidf

    # Predict
    y_pred = model.predict(X_combined)

    print(f"\n🔹 Classification Report ({split_name} data):\n")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2]))


def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, "params.yaml"))["model_building"]

        # Load train & test data
        train_data = load_data(os.path.join(root_dir, "data/interim/train_processed.csv"))
        test_data  = load_data(os.path.join(root_dir, "data/interim/test_processed.csv"))

        # Load TF-IDF vectorizer
        with open(os.path.join(root_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf = pickle.load(f)

        # Load trained model
        with open(os.path.join(root_dir, "logreg_model.pkl"), "rb") as f:
            model = pickle.load(f)

        # Evaluate
        evaluate_split(model, tfidf, train_data, params, split_name="Train")
        evaluate_split(model, tfidf, test_data, params, split_name="Test")

        logger.debug("Evaluation completed successfully")

    except Exception as e:
        logger.error("Evaluation pipeline failed: %s", e)
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
