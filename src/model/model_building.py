import os
import pickle
import yaml
import logging
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from textblob import TextBlob
from scipy.sparse import hstack

# -------------------------------------------------
# Logging configuration
# -------------------------------------------------
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler (only errors go here)
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Avoid duplicate handlers
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def get_root_directory() -> str:
    """Return the project root directory (two levels up)."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
        logger.debug("Root directory resolved: %s", root_dir)
        return root_dir
    except Exception as e:
        logger.error("Error resolving root directory: %s", e)
        raise

def load_params(params_path: str) -> dict:
    """Load parameters from params.yaml."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("Params file not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file %s: %s", params_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error loading params: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load preprocessed training data."""
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['clean_comment', 'category'], inplace=True)
        df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})
        logger.debug("Training data loaded: %s rows", df.shape[0])
        return df
    except FileNotFoundError:
        logger.error("Training data file not found: %s", file_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Error parsing training data CSV: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error loading data: %s", e)
        raise

def create_text_features(series: pd.Series, features_list: list) -> pd.DataFrame:
    """Generate handcrafted features from text based on features_list."""
    try:
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

        df_features = pd.DataFrame(features)
        logger.debug("Extra features created: %s", list(df_features.columns))
        return df_features
    except Exception as e:
        logger.error("Error creating handcrafted features: %s", e)
        raise

# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def main():
    try:
        root_dir = get_root_directory()

        # Load params
        params = load_params(os.path.join(root_dir, "params.yaml"))["model_building"]

        # Load training data
        train_data = load_data(os.path.join(root_dir, "data/interim/train_processed.csv"))
        X_raw = train_data['clean_comment']
        y_raw = train_data['category']

        # Handcrafted features
        if params.get("extra_features", True):
            X_extra = create_text_features(X_raw, params["features_list"])
            scaler = StandardScaler()
            X_extra_scaled = scaler.fit_transform(X_extra)
        else:
            X_extra_scaled = None
            logger.debug("Skipping extra handcrafted features")

        # TF-IDF
        tfidf = TfidfVectorizer(
            ngram_range=tuple(params["ngram_range"]),
            max_features=params["max_features"]
        )
        X_tfidf = tfidf.fit_transform(X_raw)
        logger.debug("TF-IDF vectorization completed with %s features", X_tfidf.shape[1])

        # Combine TF-IDF + handcrafted
        if X_extra_scaled is not None:
            X_combined = hstack([X_tfidf, X_extra_scaled])
            logger.debug("Features combined: TF-IDF + handcrafted")
        else:
            X_combined = X_tfidf
            logger.debug("Only TF-IDF features used")

        # Save TF-IDF vectorizer
        with open(os.path.join(root_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(tfidf, f)
        logger.debug("TF-IDF vectorizer saved")

        # Handle imbalance with SMOTE
        smote = SMOTE(random_state=params["smote_random_state"])
        X_resampled, y_resampled = smote.fit_resample(X_combined, y_raw)
        logger.debug("SMOTE applied. Resampled dataset size: %s", X_resampled.shape[0])

        # Logistic Regression params from YAML
        model = LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            solver=params["solver"],
            multi_class=params["multi_class"],
            class_weight=params["class_weight"],
            max_iter=params["max_iter"],
            random_state=params["random_state"]
        )
        model.fit(X_resampled, y_resampled)
        logger.debug("Logistic Regression model trained")

        # Save model
        save_path = os.path.join(root_dir, "logreg_model.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.debug("Model saved at %s", save_path)

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        print(f"Error: {e}")
        raise  # important so DVC marks the stage as failed

if __name__ == "__main__":
    main()
