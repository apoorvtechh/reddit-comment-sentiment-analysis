import os
import pickle
import yaml
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from textblob import TextBlob
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import json
from dotenv import load_dotenv

# Load environment variables (for MLflow tracking URI) added
load_dotenv()

# -------------------------------------------------
# 📜 Logging configuration
# -------------------------------------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# -------------------------------------------------
# 🛠 Utility functions
# -------------------------------------------------
def get_root_directory() -> str:
    """Return the project root directory (two levels up from this file)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../"))

def load_params(params_path: str) -> dict:
    """Load model parameters from params.yaml."""
    with open(params_path, "r") as file:
        return yaml.safe_load(file)

def load_data(file_path: str) -> pd.DataFrame:
    """Load preprocessed data, clean nulls, and map labels for consistency."""
    df = pd.read_csv(file_path)
    df.dropna(subset=["clean_comment", "category"], inplace=True)
    # Remap -1 → 2 for negative, 0 for neutral, 1 for positive
    df["category"] = df["category"].map({-1: 2, 0: 0, 1: 1})
    logger.debug("Data loaded: %s rows from %s", df.shape[0], file_path)
    return df

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

def log_confusion_matrix(cm, dataset_name):
    """Save and log a confusion matrix plot as an MLflow artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_file_path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def evaluate_split_with_logging(model, tfidf, data, params, scaler, split_name=""):
    """Evaluate model on a dataset split, log metrics + confusion matrix to MLflow."""
    X_raw = data["clean_comment"]
    y_true = data["category"]

    # Handcrafted features (same as training)
    if params.get("extra_features", True):
        X_extra = create_text_features(X_raw, params["features_list"])# added
        X_extra_scaled = scaler.transform(X_extra)
    else:
        X_extra_scaled = None

    # TF-IDF vectorization
    X_tfidf = tfidf.transform(X_raw)

    # Combine TF-IDF and handcrafted features exactly like during training
    if X_extra_scaled is not None:
        X_combined = hstack([X_tfidf, X_extra_scaled])
    else:
        X_combined = X_tfidf

    # Make predictions
    y_pred = model.predict(X_combined)

    # Generate classification metrics and confusion matrix
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Log precision/recall/F1 per class to MLflow
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            mlflow.log_metrics({
                f"{split_name}_{label}_precision": metrics["precision"],
                f"{split_name}_{label}_recall": metrics["recall"],
                f"{split_name}_{label}_f1": metrics["f1-score"],
            })

    # Log confusion matrix image
    log_confusion_matrix(cm, split_name)

    print(f"\n🔹 Classification Report ({split_name} data):\n")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2]))

def save_model_info(run_id: str, model_path: str, file_path: str, accuracy: float) -> None:
    """Save the model run ID, path, and accuracy to a JSON file for later registration/testing."""
    model_info = {
        "run_id": run_id,
        "model_path": model_path,
        "accuracy": accuracy
    }
    with open(file_path, "w") as file:
        json.dump(model_info, file, indent=4)
    logger.debug("Model info saved to %s", file_path)

# -------------------------------------------------
# 🚀 Main Evaluation Pipeline
# -------------------------------------------------
from mlflow.models.signature import infer_signature

def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, "params.yaml"))["model_building"]

        # 1️⃣ Load train/test data
        train_data = load_data(os.path.join(root_dir, "data/interim/train_processed.csv"))
        test_data = load_data(os.path.join(root_dir, "data/interim/test_processed.csv"))

        # 2️⃣ Load vectorizer, model & scaler (important!)
        with open(os.path.join(root_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        with open(os.path.join(root_dir, "logreg_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(root_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        # 3️⃣ Configure MLflow tracking
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("dvc-pipeline-runs-Reddit-sentiments-analysis")

        with mlflow.start_run() as run:
            # Log training params
            for key, value in params.items():
                mlflow.log_param(key, value)#added

            # 4️⃣ Signature inference — replicate full feature pipeline (TF-IDF + handcrafted)
            sample_input = train_data[["clean_comment"]].head(5)
            X_extra = create_text_features(sample_input["clean_comment"], params["features_list"])
            X_extra_scaled = scaler.transform(X_extra)
            X_tfidf = tfidf.transform(sample_input["clean_comment"])
            X_combined = hstack([X_tfidf, X_extra_scaled])
            sample_pred = model.predict(X_combined)

            # Create MLflow signature
            signature = infer_signature(
                model_input=sample_input,
                model_output=pd.Series(sample_pred)
            )

            # Log model to MLflow with signature
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="logreg_model",
                signature=signature
            )

            # Log vectorizer as artifact
            mlflow.log_artifact(os.path.join(root_dir, "tfidf_vectorizer.pkl"))

            # 5️⃣ Evaluate model on train and test sets
            evaluate_split_with_logging(model, tfidf, train_data, params, scaler, split_name="Train")
            evaluate_split_with_logging(model, tfidf, test_data, params, scaler, split_name="Test")

            # 6️⃣ Calculate & log test accuracy
            X_test_extra = create_text_features(test_data["clean_comment"], params["features_list"])
            X_test_extra_scaled = scaler.transform(X_test_extra)
            X_test_tfidf = tfidf.transform(test_data["clean_comment"])
            X_test_combined = hstack([X_test_tfidf, X_test_extra_scaled])# added

            y_true = test_data["category"]
            y_pred = model.predict(X_test_combined)
            test_accuracy = accuracy_score(y_true, y_pred)
            mlflow.log_metric("test_accuracy", test_accuracy)
            logger.info(f"⭐ Test accuracy: {test_accuracy:.4f}")# added

            # 7️⃣ Save run info for test_model_accuracy.py
            save_model_info(run.info.run_id, "logreg_model", "experiment_info.json", test_accuracy)

            # add additional tags
            mlflow.set_tag("model_type", "Logistic Regression")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "Reddit Comments")

        logger.debug("Evaluation completed successfully ✅")# added new

    except Exception as e:
        logger.error("Evaluation pipeline failed: %s", e)
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
