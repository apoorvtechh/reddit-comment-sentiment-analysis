import os
import re
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()


# ----------------------------
# Flask App Setup
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# Global Preprocessing Tools
# ----------------------------
stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Load Model from MLflow Registry
# ----------------------------
 # ✅ replace if different
MODEL_NAME = "reddit_chrome_plugin_model"       # ✅ replace with your model registry name
MODEL_VERSION = "1"           # ✅ replace with version you want to load

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(model_uri)

# ----------------------------
# Load Vectorizer from Project Folder
# ----------------------------
VECTORIZER_PATH = "./tfidf_vectorizer.pkl"   # ✅ adjust path if needed
vectorizer = joblib.load(VECTORIZER_PATH)

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_comment(comment: str) -> str:
    """Clean and normalize the text for sentiment analysis."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        words = [w for w in comment.split() if w not in stop_words]
        lemmatized = [lemmatizer.lemmatize(w) for w in words]
        return ' '.join(lemmatized)
    except Exception as e:
        print(f"[Preprocess Error] {e}")
        return comment

# ----------------------------
# Handcrafted Feature Extraction
# ----------------------------
def extract_handcrafted_features(comments, features_list=None):
    """Generate numeric features for each comment."""
    if features_list is None:
        features_list = [
            "comment_length",
            "word_count",
            "unique_word_count",
            "num_exclamations",
            "num_questions",
            "sentiment"
        ]
    s = pd.Series(comments)
    feats = {}
    if "comment_length" in features_list:
        feats["comment_length"] = s.apply(len)
    if "word_count" in features_list:
        feats["word_count"] = s.apply(lambda x: len(x.split()))
    if "unique_word_count" in features_list:
        feats["unique_word_count"] = s.apply(lambda x: len(set(x.split())))
    if "num_exclamations" in features_list:
        feats["num_exclamations"] = s.apply(lambda x: x.count("!"))
    if "num_questions" in features_list:
        feats["num_questions"] = s.apply(lambda x: x.count("?"))
    if "sentiment" in features_list:
        feats["sentiment"] = s.apply(lambda x: TextBlob(x).sentiment.polarity)

    return pd.DataFrame(feats)

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return jsonify({"message": "🚀 Sentiment API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """POST endpoint to analyze sentiment of comments."""
    data = request.get_json()
    comments = data.get("comments", [])

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # 1. Preprocess
        processed_comments = [preprocess_comment(c) for c in comments]

        # 2. TF–IDF Transform
        tfidf_features = vectorizer.transform(processed_comments)

        # 3. Handcrafted Feature Extraction
        handcrafted_features = extract_handcrafted_features(processed_comments)

        # 4. Combine TF–IDF and handcrafted
        X_infer = hstack([tfidf_features, handcrafted_features.values])

        # 5. Predict
        preds = model.predict(X_infer).tolist()

        # 6. Build response
        response = [{"comment": c, "sentiment": str(p)} for c, p in zip(comments, preds)]
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
