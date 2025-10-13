import os
import re
import pickle
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import mlflow

# ==============================================
# 🌐 Load Environment Variables
# ==============================================
load_dotenv()

# ==============================================
# 🚀 Flask App

app = Flask(__name__)
CORS(app)

# ==============================================
# 🤖 Load Model from MLflow Registry
# ==============================================
MODEL_NAME = "reddit_chrome_plugin_model"
MODEL_VERSION = "1"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(model_uri)
print(f"✅ Loaded model from MLflow Registry: {MODEL_NAME} v{MODEL_VERSION}")

# ==============================================
# 🧠 Load TF–IDF & Scaler
# ==============================================
TFIDF_PATH = "./tfidf_vectorizer.pkl"
SCALER_PATH = "./scaler.pkl"

tfidf = joblib.load(TFIDF_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("✅ TF–IDF Vectorizer & Scaler loaded")

# ==============================================
# 🧠 Preprocessing Tools
# ==============================================
stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
lemmatizer = WordNetLemmatizer()

# ==============================================
# 🧼 Cleaning + Preprocessing
# ==============================================
def clean_comment(text: str) -> str:
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_comment(comment: str) -> str:
    """Deep preprocessing matching training pipeline."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'[^a-z0-9\s!?.,]', '', comment)
        comment = ' '.join([w for w in comment.split() if w not in stop_words])
        comment = ' '.join([lemmatizer.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        print(f"[Preprocess Error] {e}")
        return comment

# ==============================================
# 📊 Handcrafted Features
# ==============================================
def create_text_features(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        'comment_length': series.apply(len),
        'word_count': series.apply(lambda x: len(x.split())),
        'unique_word_count': series.apply(lambda x: len(set(x.split()))),
        'num_exclamations': series.apply(lambda x: x.count('!')),
        'num_questions': series.apply(lambda x: x.count('?')),
        'sentiment': series.apply(lambda x: TextBlob(x).sentiment.polarity)
    })

# ==============================================
# 🔐 Reddit API Auth
# ==============================================
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

def get_access_token():
    auth = HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}
    headers = {"User-Agent": USER_AGENT}
    response = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)
    response.raise_for_status()
    return response.json()["access_token"]

# ==============================================
# 📝 Fetch Reddit Comments
# ==============================================
def fetch_comments(post_id: str, limit: int = 200):
    token = get_access_token()
    headers = {
        "Authorization": f"bearer {token}",
        "User-Agent": USER_AGENT
    }
    url = f"https://oauth.reddit.com/comments/{post_id}?limit={limit}&depth=1"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    data = res.json()

    comments = []
    if len(data) > 1:
        for c in data[1]["data"]["children"]:
            if c["kind"] == "t1":
                body = c["data"].get("body", "").strip()
                if body and body.lower() not in ("[deleted]", "[removed]"):
                    cleaned = clean_comment(body)
                    if cleaned and not cleaned.lower().startswith(">namaste"):
                        comments.append(cleaned)
    return comments

# ==============================================
# 📈 Prediction Pipeline
# ==============================================
def predict_sentiment(comments_list):
    # Stage 1: Clean + Preprocess
    processed = [preprocess_comment(clean_comment(c)) for c in comments_list if c]
    if not processed:
        return None

    # Stage 2: TF–IDF
    tfidf_feats = tfidf.transform(processed)

    # Stage 3: Handcrafted + Scaling
    handcrafted = create_text_features(pd.Series(processed))
    handcrafted_scaled = scaler.transform(handcrafted)

    # Stage 4: Combine Features
    X = hstack([tfidf_feats, handcrafted_scaled])

    # Stage 5: Predict
    preds = model.predict(X).tolist()

    # Map class → sentiment values
    sentiment_value_map = {0: 0, 1: 1, 2: -1}
    sentiment_text_map = {0: "Neutral", 1: "Positive", 2: "Negative"}

    # Convert predictions to numeric sentiment scores
    numeric_sentiments = [sentiment_value_map[p] for p in preds]

    # Sentiment counts
    sentiment_counts = {"1": 0, "0": 0, "-1": 0}
    sentiment_data = []

    for idx, val in enumerate(numeric_sentiments):
        sentiment_counts[str(val)] += 1
        sentiment_data.append({
            "timestamp": idx,
            "sentiment": val
        })

    total = len(preds)
    pos = sentiment_counts["1"]
    neu = sentiment_counts["0"]
    neg = sentiment_counts["-1"]

    # Percentages
    pos_perc = round((pos / total) * 100, 2)
    neu_perc = round((neu / total) * 100, 2)
    neg_perc = round((neg / total) * 100, 2)

    percentages = {
        "positive": pos_perc,
        "neutral": neu_perc,
        "negative": neg_perc,
    }

    # ✅ Sentiment Score out of 10
    sentiment_score_10 = round(((pos_perc * 1) + (neu_perc * 0.5) + (neg_perc * 0)) / 100 * 10, 2)

    # Comments with labels
    labeled_comments = [
        {
            "comment": c,
            "sentiment": sentiment_text_map[p],
            "numeric_sentiment": sentiment_value_map[p]
        }
        for c, p in zip(comments_list, preds)
    ]

    # Extra metrics
    avg_length = round(np.mean([len(c) for c in comments_list]), 2) if comments_list else 0
    unique_comments = len(set(comments_list))
    total_comments = len(comments_list)

    metrics = {
        "total_comments": total_comments,
        "unique_comments": unique_comments,
        "avg_comment_length": avg_length,
        "sentiment_score_out_of_10": sentiment_score_10
    }

    return {
        "percentages": percentages,
        "results": labeled_comments,
        "metrics": metrics,
        "trend_data": sentiment_data
    }


# ==============================================
# 🌐 Routes
# ==============================================
@app.route("/")
def home():
    return jsonify({"message": "🚀 Reddit Sentiment API (MLflow Model) is live!"})

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    comments = data.get("comments", [])
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    result = predict_sentiment(comments)
    if result is None:
        return jsonify({"error": "No valid comments after preprocessing"}), 400
    return jsonify(result)

@app.route("/fetch/<post_id>", methods=["GET"])
def fetch_and_predict(post_id):
    try:
        comments = fetch_comments(post_id)
        result = predict_sentiment(comments)
        if result is None:
            return jsonify({"error": "No valid comments fetched"}), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch or predict: {str(e)}"}), 500

# ==============================================
# ▶️ Run App
# ==============================================
if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000, debug=True)
    pass
