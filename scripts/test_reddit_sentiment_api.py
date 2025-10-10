import pytest          # ✅ Pytest framework for writing & running tests
import requests        # ✅ Used to make HTTP requests to the Flask API

# 🌐 Base URL of the Flask app — change if deployed remotely
BASE_URL = "http://localhost:5000"

# ====================================================
# 🟩 1️⃣ Test the Home Endpoint
# ====================================================

def test_home_endpoint():
    """
    ✅ TEST: GET /
    👉 Purpose: To check if the root endpoint returns a success message 
               indicating the API is live.
    """
    # Send GET request to the root endpoint
    response = requests.get(f"{BASE_URL}/")
    
    # Assert that the HTTP response code is 200 (OK)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    # Convert the response to JSON
    json_data = response.json()
    
    # Check if the response contains the expected key and content
    assert "message" in json_data
    assert "Reddit Sentiment API" in json_data["message"]


# ====================================================
# 🟦 2️⃣ Test the /predict Endpoint (Valid Input)
# ====================================================

def test_predict_endpoint_valid():
    """
    ✅ TEST: POST /predict
    👉 Purpose: To verify that the /predict endpoint works correctly when 
               provided with a valid list of comments.
    """
    # Define input data: a JSON payload with 3 sample comments
    data = {
        "comments": [
            "I absolutely love this product!",   # positive
            "This is terrible.",                 # negative
            "It's okay, not great."              # neutral
        ]
    }

    # Send POST request to /predict with the JSON payload
    response = requests.post(f"{BASE_URL}/predict", json=data)

    # Assert that the API responds with 200 OK for valid input
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    # Convert response to JSON
    json_data = response.json()

    # Check the structure of the response
    assert "percentages" in json_data          # Contains positive/neutral/negative %
    assert "results" in json_data              # Contains predictions for each comment
    assert "metrics" in json_data              # Contains stats like avg length, total comments
    assert "trend_data" in json_data           # Contains timestamp → sentiment data

    # Ensure 'results' is a list and its length matches the input
    assert isinstance(json_data["results"], list)
    assert len(json_data["results"]) == len(data["comments"])

    # Validate structure of each result item
    first_item = json_data["results"][0]
    assert "comment" in first_item            # Original text
    assert "sentiment" in first_item          # Sentiment label (Positive / Negative / Neutral)
    assert "numeric_sentiment" in first_item  # Numeric code (1, 0, -1)


# ====================================================
# 🟥 3️⃣ Test the /predict Endpoint (Invalid Input)
# ====================================================

def test_predict_endpoint_no_comments():
    """
    ❌ TEST: POST /predict with empty 'comments'
    👉 Purpose: To ensure the API returns 400 Bad Request 
               when 'comments' list is empty.
    """
    response = requests.post(f"{BASE_URL}/predict", json={"comments": []})
    assert response.status_code == 400
    json_data = response.json()
    assert "error" in json_data


def test_predict_endpoint_missing_key():
    """
    ❌ TEST: POST /predict with missing 'comments' key
    👉 Purpose: To ensure API handles invalid JSON payload gracefully.
    """
    response = requests.post(f"{BASE_URL}/predict", json={})
    assert response.status_code == 400
    json_data = response.json()
    assert "error" in json_data


# ====================================================
# 🟨 4️⃣ Test the /fetch/<post_id> Endpoint
# ====================================================

@pytest.mark.skip(reason="Requires valid Reddit API credentials and real post_id")
def test_fetch_endpoint():
    """
    ⚠️ TEST: GET /fetch/<post_id>
    👉 Purpose: To verify the API can fetch Reddit comments and return sentiment.
    🚨 Skipped by default because it needs valid Reddit credentials and a real post ID.
    """
    test_post_id = "xyz123"  # Replace with a valid post ID for real testing

    # Send GET request to /fetch/<post_id>
    response = requests.get(f"{BASE_URL}/fetch/{test_post_id}")

    # Since the Reddit post may or may not exist, allow multiple valid status codes
    assert response.status_code in [200, 400, 500]

    # If successful, check the expected structure
    if response.status_code == 200:
        json_data = response.json()
        assert "results" in json_data
        assert "percentages" in json_data


# ====================================================
# 🧪 Optional: Edge Case Test for Special Characters
# ====================================================

def test_predict_with_special_characters():
    """
    🧪 TEST: POST /predict with emojis and special characters
    👉 Purpose: To ensure API handles preprocessing of unusual input without crashing.
    """
    data = {
        "comments": ["🔥🔥🔥 BEST EVER!!!! 💯", "???", "   "]
    }

    # Send POST request with unusual input
    response = requests.post(f"{BASE_URL}/predict", json=data)

    # Allow 200 OK or 400 if preprocessing removes all text
    assert response.status_code in [200, 400]
