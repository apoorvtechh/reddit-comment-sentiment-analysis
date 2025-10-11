# ================================
# ✅ Base lightweight Python image
# ================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files & use unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app


# ================================
# 2️⃣ Install Python dependencies
# ================================
COPY requirements_flask.txt .
RUN pip install --no-cache-dir -r requirements_flask.txt

# ================================
# 3️⃣ Download only necessary NLTK/TextBlob data
# ================================
RUN python -m nltk.downloader stopwords wordnet \
    && python -m textblob.download_corpora lite || true

# ================================
# 4️⃣ Copy your app code and model artifacts
# ================================
COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl
COPY scaler.pkl /app/scaler.pkl

# ================================
# 5️⃣ Expose port and set default command
# ================================
EXPOSE 5000
CMD ["python", "app.py"]
