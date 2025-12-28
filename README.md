# Reddit Comment Sentiment Analysis

An end-to-end Machine Learning system for classifying Reddit comments into **Positive**, **Negative**, and **Neutral** sentiments.  
Designed as a production-ready NLP project with a complete **MLOps workflow**, including DVC pipelines, automated training, CI/CD, Docker deployment, and reproducible experiments.

**Project Synopsis:** https://synopsis-yqdbpufcczaocxsai2zp3w.streamlit.app/

---

## Project Overview

This project implements a **production-grade sentiment classification system** with a clean, modular, and fully automated ML pipeline.

Key capabilities include:

- Automated data versioning using DVC  
- Modular and reproducible ML training pipeline  
- Advanced NLP preprocessing workflow  
- TF-IDF feature extraction  
- Multiple model experimentation:
  - KNN  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - Support Vector Machine (SVM)  
- Hyperparameter tuning and model comparison  
- Automated evaluation and metric reporting  
- Dockerized deployment setup  
- CI/CD-ready project structure  

---

## Model Used

Models trained and evaluated during experimentation:

- Logistic Regression  

### Best Model: Logistic Regression with TF-IDF

- Achieved the highest accuracy among tested models  
- Demonstrated strong generalization performance  
- Delivered consistent results across all sentiment classes  

---

## NLP Preprocessing Pipeline

The text processing workflow includes:

- Text cleaning and normalization  
- Tokenization  
- Stopword removal  
- Lemmatization  
- TF-IDF vectorization  
- Efficient sparse feature handling  

---

## DVC Pipeline Workflow

The project uses a fully automated DVC pipeline to ensure reproducibility and traceability.

### Pipeline Stages

1. **clean** – Clean and normalize raw Reddit comments  
2. **preprocess** – Apply NLP preprocessing (tokenization, stopwords, lemmatization)  
3. **vectorize** – Transform text into TF-IDF features  
4. **train** – Train sentiment classification models  
5. **evaluate** – Generate evaluation metrics and reports  
6. **push** – Push updated data and models to the DVC remote  

---

## Project Repositories

- **Main Project Repository:**  
  https://github.com/apoorvtechh/reddit-comment-sentiment-analysis  

- **Chrome Extension Repository:**  
  https://github.com/apoorvtechh/reddit-yt-plugin  

- **Experimentation Repository:**  
  https://github.com/apoorvtechh/Second_project  
