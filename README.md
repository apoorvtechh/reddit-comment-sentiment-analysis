## 📘 Reddit Comment Sentiment Analysis
A complete end-to-end Machine Learning project that classifies Reddit comments into Positive, Negative, and Neutral sentiment.
This repository includes data preprocessing, EDA, DVC pipelines, ML model training, CI/CD, Docker deployment, and production-ready scripts.

## 🚀 Project Overview
This project follows a production-grade MLOps workflow:
1)Automated data versioning using DVC  

2)Modular ML training pipeline

3)Full NLP preprocessing pipeline

4)TF-IDF vectorization

5)Multiple ML models (LR, SVM, RF, XGBoost)

6)Hyperparameter tuning

7)Automated evaluation

8)Docker container for deployment

9)CI/CD pipeline for automated build + deploy

## Pipeline stages 
### clean – clean raw Reddit comments
### preprocess – tokenization, stopword removal, lemmatization
### vectorize – convert text into TF-IDF features
### train – train ML model
### evaluate – evaluate and log metric
### push – push model & data versions to remote storage


## 📂 Folder Structure
The project is designed to be fully reproducible and deployment-ready.

reddit-comment-sentiment-analysis

├── data 

│   ├── raw

│   │   └── reddit_comments_raw.csv

│   ├── processed

│   │   ├── cleaned.csv

│   │   └── preprocessed.csv

│   └── vectorized

│       └── tfidf.pkl

│

├── models

│   └── final_model.pkl

│

├── src

│   ├── __init__.py

│   ├── config.py

│   ├── cleaning.py

│   ├── preprocess.py

│   ├── vectorize.py

│   ├── train.py

│   ├── evaluate.py

│   ├── inference.py

│   └── utils.py

│

├── notebooks

│   ├── 01_EDA.ipynb

│   └── 02_Model_Training.ipynb

│

├── reports

│   ├── metrics.json

│   └── confusion_matrix.png

│

├── dvc.yaml

├── params.yaml

├── requirements.txt

├── Dockerfile

├── Makefile

├── .gitignore

├── README.md

│

└── .github

    └── workflows

        └── ci.yml





