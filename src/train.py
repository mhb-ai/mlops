# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import joblib
import os

# Load dataset
df = pd.read_csv("data/data.csv")
X = df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

# MLflow logging
mlflow.set_experiment("intent_classifier")

with mlflow.start_run():
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("data/data.csv")

print(f"Model trained with accuracy: {acc}")
os.makedirs("models", exist_ok=True)
joblib.dump((vectorizer, model), "models/model.pkl")
