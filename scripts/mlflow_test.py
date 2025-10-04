import mlflow
import random
import os 
from dotenv import load_dotenv
load_dotenv()  #
# Set the MLflow tracking URI

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Start an MLflow run
with mlflow.start_run():
    # Log some random parameters
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_param("param2", random.random())

    # Log some random metrics
    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(0.5, 1.5))

    print("Logged random parameters and metrics.")