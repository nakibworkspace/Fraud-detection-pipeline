# End to End Machine Learning Model Pipeline 

## Overview

This Machine Learning model pipeline does following:
- Data Ingestion from MongoDB and builds a Feature Engineering Pipeline using Apache Airflow.
- The Pipeline stores featured data into Feast which is running with Online store of Redis and Offline store as local files.
- The Training Pipeline ingest featured data to train Machine Learning models (XGBoost, LGBM, Isolation Forest) and pick the best model to serve.
- Also stores the artifacts on MLFlow.

## Project Setup

Clone the git repository
```
git clone https://github.com/nakibworkspace/Fraud-detection-pipeline.git
```
Setup the VSCode
```sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-venv python3-pip
```

Create a Virtual Environment
```python3 -m venv airflow_env
source airflow_env/bin/activate
```

Install Airflow
```AIRFLOW_VERSION=2.7.3
PYTHON_VERSION="$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```
Install the dependencies
``` 
pip install -r requirements.txt
```

Initialize Airflow db
```
airflow db init
```

Create Airflow user
```
```

Start Airflow UI
```
airflow webserver --port 8081 (Terminal 1)
airflow scheduler (Terminal 2)
```

Access the Airflow UI using Load Balancer
IP:
```
ip addr show eth0
```
and port 8081

