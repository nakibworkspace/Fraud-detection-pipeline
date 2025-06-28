from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import feast
from feast import FeatureStore
import logging
import joblib
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from airflow import DAG
from airflow.operators.python import PythonOperator

# Default arguments
default_args = {
    'owner': 'fraud-detection-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create DAG
dag = DAG(
    'fraud_detection_training',
    default_args=default_args,
    description='Fraud Detection Model Training Pipeline',
    schedule_interval=timedelta(days=7),  # Weekly training
    catchup=False,
    tags=['fraud_detection', 'training', 'mlflow']
)

def load_data_from_feast(**context):
    """Task 1: Load training data from Feast offline store"""
    
    try:
        # Initialize Feast client with correct path
        feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
        os.chdir(feast_repo_path)
        store = FeatureStore(repo_path=".")
        
        logger.info("Feast store initialized successfully")
        
        # Create entity dataframe for historical feature retrieval
        # This should contain the entities and timestamps for which we want features
        
        # Option 1: If you have a specific training dataset with known transaction IDs
        try:
            # Try to load existing entity data
            entity_df = pd.read_parquet(f'{feast_repo_path}/data/processed_features.parquet')
            logger.info(f"Loaded entity dataframe with {len(entity_df)} records")
        except FileNotFoundError:
            # Option 2: Create entity dataframe from the source data
            source_data = pd.read_parquet(f'{feast_repo_path}/data/data_source.parquet')
            
            # Create entity dataframe with required columns for Feast
            entity_df = pd.DataFrame({
                'TransactionID': source_data['TransactionID'],
                'event_timestamp': pd.to_datetime(source_data['TransactionDT'], unit='s')
            })
            
            # Filter to recent data for training (e.g., last 30 days from max timestamp)
            max_timestamp = entity_df['event_timestamp'].max()
            cutoff_date = max_timestamp - timedelta(days=30)
            entity_df = entity_df[entity_df['event_timestamp'] >= cutoff_date]
            
            logger.info(f"Created entity dataframe with {len(entity_df)} records from source data")
        
        # Dynamically discover available features
        feature_views = store.list_feature_views()
        if not feature_views:
            raise ValueError("No feature views found in the feature store. Run 'feast apply' first.")
        
        logger.info(f"Available feature views: {[fv.name for fv in feature_views]}")
        
        # Use the first available feature view (since you have 'fraud_features')
        if not feature_views:
            raise ValueError("No feature views found in the feature store. Run 'feast apply' first.")
        
        # Use the first feature view available
        primary_fv = feature_views[0]
        logger.info(f"Using primary feature view: {primary_fv.name}")
        
        # Try to find transaction_features specifically, but fall back to first available
        transaction_fv = primary_fv
        for fv in feature_views:
            if fv.name in ["transaction_features", "fraud_features"]:
                transaction_fv = fv
                logger.info(f"Found preferred feature view: {fv.name}")
                break
        
        # Build feature references dynamically from the available feature view
        feature_refs = []
        
        # Core transaction features to prioritize (if available)
        priority_features = [
            'TransactionAmt', 'card1', 'card2', 'addr1', 'addr2', 
            'dist1', 'dist2', 'card1_freq', 'card1_count', 'addr1_count',
            'TransactionAmt_to_mean', 'C1', 'C2', 'C3', 'C4', 'V1', 'V2', 'V3'
        ]
        
        # Add priority features first from the main feature view
        for feature in transaction_fv.features:
            if feature.name != 'isFraud' and feature.name in priority_features:
                feature_refs.append(f"{transaction_fv.name}:{feature.name}")
        
        # If we don't have enough priority features, add others from the main feature view
        if len(feature_refs) < 10:
            for feature in transaction_fv.features:
                if feature.name != 'isFraud':  # Exclude target variable
                    feature_ref = f"{transaction_fv.name}:{feature.name}"
                    if feature_ref not in feature_refs:
                        feature_refs.append(feature_ref)
                        if len(feature_refs) >= 20:  # Limit to avoid too many features
                            break
        
        # Add features from other feature views if available
        for fv in feature_views:
            if fv.name != transaction_fv.name:  # Skip the main one we already processed
                for feature in fv.features:
                    if feature.name != 'isFraud' and feature.name in priority_features:
                        feature_ref = f"{fv.name}:{feature.name}"
                        if feature_ref not in feature_refs:
                            feature_refs.append(feature_ref)
                            if len(feature_refs) >= 20:
                                break
                if len(feature_refs) >= 20:
                    break
        
        # Ensure we have at least some features
        if not feature_refs:
            # If no priority features found, just take the first few from the main feature view
            for feature in transaction_fv.features[:10]:
                if feature.name != 'isFraud':
                    feature_refs.append(f"{transaction_fv.name}:{feature.name}")
        
        logger.info(f"Using {len(feature_refs)} features for training")
        logger.info(f"Selected features: {feature_refs[:10]}...")  # Show first 10
        
        # Get historical features from Feast
        logger.info("Retrieving historical features from Feast...")
        training_df = store.get_historical_features(
            entity_df=entity_df[['TransactionID', 'event_timestamp']],
            features=feature_refs
        ).to_df()
        
        logger.info(f"Retrieved {len(training_df)} samples with {len(training_df.columns)} features")
        
        # Save training data for next task
        output_path = '/tmp/feast_training_data.csv'
        training_df.to_csv(output_path, index=False)
        
        # Store metadata in XCom for downstream tasks
        context['task_instance'].xcom_push(
            key='training_data_info',
            value={
                'num_samples': len(training_df),
                'num_features': len(training_df.columns),
                'feature_names': list(training_df.columns),
                'output_path': output_path
            }
        )
        
        logger.info(f"Successfully loaded {len(training_df)} samples from Feast offline store")
        logger.info(f"Features available: {list(training_df.columns)}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error loading data from Feast: {str(e)}")
        raise

def prepare_training_data(**context):
    """Task 2: Prepare data for training using Feast features"""
    
    try:
        # Get info from previous task
        training_data_info = context['task_instance'].xcom_pull(
            task_ids='load_data_from_feast',
            key='training_data_info'
        )
        
        if not training_data_info:
            raise ValueError("No training data info received from previous task")
        
        # Load the Feast training data
        feast_df = pd.read_csv(training_data_info['output_path'])
        logger.info(f"Loaded Feast data with shape: {feast_df.shape}")
        
        # Load target variable from source data if available
        try:
            source_data = pd.read_parquet('/root/code/fraud_detection_pipeline/fraud_detection_features/data/data_source.parquet')
            
            # Merge with source data to get target variable
            if 'isFraud' in source_data.columns:
                # Merge on TransactionID to get the target - use inner join to avoid NaN
                logger.info(f"Merging {len(feast_df)} Feast records with {len(source_data)} source records")
                
                full_df = feast_df.merge(
                    source_data[['TransactionID', 'isFraud']], 
                    on='TransactionID', 
                    how='inner'  # Changed from 'left' to 'inner' to avoid NaN
                )
                
                logger.info(f"Successfully merged with target variable from source data. Final records: {len(full_df)}")
                
                # Check if we lost too many records
                if len(full_df) < len(feast_df) * 0.5:
                    logger.warning(f"Lost {len(feast_df) - len(full_df)} records in merge ({(1 - len(full_df)/len(feast_df))*100:.1f}%)")
                
            else:
                full_df = feast_df.copy()
                logger.warning("No target variable found in source data")
        except Exception as e:
            logger.warning(f"Could not load source data for target: {str(e)}")
            full_df = feast_df.copy()
        
        # Define feature columns (exclude ID and timestamp columns)
        exclude_cols = ['TransactionID', 'event_timestamp']
        feature_columns = [col for col in full_df.columns if col not in exclude_cols and col != 'isFraud']
        
        logger.info(f"Using features: {feature_columns}")
        
        # Prepare features
        X = full_df[feature_columns].copy()
        
        # Handle missing values if any
        if X.isnull().sum().sum() > 0:
            logger.warning("Found missing values, filling with median/mode")
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0, inplace=True)
        
        # Handle target variable
        if 'isFraud' in full_df.columns:
            y = full_df['isFraud'].copy()
            
            # Check for missing values in target
            if y.isnull().sum() > 0:
                logger.warning(f"Found {y.isnull().sum()} missing values in target variable")
                
                # Option 1: Drop rows with missing targets
                missing_mask = y.isnull()
                if missing_mask.sum() < len(y) * 0.1:  # If less than 10% missing
                    logger.info("Dropping rows with missing target values")
                    y = y[~missing_mask]
                    X = X[~missing_mask]
                    full_df = full_df[~missing_mask]
                else:
                    # Option 2: Fill missing targets with 0 (non-fraud)
                    logger.warning("Too many missing targets, filling with 0 (non-fraud)")
                    y = y.fillna(0)
            
            # Ensure target is integer type
            y = y.astype(int)
            
            logger.info(f"Using actual fraud labels. Fraud rate: {y.mean():.3f}")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        else:
            # Create synthetic target for demonstration
            np.random.seed(42)
            y = np.random.choice([0, 1], size=len(X), p=[0.97, 0.03])  # 3% fraud rate
            logger.warning("Created synthetic fraud labels for demonstration")
        
        # Final data quality checks
        logger.info(f"Final data shapes - X: {X.shape}, y: {y.shape}")
        
        # Ensure X and y have same length
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len] if hasattr(y, 'iloc') else y[:min_len]
            logger.warning(f"Aligned X and y to same length: {min_len}")
        
        # Check for any remaining NaN values
        if X.isnull().sum().sum() > 0:
            logger.warning("Still have missing values in features after cleaning")
        
        if pd.isna(y).sum() > 0:
            logger.error("Still have missing values in target after cleaning")
            raise ValueError("Target variable still contains NaN values after cleaning")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save splits
        train_data_paths = {
            'X_train': '/tmp/X_train.csv',
            'X_test': '/tmp/X_test.csv', 
            'y_train': '/tmp/y_train.csv',
            'y_test': '/tmp/y_test.csv'
        }
        
        X_train.to_csv(train_data_paths['X_train'], index=False)
        X_test.to_csv(train_data_paths['X_test'], index=False)
        pd.Series(y_train, name='isFraud').to_csv(train_data_paths['y_train'], index=False)
        pd.Series(y_test, name='isFraud').to_csv(train_data_paths['y_test'], index=False)
        
        # Store metadata for downstream tasks
        training_metadata = {
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'fraud_rate_train': float(y_train.mean()),
            'fraud_rate_test': float(y_test.mean()),
            'feature_names': list(X_train.columns),
            'file_paths': train_data_paths
        }
        
        context['task_instance'].xcom_push(
            key='training_metadata',
            value=training_metadata
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Fraud rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return training_metadata
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise

def validate_feast_setup(**context):
    """Task 0: Validate Feast setup and feature store"""
    
    try:
        feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
        os.chdir(feast_repo_path)
        
        # Initialize Feast store
        store = FeatureStore(repo_path=".")
        
        # List available feature views
        feature_views = store.list_feature_views()
        logger.info(f"Available feature views: {[fv.name for fv in feature_views]}")
        
        # List available entities
        entities = store.list_entities()
        logger.info(f"Available entities: {[entity.name for entity in entities]}")
        
        # Check if data source exists
        data_source_path = f"{feast_repo_path}/data/data_source.parquet"
        if os.path.exists(data_source_path):
            source_df = pd.read_parquet(data_source_path)
            logger.info(f"Data source found with {len(source_df)} records and columns: {list(source_df.columns)}")
        else:
            logger.error(f"Data source not found at {data_source_path}")
            raise FileNotFoundError(f"Data source not found at {data_source_path}")
        
        # Validate feature store registry
        try:
            store.refresh_registry()
            logger.info("Feature store registry refreshed successfully")
        except Exception as e:
            logger.warning(f"Could not refresh registry: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Feast setup validation failed: {str(e)}")
        raise

# Additional utility functions for Airflow integration

def get_feast_feature_metadata(**context):
    """Get metadata about available features in Feast"""
    
    try:
        feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
        os.chdir(feast_repo_path)
        store = FeatureStore(repo_path=".")
        
        metadata = {
            'feature_views': [],
            'entities': [],
            'data_sources': []
        }
        
        # Get feature view information
        for fv in store.list_feature_views():
            fv_info = {
                'name': fv.name,
                'features': [f.name for f in fv.features],
                'entities': fv.entities,
                'ttl': str(fv.ttl) if fv.ttl else None
            }
            metadata['feature_views'].append(fv_info)
        
        # Get entity information  
        for entity in store.list_entities():
            entity_info = {
                'name': entity.name,
                'join_keys': entity.join_keys,
                'value_type': str(entity.value_type)
            }
            metadata['entities'].append(entity_info)
        
        context['task_instance'].xcom_push(key='feast_metadata', value=metadata)
        logger.info(f"Feast metadata collected: {metadata}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error collecting Feast metadata: {str(e)}")
        raise

def preprocess_features(X_train, X_test):
    """Preprocess features to handle categorical variables and data types"""
    from sklearn.preprocessing import LabelEncoder
    
    logger.info("Preprocessing features...")
    logger.info(f"Before preprocessing - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Make copies to avoid modifying original data
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Handle categorical/object columns
    categorical_columns = X_train_processed.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        logger.info(f"Found categorical columns: {categorical_columns}")
        
        # Label encode categorical columns
        for col in categorical_columns:
            logger.info(f"Encoding column: {col}")
            
            # Create label encoder
            le = LabelEncoder()
            
            # Combine train and test data to fit encoder
            combined_data = pd.concat([X_train_processed[col], X_test_processed[col]], axis=0)
            
            # Handle missing values
            combined_data = combined_data.fillna('missing')
            
            # Fit encoder
            le.fit(combined_data)
            
            # Transform train and test data
            X_train_processed[col] = le.transform(X_train_processed[col].fillna('missing'))
            X_test_processed[col] = le.transform(X_test_processed[col].fillna('missing'))
    
    # Convert all columns to numeric
    for col in X_train_processed.columns:
        X_train_processed[col] = pd.to_numeric(X_train_processed[col], errors='coerce')
        X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce')
    
    # Handle any remaining missing values
    X_train_processed = X_train_processed.fillna(0)
    X_test_processed = X_test_processed.fillna(0)
    
    logger.info(f"After preprocessing - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed


def train_xgboost_model(**context):
    """Task 3: Train XGBoost model"""
    
    # Load data
    X_train = pd.read_csv('/tmp/X_train.csv')
    X_test = pd.read_csv('/tmp/X_test.csv')
    y_train = pd.read_csv('/tmp/y_train.csv').iloc[:, 0]
    y_test = pd.read_csv('/tmp/y_test.csv').iloc[:, 0]
    
    # Preprocess features to handle categorical columns
    X_train, X_test = preprocess_features(X_train, X_test)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud_detection_xgboost")
    
    with mlflow.start_run(run_name="xgboost_training"):
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metrics({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        })
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Save model locally
        joblib.dump(model, '/tmp/xgboost_model.pkl')
        
        print(f"XGBoost - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

def train_lightgbm_model(**context):
    """Task 4: Train LightGBM model"""
    
    # Load data
    X_train = pd.read_csv('/tmp/X_train.csv')
    X_test = pd.read_csv('/tmp/X_test.csv')
    y_train = pd.read_csv('/tmp/y_train.csv').iloc[:, 0]
    y_test = pd.read_csv('/tmp/y_test.csv').iloc[:, 0]
    
    # Preprocess features to handle categorical columns
    X_train, X_test = preprocess_features(X_train, X_test)
    
    # Set MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud_detection_lightgbm")
    
    with mlflow.start_run(run_name="lightgbm_training"):
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metrics({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        })
        
        # Log model
        mlflow.lightgbm.log_model(model, "model")
        
        # Save model locally
        joblib.dump(model, '/tmp/lightgbm_model.pkl')
        
        print(f"LightGBM - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

def train_isolation_forest(**context):
    """Task 5: Train Isolation Forest (unsupervised anomaly detection)"""
    
    # Load data
    X_train = pd.read_csv('/tmp/X_train.csv')
    X_test = pd.read_csv('/tmp/X_test.csv')
    y_test = pd.read_csv('/tmp/y_test.csv').iloc[:, 0]
    
    # Preprocess features to handle categorical columns
    X_train, X_test = preprocess_features(X_train, X_test)
    
    # Set MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud_detection_isolation_forest")
    
    with mlflow.start_run(run_name="isolation_forest_training"):
        # Isolation Forest parameters
        params = {
            'contamination': 0.03,  # Expected fraud rate
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model (unsupervised)
        model = IsolationForest(**params)
        model.fit(X_train)
        
        # Predictions (-1 for anomalies, 1 for normal)
        y_pred_raw = model.predict(X_test)
        y_pred = (y_pred_raw == -1).astype(int)  # Convert to 0/1
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        joblib.dump(model, '/tmp/isolation_forest_model.pkl')
        
        print(f"Isolation Forest - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

def compare_models_and_select_best(**context):
    """Task 6: Compare models and select the best one"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Get all experiments
    experiments = {
        'xgboost': mlflow.get_experiment_by_name("fraud_detection_xgboost"),
        'lightgbm': mlflow.get_experiment_by_name("fraud_detection_lightgbm"),
        'isolation_forest': mlflow.get_experiment_by_name("fraud_detection_isolation_forest")
    }
    
    best_model = None
    best_auc = 0
    best_model_info = {}
    
    for model_name, experiment in experiments.items():
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs.empty:
                latest_run = runs.iloc[0]
                
                # Get AUC score (use F1 for Isolation Forest since it doesn't have AUC)
                auc_score = latest_run.get('metrics.auc', latest_run.get('metrics.f1_score', 0))
                
                print(f"{model_name} - Latest AUC/F1: {auc_score:.4f}")
                
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_model = model_name
                    best_model_info = {
                        'model_name': model_name,
                        'run_id': latest_run['run_id'],
                        'auc_score': auc_score,
                        'accuracy': latest_run.get('metrics.accuracy', 0),
                        'f1_score': latest_run.get('metrics.f1_score', 0)
                    }
    
    # Log best model selection
    with mlflow.start_run(run_name="model_comparison"):
        # Log parameters (string values)
        mlflow.log_param('best_model', best_model)
        mlflow.log_param('best_model_run_id', best_model_info.get('run_id', ''))
        
        # Log metrics (numeric values only)
        mlflow.log_metrics({
            'best_auc_score': best_model_info.get('auc_score', 0),
            'best_accuracy': best_model_info.get('accuracy', 0),
            'best_f1_score': best_model_info.get('f1_score', 0)
        })
    
    # Save best model info
    import json
    with open('/tmp/best_model_info.json', 'w') as f:
        json.dump(best_model_info, f)
    
    print(f"Best model: {best_model} with AUC/F1: {best_auc:.4f}")
    
    return best_model_info

def register_best_model(**context):
    """Task 7: Register the best model in MLflow Model Registry"""
    
    import json
    import mlflow
    from mlflow.exceptions import MlflowException
    
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Load best model info
    with open('/tmp/best_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    
    model_name = best_model_info['model_name']
    stored_run_id = best_model_info['run_id']
    
    print(f"Attempting to register model: {model_name}")
    print(f"Stored run ID: {stored_run_id}")
    
    # Verify the run exists and find the correct run if needed
    client = mlflow.MlflowClient()
    
    try:
        # Try to get the stored run first
        run = client.get_run(stored_run_id)
        run_id = stored_run_id
        print(f"Found stored run: {run_id}")
    except MlflowException as e:
        print(f"Stored run ID not found: {e}")
        
        # Fallback: Find the latest run for the best model from its experiment
        experiment_name = f"fraud_detection_{model_name}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            print(f"Searching for latest run in experiment: {experiment_name}")
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs.empty:
                run_id = runs.iloc[0]['run_id']
                print(f"Using latest run from experiment: {run_id}")
            else:
                raise Exception(f"No runs found in experiment {experiment_name}")
        else:
            raise Exception(f"Experiment {experiment_name} not found")
    
    # Verify the model artifact exists
    try:
        model_uri = f"runs:/{run_id}/model"
        
        # Check if model artifact exists
        artifacts = client.list_artifacts(run_id)
        model_artifact_exists = any(artifact.path == "model" for artifact in artifacts)
        
        if not model_artifact_exists:
            raise Exception(f"Model artifact not found in run {run_id}")
        
        print(f"Model artifact confirmed in run: {run_id}")
        
        # Register model
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=f"fraud_detection_{model_name}"
        )
        
        print(f"Successfully registered model version {registered_model.version}")
        
        # Transition to staging
        client.transition_model_version_stage(
            name=f"fraud_detection_{model_name}",
            version=registered_model.version,
            stage="Staging"
        )
        
        print(f"Transitioned model version {registered_model.version} to Staging")
        print(f"Registered {model_name} model version {registered_model.version} in MLflow Registry")
        
        # Update the best model info with the actual run ID used
        best_model_info['actual_run_id'] = run_id
        with open('/tmp/best_model_info.json', 'w') as f:
            json.dump(best_model_info, f)
        
        return {
            'model_name': model_name,
            'registered_version': registered_model.version,
            'run_id': run_id,
            'model_uri': model_uri
        }
        
    except Exception as e:
        print(f"Error registering model: {str(e)}")
        
        # Additional debugging information
        print("\nDebugging information:")
        print(f"Available artifacts in run {run_id}:")
        try:
            artifacts = client.list_artifacts(run_id)
            for artifact in artifacts:
                print(f"  - {artifact.path}")
        except Exception as debug_e:
            print(f"  Could not list artifacts: {debug_e}")
        
        raise
    
def create_model_performance_report(**context):
    """Task 8: Create model performance report"""
    
    import json
    
    # Load best model info
    with open('/tmp/best_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    
    # Create performance report
    report = {
        'training_date': datetime.now().isoformat(),
        'best_model': best_model_info['model_name'],
        'model_metrics': {
            'accuracy': best_model_info.get('accuracy', 0),
            'f1_score': best_model_info.get('f1_score', 0),
            'auc_score': best_model_info.get('auc_score', 0)
        },
        'data_summary': {
            'training_samples': len(pd.read_csv('/tmp/X_train.csv')),
            'test_samples': len(pd.read_csv('/tmp/X_test.csv')),
            'features_count': len(pd.read_csv('/tmp/X_train.csv').columns)
        }
    }
    
    # Save report
    with open('/tmp/model_performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Model performance report created successfully")

# Define tasks
load_data_task = PythonOperator(
    task_id='load_data_from_feast',
    python_callable=load_data_from_feast,
    dag=dag
)

prepare_data_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag
)

xgboost_task = PythonOperator(
    task_id='train_xgboost_model',
    python_callable=train_xgboost_model,
    dag=dag
)

lightgbm_task = PythonOperator(
    task_id='train_lightgbm_model',
    python_callable=train_lightgbm_model,
    dag=dag
)

isolation_forest_task = PythonOperator(
    task_id='train_isolation_forest',
    python_callable=train_isolation_forest,
    dag=dag
)

compare_models_task = PythonOperator(
    task_id='compare_models_and_select_best',
    python_callable=compare_models_and_select_best,
    dag=dag
)

register_model_task = PythonOperator(
    task_id='register_best_model',
    python_callable=register_best_model,
    dag=dag
)

performance_report_task = PythonOperator(
    task_id='create_model_performance_report',
    python_callable=create_model_performance_report,
    dag=dag
)

# Define task dependencies
load_data_task >> prepare_data_task
prepare_data_task >> [xgboost_task, lightgbm_task, isolation_forest_task]
[xgboost_task, lightgbm_task, isolation_forest_task] >> compare_models_task
compare_models_task >> register_model_task >> performance_report_task