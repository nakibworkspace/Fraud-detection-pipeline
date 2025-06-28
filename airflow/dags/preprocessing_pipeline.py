from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
import os
from feast import FeatureStore
from datetime import datetime, timedelta

# Configuration
TMP_DIR = '/root/code/fraud_detection_pipeline/tmp'
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'fraud_detection'
FEAST_REPO_PATH = '/root/code/fraud_detection_pipeline/fraud_detection_features'


# Utility Functions
def get_mongo_client():
    """Establish connection to MongoDB"""
    try:
        return MongoClient(MONGO_URI)
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise

def load_data_from_mongodb(collection_name, limit=500): ##used limit=1000 instead of chunk_size=10000 to test the process
    """Load a limited number of documents from a MongoDB collection"""
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[collection_name]

    cursor = collection.find({}).limit(limit)
    df = pd.DataFrame(list(cursor))
    if '_id' in df.columns:
        df.drop('_id', axis=1, inplace=True)
    return df

def save_data_to_mongodb(data, collection_name):
    try:
        client = get_mongo_client()
        db = client['fraud_detection']
        collection = db[collection_name]
        collection.drop()
        records = data.to_dict('records')
        collection.insert_many(records, ordered=False)
        print(f"Saved data to MongoDB collection {collection_name}")
    except Exception as e:
        print(f"Error saving data to MongoDB collection {collection_name}: {e}")
        raise

def load_data():
    """Load and merge limited transaction and identity data from MongoDB"""
    print("Loading 1000 train and test documents from MongoDB...")
    try:
        # Load limited datasets
        train_transaction = load_data_from_mongodb('train_transaction', limit=1000)
        test_transaction = load_data_from_mongodb('test_transaction', limit=1000)
        train_identity = load_data_from_mongodb('train_identity', limit=1000)
        test_identity = load_data_from_mongodb('test_identity', limit=1000)

        # Merge datasets
        X_train = train_transaction.merge(train_identity, on='TransactionID', how='left')
        X_test = test_transaction.merge(test_identity, on='TransactionID', how='left')

        # Display basic info
        print(f"Train data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        if 'isFraud' in X_train.columns:
            print(f"Target distribution: {X_train['isFraud'].value_counts()}")
        else:
            print("Warning: 'isFraud' column not found in training data")

        os.makedirs(TMP_DIR, exist_ok=True)
        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))
        print("Limited data loading completed successfully")
    except Exception as e:
        print(f"Error in load_data: {e}")
        raise

def normalize_d_columns_task():
    """Normalize D columns (D1-D15)"""
    print("Starting D columns normalization task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))

        d_columns = [f'D{i}' for i in range(1, 16) if f'D{i}' in X_train.columns]
        for col in d_columns:
            X_train[col] = X_train[col] - X_train['TransactionDT']
            X_test[col] = X_test[col] - X_test['TransactionDT']

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))
        print(f"Normalized D columns: {d_columns}")
    except Exception as e:
        print(f"Error in normalize_d_columns_task: {e}")
        raise

def frequency_encoding_task():
    """Apply frequency encoding to card and address columns"""
    print("Starting frequency encoding task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))

        card_addr_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2']
        for col in card_addr_cols:
            if col in X_train.columns:
                freq = X_train[col].value_counts().to_dict()
                X_train[f'{col}_freq'] = X_train[col].map(freq)
                X_test[f'{col}_freq'] = X_test[col].map(freq)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))
        print("Frequency encoding task completed")
    except Exception as e:
        print(f"Error in frequency_encoding_task: {e}")
        raise

def combine_features_task():
    """Combine features to create new categorical variables"""
    print("Starting combine features task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_fe.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_fe.pkl'))

        for df in [X_train, X_test]:
            df['card1_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
            df['card1_addr1_P_emaildomain'] = df['card1_addr1'].astype(str) + '_' + df['P_emaildomain'].astype(str)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_combined.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_combined.pkl'))
        print("Combine features task completed")
    except Exception as e:
        print(f"Error in combine_features_task: {e}")
        raise

def label_encoding_task():
    """Apply label encoding to combined features"""
    print("Starting label encoding task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_combined.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_combined.pkl'))

        cols = ['card1_addr1', 'card1_addr1_P_emaildomain']
        for col in cols:
            if col in X_train.columns:
                le = LabelEncoder()
                train_values = X_train[col].astype(str)
                test_values = X_test[col].astype(str)
                le.fit(pd.concat([train_values, test_values]))
                X_train[f'{col}_encoded'] = le.transform(train_values)
                X_test[f'{col}_encoded'] = le.transform(test_values)
                print(f"Label encoded {col}")

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Label encoding task completed")
    except Exception as e:
        print(f"Error in label_encoding_task: {e}")
        raise

def group_aggregation_task():
    """Apply group aggregation features (count)"""
    print("Starting group aggregation task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        agg_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2']
        for col in agg_cols:
            if col in X_train.columns:
                agg = X_train.groupby(col)['TransactionID'].count().to_dict()
                X_train[f'{col}_count'] = X_train[col].map(agg)
                X_test[f'{col}_count'] = X_test[col].map(agg)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Group aggregation task completed")
    except Exception as e:
        print(f"Error in group_aggregation_task: {e}")
        raise

def group_aggregation_nunique_task():
    """Apply group aggregation with nunique for TransactionAmt"""
    print("Starting group aggregation nunique task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        agg_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2']
        for col in agg_cols:
            if col in X_train.columns:
                agg = X_train.groupby(col)['TransactionAmt'].nunique().to_dict()
                X_train[f'{col}_nunique'] = X_train[col].map(agg)
                X_test[f'{col}_nunique'] = X_test[col].map(agg)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Group aggregation nunique task completed")
    except Exception as e:
        print(f"Error in group_aggregation_nunique_task: {e}")
        raise

def transaction_cents_task():
    """Extract transaction cents and amount features"""
    print("Starting transaction features task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        for df in [X_train, X_test]:
            df['TransactionAmt_to_mean'] = df['TransactionAmt'] / df['TransactionAmt'].mean()
            df['TransactionCents'] = df['TransactionAmt'] - df['TransactionAmt'].astype(int)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Transaction features task completed")
    except Exception as e:
        print(f"Error in transaction_cents_task: {e}")
        raise

def frequency_encoding_combined_task():
    """Apply frequency encoding to combined features"""
    print("Starting frequency encoding combined task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        combined_cols = ['card1_addr1', 'card1_addr1_P_emaildomain']
        for col in combined_cols:
            if col in X_train.columns:
                freq = X_train[col].value_counts().to_dict()
                X_train[f'{col}_freq'] = X_train[col].map(freq)
                X_test[f'{col}_freq'] = X_test[col].map(freq)

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))
        print("Frequency encoding combined task completed")
    except Exception as e:
        print(f"Error in frequency_encoding_combined_task: {e}")
        raise

def remove_columns_task():
    """Remove unnecessary columns and finalize preprocessing"""
    print("Starting finalization task...")
    try:
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_le.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_le.pkl'))

        cols_to_drop = [f'D{i}' for i in range(1, 16)] + ['TransactionDT']
        X_train.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        X_test.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        print("FRAUD DETECTION PREPROCESSING PIPELINE COMPLETED")
        print(f"Final training data shape: {X_train.shape}")
        print(f"Final test data shape: {X_test.shape}")
        print(f"Total features: {len(X_train.columns)}")

        train_memory = X_train.memory_usage(deep=True).sum() / 1024**2
        test_memory = X_test.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory usage - Train: {train_memory:.2f} MB")
        print(f"Memory usage - Test: {test_memory:.2f} MB")

        train_missing = X_train.isnull().sum().sum()
        test_missing = X_test.isnull().sum().sum()
        print(f"Missing values - Train: {train_missing}, Test: {test_missing}")

        X_train.to_pickle(os.path.join(TMP_DIR, 'X_train_final.pkl'))
        X_test.to_pickle(os.path.join(TMP_DIR, 'X_test_final.pkl'))
        save_data_to_mongodb(X_train, 'X_train_final')
        save_data_to_mongodb(X_test, 'X_test_final')
        print("Final data saved to: X_train_final.pkl, X_test_final.pkl")
    except Exception as e:
        print(f"Error in remove_columns_task: {e}")
        raise

def push_to_feast():
    """Push final processed data to Feast feature store"""
    print("Starting push to Feast task...")
    try:
        # Verify the Feast configuration exists
        config_path = os.path.join(FEAST_REPO_PATH, 'feature_store.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"feature_store.yaml not found at {config_path}")
        
        features_path = os.path.join(FEAST_REPO_PATH, 'features.py')
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"features.py not found at {features_path}")
        
        print(f"Using Feast repo at: {FEAST_REPO_PATH}")
        print(f"Config file: {config_path}")
        print(f"Features file: {features_path}")
        
        # Load and prepare data
        X_train = pd.read_pickle(os.path.join(TMP_DIR, 'X_train_final.pkl'))
        X_test = pd.read_pickle(os.path.join(TMP_DIR, 'X_test_final.pkl'))

        print(f"Loaded X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Combine train and test for Feast
        data = pd.concat([X_train, X_test], axis=0)
        data = data.reset_index(drop=True)
        
        # Add required columns for Feast with proper data types
        if 'TransactionID' not in data.columns:
            data['TransactionID'] = range(1, len(data) + 1)  # Start from 1, not 0
        
        # Ensure TransactionID is int64 as specified in your Entity
        data['TransactionID'] = data['TransactionID'].astype('int64')
        
        if 'TransactionDT' not in data.columns:
            # Use proper timestamps for materialization - must be timezone-aware
            base_time = datetime.now().replace(microsecond=0)
            data['TransactionDT'] = [base_time + timedelta(seconds=i) for i in range(len(data))]
        
        # Ensure TransactionDT is datetime64[ns] and timezone-naive for Parquet
        if pd.api.types.is_datetime64_any_dtype(data['TransactionDT']):
            data['TransactionDT'] = pd.to_datetime(data['TransactionDT']).dt.tz_localize(None)

        print(f"Combined data shape: {data.shape}")
        print(f"Data columns: {list(data.columns)}")
        print(f"TransactionID dtype: {data['TransactionID'].dtype}")
        print(f"TransactionDT dtype: {data['TransactionDT'].dtype}")

        # Save to the exact path specified in features.py
        feast_data_path = "/root/code/fraud_detection_pipeline/fraud_detection_features/data/data_source.parquet"
        os.makedirs(os.path.dirname(feast_data_path), exist_ok=True)
        
        # Save with proper Parquet settings
        data.to_parquet(feast_data_path, index=False)
        print(f"Saved feast data to {feast_data_path}")
        
        # Verify the file was created and is readable
        if os.path.exists(feast_data_path):
            test_df = pd.read_parquet(feast_data_path)
            print(f"Verified: File exists and contains {len(test_df)} rows")
        else:
            raise FileNotFoundError(f"Failed to create {feast_data_path}")

        # CRITICAL: Clear Feast registry cache to ensure it uses updated configuration
        print("Clearing Feast registry cache...")
        try:
            import shutil
            registry_cache_path = os.path.join(FEAST_REPO_PATH, '.feast')
            if os.path.exists(registry_cache_path):
                shutil.rmtree(registry_cache_path)
                print("Cleared .feast cache directory")
        except Exception as cache_e:
            print(f"Warning: Could not clear cache: {cache_e}")

        # Initialize Feast store AFTER clearing cache
        print(f"Initializing FeatureStore with repo_path: {FEAST_REPO_PATH}")
        store = FeatureStore(repo_path=FEAST_REPO_PATH)
        
        # Apply feature definitions (this will re-read your features.py)
        print("Applying feature definitions...")
        store.apply([])
        
        # Debug: Check what data sources are actually configured
        print("\n=== FEAST CONFIGURATION DEBUG ===")
        feature_views = store.list_feature_views()
        for fv in feature_views:
            print(f"Feature View: {fv.name}")
    
            # Handle different Feast versions
            data_source = None
            if hasattr(fv, 'source'):
                data_source = fv.source
            elif hasattr(fv, 'batch_source'):
                data_source = fv.batch_source
            elif hasattr(fv, 'stream_source'):
                data_source = fv.stream_source
            
            if data_source:
                print(f"Data Source Type: {type(data_source)}")
                if hasattr(data_source, 'path'):
                    print(f"Data Source Path: {data_source.path}")
                    # Verify the path exists
                    if os.path.exists(data_source.path):
                        print(f"✓ Data source file exists")
                    else:
                        print(f"✗ Data source file MISSING: {data_source.path}")
            else:
                print("No data source found or unable to access")
            print("=== END DEBUG ===\n")
        
        # Materialize features with proper time range
        print("Materializing features...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)  # 1 day window to cover all data
        
        print(f"Materializing from {start_time} to {end_time}")
        
        store.materialize_incremental(
            end_date=end_time,
            feature_views=None  # This will materialize all feature views
        )
        
        print("Data pushed to Feast successfully")
        
    except Exception as e:
        print(f"Error in push_to_feast: {e}")
        import traceback
        traceback.print_exc()
        
        # Enhanced debugging
        print("\n=== ENHANCED ERROR DEBUGGING ===")
        try:
            # Check if the expected file exists
            expected_path = "/root/code/fraud_detection_pipeline/fraud_detection_features/data/data_source.parquet"
            print(f"Expected path exists: {os.path.exists(expected_path)}")
            
            # Check what's in the Feast repo directory
            print(f"Feast repo contents: {os.listdir(FEAST_REPO_PATH)}")
            
            # Check data directory
            data_dir = os.path.join(FEAST_REPO_PATH, 'data')
            if os.path.exists(data_dir):
                print(f"Data directory contents: {os.listdir(data_dir)}")
            
            # Try to read the features.py file to see what's actually configured
            with open(os.path.join(FEAST_REPO_PATH, 'features.py'), 'r') as f:
                features_content = f.read()
                if 'FileSource' in features_content:
                    lines = features_content.split('\n')
                    for i, line in enumerate(lines):
                        if 'path=' in line:
                            print(f"Found path config in features.py line {i+1}: {line.strip()}")
                            
        except Exception as debug_e:
            print(f"Debug error: {debug_e}")
        
        print("=== END ENHANCED DEBUG ===")
        raise

# Define DAG
with DAG(
    'preprocessing_pipeline',
    start_date=datetime(2025, 6, 28),
    schedule_interval='@daily',
    catchup=False
) as dag:
    t1 = PythonOperator(task_id='load_data', python_callable=load_data)
    t2 = PythonOperator(task_id='normalize_d_columns', python_callable=normalize_d_columns_task)
    t3 = PythonOperator(task_id='frequency_encoding', python_callable=frequency_encoding_task)
    t4 = PythonOperator(task_id='combine_features', python_callable=combine_features_task)
    t5 = PythonOperator(task_id='label_encoding', python_callable=label_encoding_task)
    t6 = PythonOperator(task_id='group_aggregation', python_callable=group_aggregation_task)
    t7 = PythonOperator(task_id='group_aggregation_nunique', python_callable=group_aggregation_nunique_task)
    t8 = PythonOperator(task_id='transaction_cents', python_callable=transaction_cents_task)
    t9 = PythonOperator(task_id='frequency_encoding_combined', python_callable=frequency_encoding_combined_task)
    t10 = PythonOperator(task_id='remove_columns', python_callable=remove_columns_task)
    t11 = PythonOperator(task_id='push_to_feast', python_callable=push_to_feast)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8 >> t9 >> t10 >> t11