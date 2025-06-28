#!/usr/bin/env python3
"""
Simplified Feast setup script focusing on core features
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import feast
from feast import FeatureStore
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_feast_simple():
    """Simple Feast setup with core features only"""
    
    feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
    
    try:
        # Change to the feast repository directory
        os.chdir(feast_repo_path)
        logger.info(f"Changed to directory: {feast_repo_path}")
        
        # Check data source first
        data_source_path = f"{feast_repo_path}/data/data_source.parquet"
        if not os.path.exists(data_source_path):
            logger.error(f"Data source not found: {data_source_path}")
            return False
            
        # Load and inspect data
        df = pd.read_parquet(data_source_path)
        logger.info(f"Data loaded: {df.shape}")
        logger.info(f"Required columns present: TransactionID={('TransactionID' in df.columns)}, TransactionDT={('TransactionDT' in df.columns)}")
        
        # Apply the feature definitions to the registry
        logger.info("Applying feature definitions...")
        result = os.system("feast apply")
        
        if result != 0:
            logger.error("feast apply failed")
            return False
        
        logger.info("✅ feast apply successful")
        
        # Initialize the feature store
        store = FeatureStore(repo_path=".")
        
        # List what was registered
        entities = store.list_entities()
        feature_views = store.list_feature_views()
        
        logger.info(f"✅ Registered {len(entities)} entities and {len(feature_views)} feature views")
        
        for entity in entities:
            logger.info(f"Entity: {entity.name}")
        
        for fv in feature_views:
            logger.info(f"Feature View: {fv.name} with {len(fv.features)} features")
        
        return store
        
    except Exception as e:
        logger.error(f"Error in setup: {str(e)}")
        return False

def test_feature_serving(store):
    """Test basic feature serving"""
    
    try:
        logger.info("Testing feature serving...")
        
        # Create a simple entity dataframe
        entity_df = pd.DataFrame({
            'TransactionID': [1, 2, 3],
            'event_timestamp': [datetime.now() - timedelta(hours=1)] * 3
        })
        
        # Get available feature views
        feature_views = store.list_feature_views()
        if not feature_views:
            logger.error("No feature views available")
            return False
        
        # Use the first feature view for testing
        fv = feature_views[0]
        
        # Test with just a few features to avoid errors
        test_features = []
        for feature in fv.features[:3]:  # Just use first 3 features
            if feature.name != 'isFraud':  # Skip target
                test_features.append(f"{fv.name}:{feature.name}")
        
        if not test_features:
            logger.warning("No suitable test features found")
            return False
        
        logger.info(f"Testing with features: {test_features}")
        
        # Get historical features
        result_df = store.get_historical_features(
            entity_df=entity_df,
            features=test_features
        ).to_df()
        
        logger.info(f"✅ Feature serving test successful!")
        logger.info(f"Result shape: {result_df.shape}")
        logger.info(f"Result columns: {list(result_df.columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature serving test failed: {str(e)}")
        return False

def inspect_data_source():
    """Inspect the data source to understand its structure"""
    
    feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
    data_source_path = f"{feast_repo_path}/data/data_source.parquet"
    
    try:
        df = pd.read_parquet(data_source_path)
        
        logger.info("=== Data Source Inspection ===")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {len(df.columns)}")
        
        # Check for required columns
        required_cols = ['TransactionID', 'TransactionDT']
        for col in required_cols:
            if col in df.columns:
                logger.info(f"✅ {col}: {df[col].dtype}")
                if col == 'TransactionDT':
                    logger.info(f"  Range: {df[col].min()} to {df[col].max()}")
            else:
                logger.error(f"❌ Missing required column: {col}")
        
        # Check target variable
        if 'isFraud' in df.columns:
            fraud_rate = df['isFraud'].mean()
            logger.info(f"✅ isFraud: fraud rate = {fraud_rate:.3f}")
        
        # Show data types breakdown
        dtype_counts = df.dtypes.value_counts()
        logger.info(f"Data types: {dict(dtype_counts)}")
        
        # Check for nulls
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            logger.info(f"Columns with nulls: {len(cols_with_nulls)}")
        else:
            logger.info("✅ No null values found")
        
        return True
        
    except Exception as e:
        logger.error(f"Error inspecting data source: {str(e)}")
        return False

def main():
    """Main function"""
    
    logger.info("=== Simple Feast Setup ===")
    
    # Inspect data source first
    if not inspect_data_source():
        logger.error("Data source inspection failed")
        return
    
    # Setup Feast
    store = setup_feast_simple()
    if not store:
        logger.error("Feast setup failed")
        return
    
    # Test feature serving
    if test_feature_serving(store):
        logger.info("✅ All tests passed! Feast is ready to use.")
    else:
        logger.warning("⚠️ Feast setup complete but serving test failed")
    
    logger.info("=== Setup Complete ===")

if __name__ == "__main__":
    main()