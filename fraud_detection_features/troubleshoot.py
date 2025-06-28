#!/usr/bin/env python3
"""
Troubleshooting script for Feast feature store issues
"""

import os
import sys
import pandas as pd
from feast import FeatureStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_feast_setup():
    """Comprehensive diagnosis of Feast setup"""
    
    feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
    
    logger.info("=== Feast Troubleshooting Diagnostic ===")
    
    # Check 1: Directory structure
    logger.info("1. Checking directory structure...")
    if os.path.exists(feast_repo_path):
        logger.info(f"✅ Feast repo directory exists: {feast_repo_path}")
        
        # List contents
        contents = os.listdir(feast_repo_path)
        logger.info(f"Directory contents: {contents}")
        
        # Check for key files
        key_files = ['feature_store.yaml', 'features.py']
        for file in key_files:
            if file in contents:
                logger.info(f"✅ {file} found")
            else:
                logger.error(f"❌ {file} missing")
    else:
        logger.error(f"❌ Feast repo directory not found: {feast_repo_path}")
        return False
    
    # Check 2: Change directory and initialize store
    logger.info("2. Checking Feast store initialization...")
    try:
        os.chdir(feast_repo_path)
        logger.info(f"✅ Changed to directory: {os.getcwd()}")
        
        store = FeatureStore(repo_path=".")
        logger.info("✅ Feast store initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Feast store: {str(e)}")
        return False
    
    # Check 3: Registry and feature definitions
    logger.info("3. Checking feature definitions...")
    try:
        entities = store.list_entities()
        feature_views = store.list_feature_views()
        
        logger.info(f"Entities found: {len(entities)}")
        for entity in entities:
            logger.info(f"  - {entity.name}: {entity.join_keys}")
        
        logger.info(f"Feature views found: {len(feature_views)}")
        for fv in feature_views:
            logger.info(f"  - {fv.name}: {[f.name for f in fv.features]}")
        
        if not feature_views:
            logger.error("❌ No feature views found! Run 'feast apply' to register features.")
            return False
        else:
            logger.info("✅ Feature views found")
            
    except Exception as e:
        logger.error(f"❌ Error listing features: {str(e)}")
        return False
    
    # Check 4: Data source
    logger.info("4. Checking data source...")
    data_source_path = f"{feast_repo_path}/data/data_source.parquet"
    
    if os.path.exists(data_source_path):
        logger.info(f"✅ Data source exists: {data_source_path}")
        
        try:
            df = pd.read_parquet(data_source_path)
            logger.info(f"✅ Data loaded successfully: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Check required columns
            required_cols = ['TransactionID', 'TransactionDT']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"❌ Missing required columns: {missing_cols}")
            else:
                logger.info("✅ Required columns present")
                
        except Exception as e:
            logger.error(f"❌ Error reading data source: {str(e)}")
            return False
    else:
        logger.error(f"❌ Data source not found: {data_source_path}")
        return False
    
    # Check 5: Feature retrieval test
    logger.info("5. Testing feature retrieval...")
    try:
        # Create test entity dataframe
        entity_df = pd.DataFrame({
            'TransactionID': [1, 2, 3],
            'event_timestamp': pd.to_datetime(['2025-06-28', '2025-06-28', '2025-06-28'])
        })
        
        # Get first feature view
        fv = feature_views[0]
        test_features = [f"{fv.name}:{fv.features[0].name}"]
        
        logger.info(f"Testing with features: {test_features}")
        
        result_df = store.get_historical_features(
            entity_df=entity_df,
            features=test_features
        ).to_df()
        
        logger.info(f"✅ Feature retrieval successful: {result_df.shape}")
        logger.info(f"Result columns: {list(result_df.columns)}")
        
    except Exception as e:
        logger.error(f"❌ Feature retrieval failed: {str(e)}")
        return False
    
    logger.info("=== Diagnosis Complete ===")
    logger.info("✅ All checks passed! Feast setup is working correctly.")
    return True

def fix_common_issues():
    """Fix common Feast setup issues"""
    
    feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
    
    logger.info("=== Attempting to fix common issues ===")
    
    # Fix 1: Apply feature definitions
    logger.info("1. Applying feature definitions...")
    try:
        os.chdir(feast_repo_path)
        result = os.system("feast apply")
        if result == 0:
            logger.info("✅ feast apply successful")
        else:
            logger.error("❌ feast apply failed")
    except Exception as e:
        logger.error(f"❌ Error running feast apply: {str(e)}")
    
    # Fix 2: Check and create data directory
    logger.info("2. Ensuring data directory exists...")
    data_dir = f"{feast_repo_path}/data"
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"✅ Data directory ready: {data_dir}")
    
    # Fix 3: Registry cleanup
    logger.info("3. Refreshing registry...")
    try:
        store = FeatureStore(repo_path=".")
        store.refresh_registry()
        logger.info("✅ Registry refreshed")
    except Exception as e:
        logger.error(f"❌ Registry refresh failed: {str(e)}")

def main():
    """Main diagnostic function"""
    
    if not diagnose_feast_setup():
        logger.info("Issues found. Attempting fixes...")
        fix_common_issues()
        
        logger.info("Re-running diagnosis after fixes...")
        diagnose_feast_setup()

if __name__ == "__main__":
    main()