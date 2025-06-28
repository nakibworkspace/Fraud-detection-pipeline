#!/usr/bin/env python3
"""
Debug data quality issues in the training pipeline
"""

import os
import pandas as pd
import numpy as np
from feast import FeatureStore
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_feast_data_quality():
    """Check data quality issues between Feast and source data"""
    
    feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
    
    try:
        os.chdir(feast_repo_path)
        
        # Check source data first
        source_data_path = f"{feast_repo_path}/data/data_source.parquet"
        source_data = pd.read_parquet(source_data_path)
        
        logger.info("=== Source Data Analysis ===")
        logger.info(f"Source data shape: {source_data.shape}")
        logger.info(f"TransactionID range: {source_data['TransactionID'].min()} to {source_data['TransactionID'].max()}")
        
        if 'isFraud' in source_data.columns:
            fraud_stats = source_data['isFraud'].value_counts()
            fraud_nulls = source_data['isFraud'].isnull().sum()
            logger.info(f"isFraud distribution: {dict(fraud_stats)}")
            logger.info(f"isFraud nulls: {fraud_nulls}")
            logger.info(f"isFraud dtype: {source_data['isFraud'].dtype}")
        
        # Now check Feast data
        logger.info("\n=== Feast Data Analysis ===")
        
        store = FeatureStore(repo_path=".")
        
        # Create entity dataframe like in training pipeline
        entity_df = pd.DataFrame({
            'TransactionID': source_data['TransactionID'][:100],  # Use first 100 for testing
            'event_timestamp': pd.to_datetime(source_data['TransactionDT'][:100], unit='s')
        })
        
        logger.info(f"Entity dataframe shape: {entity_df.shape}")
        logger.info(f"Entity TransactionID range: {entity_df['TransactionID'].min()} to {entity_df['TransactionID'].max()}")
        
        # Get feature views
        feature_views = store.list_feature_views()
        if not feature_views:
            logger.error("No feature views found!")
            return False
        
        primary_fv = feature_views[0]
        logger.info(f"Using feature view: {primary_fv.name}")
        
        # Get features (excluding target)
        feature_refs = []
        for feature in primary_fv.features[:5]:  # Just test with first 5
            if feature.name != 'isFraud':
                feature_refs.append(f"{primary_fv.name}:{feature.name}")
        
        logger.info(f"Testing with features: {feature_refs}")
        
        # Get historical features
        feast_df = store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()
        
        logger.info(f"Feast data shape: {feast_df.shape}")
        logger.info(f"Feast TransactionID range: {feast_df['TransactionID'].min()} to {feast_df['TransactionID'].max()}")
        
        # Check merge compatibility
        logger.info("\n=== Merge Analysis ===")
        
        # Check TransactionID overlap
        source_ids = set(source_data['TransactionID'])
        feast_ids = set(feast_df['TransactionID'])
        
        overlap = source_ids.intersection(feast_ids)
        logger.info(f"Source TransactionIDs: {len(source_ids)}")
        logger.info(f"Feast TransactionIDs: {len(feast_ids)}")
        logger.info(f"Overlapping TransactionIDs: {len(overlap)}")
        logger.info(f"Overlap percentage: {len(overlap)/len(feast_ids)*100:.1f}%")
        
        # Test the merge
        if 'isFraud' in source_data.columns:
            # Left join (like in original code)
            left_merge = feast_df.merge(
                source_data[['TransactionID', 'isFraud']], 
                on='TransactionID', 
                how='left'
            )
            
            # Inner join (like in fixed code)
            inner_merge = feast_df.merge(
                source_data[['TransactionID', 'isFraud']], 
                on='TransactionID', 
                how='inner'
            )
            
            logger.info(f"Left merge result: {left_merge.shape}")
            logger.info(f"Left merge isFraud nulls: {left_merge['isFraud'].isnull().sum()}")
            
            logger.info(f"Inner merge result: {inner_merge.shape}")
            logger.info(f"Inner merge isFraud nulls: {inner_merge['isFraud'].isnull().sum()}")
            
            if left_merge['isFraud'].isnull().sum() > 0:
                logger.warning("Left merge produces NaN values in target!")
                missing_ids = left_merge[left_merge['isFraud'].isnull()]['TransactionID'].tolist()
                logger.info(f"Missing IDs (first 10): {missing_ids[:10]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data quality check: {str(e)}")
        return False

def test_fixed_training_logic():
    """Test the fixed training data preparation logic"""
    
    feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
    
    try:
        os.chdir(feast_repo_path)
        
        logger.info("\n=== Testing Fixed Training Logic ===")
        
        # Simulate the training pipeline
        store = FeatureStore(repo_path=".")
        source_data = pd.read_parquet(f"{feast_repo_path}/data/data_source.parquet")
        
        # Create small entity dataframe
        entity_df = pd.DataFrame({
            'TransactionID': source_data['TransactionID'][:50],
            'event_timestamp': pd.to_datetime(source_data['TransactionDT'][:50], unit='s')
        })
        
        # Get features
        feature_views = store.list_feature_views()
        primary_fv = feature_views[0]
        
        feature_refs = []
        for feature in primary_fv.features[:5]:
            if feature.name != 'isFraud':
                feature_refs.append(f"{primary_fv.name}:{feature.name}")
        
        feast_df = store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()
        
        # Test merge with inner join
        if 'isFraud' in source_data.columns:
            full_df = feast_df.merge(
                source_data[['TransactionID', 'isFraud']], 
                on='TransactionID', 
                how='inner'
            )
            
            logger.info(f"Merged data shape: {full_df.shape}")
            
            # Prepare features and target
            exclude_cols = ['TransactionID', 'event_timestamp']
            feature_columns = [col for col in full_df.columns if col not in exclude_cols and col != 'isFraud']
            
            X = full_df[feature_columns].copy()
            y = full_df['isFraud'].copy()
            
            logger.info(f"X shape: {X.shape}")
            logger.info(f"y shape: {y.shape}")
            logger.info(f"y nulls: {pd.isna(y).sum()}")
            logger.info(f"y dtype: {y.dtype}")
            logger.info(f"y distribution: {y.value_counts().to_dict()}")
            
            # Handle missing values
            if X.isnull().sum().sum() > 0:
                logger.info(f"X nulls before cleaning: {X.isnull().sum().sum()}")
                for col in X.columns:
                    if X[col].dtype in ['float64', 'int64']:
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0, inplace=True)
                logger.info(f"X nulls after cleaning: {X.isnull().sum().sum()}")
            
            # Test train_test_split
            from sklearn.model_selection import train_test_split
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                logger.info("✅ train_test_split successful!")
                logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
                logger.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
                
            except Exception as e:
                logger.error(f"❌ train_test_split failed: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing training logic: {str(e)}")
        return False

def main():
    """Main function"""
    
    if check_feast_data_quality():
        test_fixed_training_logic()
    else:
        logger.error("Data quality check failed")

if __name__ == "__main__":
    main()