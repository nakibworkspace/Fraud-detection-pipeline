#!/usr/bin/env python3
"""
Inspect the fraud_features feature view to see what's available
"""

import os
import pandas as pd
from feast import FeatureStore
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_fraud_features():
    """Inspect what's in the fraud_features feature view"""
    
    feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
    
    try:
        os.chdir(feast_repo_path)
        store = FeatureStore(repo_path=".")
        
        logger.info("=== Feast Feature Store Inspection ===")
        
        # Get all feature views
        feature_views = store.list_feature_views()
        logger.info(f"Available feature views: {[fv.name for fv in feature_views]}")
        
        # Inspect each feature view
        for fv in feature_views:
            logger.info(f"\n=== Feature View: {fv.name} ===")
            logger.info(f"Number of features: {len(fv.features)}")
            logger.info(f"TTL: {fv.ttl}")
            logger.info(f"Entities: {fv.entities}")
            
            logger.info("Features:")
            for i, feature in enumerate(fv.features):
                logger.info(f"  {i+1:2d}. {feature.name:<25} ({feature.dtype})")
                if i >= 19:  # Show first 20 features
                    logger.info(f"  ... and {len(fv.features) - 20} more features")
                    break
        
        # Test feature retrieval with the fraud_features view
        fraud_fv = None
        for fv in feature_views:
            if fv.name == "fraud_features":
                fraud_fv = fv
                break
        
        if fraud_fv:
            logger.info(f"\n=== Testing Feature Retrieval ===")
            
            # Create test entity dataframe
            entity_df = pd.DataFrame({
                'TransactionID': [1, 2, 3],
                'event_timestamp': [datetime.now() - timedelta(hours=1)] * 3
            })
            
            # Test with first few features (non-target)
            test_features = []
            for feature in fraud_fv.features[:5]:
                if feature.name != 'isFraud':
                    test_features.append(f"fraud_features:{feature.name}")
            
            if test_features:
                logger.info(f"Testing with features: {test_features}")
                
                try:
                    result_df = store.get_historical_features(
                        entity_df=entity_df,
                        features=test_features
                    ).to_df()
                    
                    logger.info(f"✅ Success! Retrieved {result_df.shape[0]} rows and {result_df.shape[1]} columns")
                    logger.info(f"Columns: {list(result_df.columns)}")
                    
                    # Show sample data
                    logger.info("Sample data:")
                    logger.info(result_df.head())
                    
                except Exception as e:
                    logger.error(f"❌ Feature retrieval failed: {str(e)}")
            else:
                logger.warning("No suitable test features found")
        
        return True
        
    except Exception as e:
        logger.error(f"Error inspecting features: {str(e)}")
        return False

def test_training_pipeline_compatibility():
    """Test if the training pipeline would work with current setup"""
    
    feast_repo_path = '/root/code/fraud_detection_pipeline/fraud_detection_features'
    
    try:
        os.chdir(feast_repo_path)
        store = FeatureStore(repo_path=".")
        
        logger.info("\n=== Training Pipeline Compatibility Test ===")
        
        # Simulate what the training pipeline does
        feature_views = store.list_feature_views()
        
        if not feature_views:
            logger.error("❌ No feature views found")
            return False
        
        # Use first available feature view
        primary_fv = feature_views[0]
        logger.info(f"Primary feature view: {primary_fv.name}")
        
        # Build feature references
        feature_refs = []
        priority_features = [
            'TransactionAmt', 'card1', 'card2', 'addr1', 'addr2', 
            'dist1', 'dist2', 'card1_freq', 'card1_count', 'addr1_count'
        ]
        
        # Add priority features first
        for feature in primary_fv.features:
            if feature.name != 'isFraud' and feature.name in priority_features:
                feature_refs.append(f"{primary_fv.name}:{feature.name}")
        
        # Add other features if needed
        if len(feature_refs) < 10:
            for feature in primary_fv.features:
                if feature.name != 'isFraud':
                    feature_ref = f"{primary_fv.name}:{feature.name}"
                    if feature_ref not in feature_refs:
                        feature_refs.append(feature_ref)
                        if len(feature_refs) >= 10:
                            break
        
        logger.info(f"Would use {len(feature_refs)} features:")
        for i, ref in enumerate(feature_refs):
            logger.info(f"  {i+1}. {ref}")
        
        if feature_refs:
            logger.info("✅ Training pipeline should work with these features")
            return True
        else:
            logger.error("❌ No suitable features found for training")
            return False
        
    except Exception as e:
        logger.error(f"Error testing training pipeline compatibility: {str(e)}")
        return False

def main():
    """Main inspection function"""
    
    if inspect_fraud_features():
        test_training_pipeline_compatibility()
    else:
        logger.error("Feature inspection failed")

if __name__ == "__main__":
    main()