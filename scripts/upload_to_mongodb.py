import pandas as pd
from pymongo import MongoClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_to_mongodb():
    """Upload IEEE-CIS Fraud Detection dataset to MongoDB in chunks"""
    try:
        # Connect to MongoDB with explicit timeouts
        logger.info("Connecting to MongoDB...")
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        db = client['fraud_detection']
        
        # Test connection
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        
        # Define dataset paths
        data_path = '/root/code/fraud_detection_pipeline/data/'
        datasets = {
            'train_transaction': 'train_transaction.csv',
            'test_transaction': 'test_transaction.csv',
            'train_identity': 'train_identity.csv',
            'test_identity': 'test_identity.csv'
        }
        
        # Process each dataset in chunks
        chunk_size = 10000  # Adjust based on VM memory
        for collection_name, file_name in datasets.items():
            logger.info(f"Reading {file_name}...")
            collection = db[collection_name]
            collection.drop()  # Clear existing data
            
            # Read CSV in chunks
            chunk_iterator = pd.read_csv(data_path + file_name, chunksize=chunk_size)
            for i, chunk in enumerate(chunk_iterator):
                try:
                    logger.info(f"Processing chunk {i+1} of {collection_name}...")
                    records = chunk.to_dict('records')
                    collection.insert_many(records, ordered=False)
                    logger.info(f"Inserted chunk {i+1} into {collection_name}")
                except Exception as e:
                    logger.error(f"Error inserting chunk {i+1} into {collection_name}: {e}")
                    raise
            
            logger.info(f"Completed uploading {collection_name}. Total documents: {collection.count_documents({})}")
        
        logger.info("All datasets uploaded successfully")
        client.close()
    
    except Exception as e:
        logger.error(f"Error in upload_to_mongodb: {e}")
        raise

if __name__ == "__main__":
    upload_to_mongodb()