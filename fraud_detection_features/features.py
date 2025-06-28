from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String
from datetime import timedelta

# Define the Entity
transaction = Entity(
    name="transaction",
    join_keys=["TransactionID"],
    value_type=ValueType.INT64,
    description="Transaction entity for fraud detection"
)

# Define the data source - CORRECTED PATH to match your project structure
data_source = FileSource(
    path="/root/code/fraud_detection_pipeline/fraud_detection_features/data/data_source.parquet",
    timestamp_field="TransactionDT",
    created_timestamp_column="TransactionDT"
)

# Define core transaction features (most important ones for fraud detection)
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[
        # Basic transaction info
        Field(name="TransactionAmt", dtype=Float64),
        Field(name="ProductCD", dtype=String),
        
        # Card features
        Field(name="card1", dtype=Float64),
        Field(name="card2", dtype=Float64),
        Field(name="card3", dtype=Float64),
        Field(name="card4", dtype=String),
        Field(name="card5", dtype=Float64),
        Field(name="card6", dtype=String),
        
        # Address features
        Field(name="addr1", dtype=Float64),
        Field(name="addr2", dtype=Float64),
        
        # Distance features
        Field(name="dist1", dtype=Float64),
        Field(name="dist2", dtype=Float64),
        
        # Email domain features
        Field(name="P_emaildomain", dtype=String),
        Field(name="R_emaildomain", dtype=String),
        
        # Device features
        Field(name="DeviceType", dtype=String),
        Field(name="DeviceInfo", dtype=String),
        
        # Frequency features (these are your engineered features)
        Field(name="card1_freq", dtype=Float64),
        Field(name="card2_freq", dtype=Float64),
        Field(name="card1_count", dtype=Int64),
        Field(name="card2_count", dtype=Int64),
        Field(name="addr1_count", dtype=Int64),
        Field(name="addr2_count", dtype=Int64),
        
        # Derived features
        Field(name="TransactionAmt_to_mean", dtype=Float64),
        Field(name="TransactionCents", dtype=Int64),
        
        # Target variable
        Field(name="isFraud", dtype=Int64),
    ],
    source=data_source,
    tags={"team": "fraud_detection", "env": "production"}
)

# Define C features (customer behavior features)
c_features = FeatureView(
    name="c_features",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[
        Field(name="C1", dtype=Float64),
        Field(name="C2", dtype=Float64),
        Field(name="C3", dtype=Float64),
        Field(name="C4", dtype=Float64),
        Field(name="C5", dtype=Float64),
        Field(name="C6", dtype=Float64),
        Field(name="C7", dtype=Float64),
        Field(name="C8", dtype=Float64),
        Field(name="C9", dtype=Float64),
        Field(name="C10", dtype=Float64),
        Field(name="C11", dtype=Float64),
        Field(name="C12", dtype=Float64),
        Field(name="C13", dtype=Float64),
        Field(name="C14", dtype=Float64),
    ],
    source=data_source,
    tags={"team": "fraud_detection", "type": "customer_behavior"}
)

# Define M features (match features)
m_features = FeatureView(
    name="m_features", 
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[
        Field(name="M1", dtype=String),
        Field(name="M2", dtype=String),
        Field(name="M3", dtype=String),
        Field(name="M4", dtype=String),
        Field(name="M5", dtype=String),
        Field(name="M6", dtype=String),
        Field(name="M7", dtype=String),
        Field(name="M8", dtype=String),
        Field(name="M9", dtype=String),
    ],
    source=data_source,
    tags={"team": "fraud_detection", "type": "match_features"}
)

# Define V features (Vesta engineered features) - only include some key ones to avoid too many features
v_features_1 = FeatureView(
    name="v_features_1",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[
        Field(name="V1", dtype=Float64),
        Field(name="V2", dtype=Float64),
        Field(name="V3", dtype=Float64),
        Field(name="V4", dtype=Float64),
        Field(name="V5", dtype=Float64),
        Field(name="V6", dtype=Float64),
        Field(name="V7", dtype=Float64),
        Field(name="V8", dtype=Float64),
        Field(name="V9", dtype=Float64),
        Field(name="V10", dtype=Float64),
        Field(name="V11", dtype=Float64),
        Field(name="V12", dtype=Float64),
        Field(name="V13", dtype=Float64),
        Field(name="V14", dtype=Float64),
        Field(name="V15", dtype=Float64),
        Field(name="V16", dtype=Float64),
        Field(name="V17", dtype=Float64),
        Field(name="V18", dtype=Float64),
        Field(name="V19", dtype=Float64),
        Field(name="V20", dtype=Float64),
    ],
    source=data_source,
    tags={"team": "fraud_detection", "type": "vesta_features"}
)

# Define ID features (identity features) - key ones only
id_features = FeatureView(
    name="id_features",
    entities=[transaction],
    ttl=timedelta(days=365), 
    schema=[
        Field(name="id_01", dtype=Float64),
        Field(name="id_02", dtype=Float64),
        Field(name="id_03", dtype=Float64),
        Field(name="id_04", dtype=Float64),
        Field(name="id_05", dtype=Float64),
        Field(name="id_06", dtype=Float64),
        Field(name="id_07", dtype=Float64),
        Field(name="id_08", dtype=Float64),
        Field(name="id_09", dtype=Float64),
        Field(name="id_10", dtype=Float64),
        Field(name="id_11", dtype=String),
        Field(name="id_12", dtype=String),
        Field(name="id_13", dtype=Float64),
        Field(name="id_14", dtype=Float64),
        Field(name="id_15", dtype=String),
        Field(name="id_16", dtype=String),
        Field(name="id_17", dtype=Float64),
        Field(name="id_18", dtype=Float64),
        Field(name="id_19", dtype=Float64),
        Field(name="id_20", dtype=Float64),
    ],
    source=data_source,
    tags={"team": "fraud_detection", "type": "identity_features"}
)