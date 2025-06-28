import pandas as pd 

X_train = pd.read_csv('/tmp/X_train.csv')
print("Data types:")
print(X_train.dtypes)
print("\nColumns with object dtype:")
print(X_train.select_dtypes(include=['object']).columns.tolist())