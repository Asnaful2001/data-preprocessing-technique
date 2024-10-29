import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

housing = fetch_california_housing(as_frame=True)
data = housing.data
data['target'] = housing.target

print("Original Data:")
print(data.head())

X = data.drop('target', axis=1)
y = data['target']

pipeline = Pipeline([
    ('scaler', MinMaxScaler())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

X_train_processed_df = pd.DataFrame(X_train_processed, columns=X.columns)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=X.columns)

print("\nProcessed Training Data:")
print(X_train_processed_df.head())
