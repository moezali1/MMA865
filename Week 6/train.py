import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the data
data = pd.read_csv("diamond.csv")

# Split features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define numeric and categorical columns
numeric_features = ["Carat Weight"]
categorical_features = ["Cut", "Color", "Clarity", "Polish", "Symmetry", "Report"]

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Create pipeline
pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions on test set
y_pred = pipeline.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Save the pipeline
with open("train_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Pipeline saved as 'train_pipeline.pkl'")
