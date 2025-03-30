import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("AB_NYC_2019.csv")

# Filter outliers
df = df[df["price"] < 500]

# Basic preprocessing
df = df.dropna(subset=["reviews_per_month", "last_review"])
df["month"] = pd.to_datetime(df["last_review"]).dt.month

features = ["neighbourhood_group", "room_type", "minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365", "month"]
target = "price"

X = df[features]
y = df[target]

# Categorical and numerical columns
cat_cols = ["neighbourhood_group", "room_type"]
num_cols = [col for col in features if col not in cat_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

pipeline = Pipeline([
    ("pre", preprocessor),
    ("model", XGBRegressor(n_estimators=100, max_depth=5, random_state=42))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "app/model.pkl")
print("✅ Model trained and saved to app/model.pkl")