import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Generate Synthetic Dataset
np.random.seed(42)
n_samples = 1000

temperature = np.random.uniform(10, 50, n_samples)
humidity = np.random.uniform(20, 100, n_samples)
rainfall = np.random.uniform(0, 300, n_samples)
wind_speed = np.random.uniform(0, 150, n_samples)
regions = np.random.choice(["North", "South", "East", "West"], n_samples)

# Define Disaster Logic (with noise for realism)
conditions = []
for t, h, r, w in zip(temperature, humidity, rainfall, wind_speed):
    if r > 200 and w < 80:
        conditions.append("Flood")
    elif w > 100 and r > 150:
        conditions.append("Cyclone")
    elif h > 80 and t > 35:
        conditions.append("Heatwave")
    else:
        conditions.append("No Disaster")

# Add slight randomness (important for realistic ML)
conditions = [
    c if np.random.rand() > 0.05 else np.random.choice(["Flood", "Cyclone", "Heatwave", "No Disaster"])
    for c in conditions
]

# Create DataFrame
data = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "rainfall": rainfall,
    "wind_speed": wind_speed,
    "region": regions,
    "disaster_type": conditions
})

# Split Data
X = data.drop("disaster_type", axis=1)
y = data["disaster_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing Pipeline
numeric_features = ["temperature", "humidity", "rainfall", "wind_speed"]
categorical_features = ["region"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Model (Balanced + Realistic)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced"
)

# Full Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Train Model
pipeline.fit(X_train, y_train)

# Evaluate Model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Training Complete!")
print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# Save Model SAFELY
os.makedirs("model", exist_ok=True)

joblib.dump({
    "model": pipeline,
    "features": numeric_features + categorical_features,
    "classes": list(pipeline.classes_)
}, "model/model.pkl")

print("\n💾 Model saved successfully at: model/model.pkl")
print("🏷️ Classes learned:", pipeline.classes_)