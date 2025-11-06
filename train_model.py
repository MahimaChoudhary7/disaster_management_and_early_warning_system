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

# ============================
# Generate Synthetic Dataset
# ============================
np.random.seed(42)
n_samples = 1000

temperature = np.random.uniform(10, 50, n_samples)       # °C
humidity = np.random.uniform(20, 100, n_samples)          # %
rainfall = np.random.uniform(0, 300, n_samples)           # mm
wind_speed = np.random.uniform(0, 150, n_samples)         # km/h
regions = np.random.choice(["North", "South", "East", "West"], n_samples)

# ============================
# Logical rules for disasters
# ============================
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

data = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "rainfall": rainfall,
    "wind_speed": wind_speed,
    "region": regions,
    "disaster_type": conditions
})

# ============================
# Split Data
# ============================
X = data.drop("disaster_type", axis=1)
y = data["disaster_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# Preprocessing & Model Pipeline
# ============================
numeric_features = ["temperature", "humidity", "rainfall", "wind_speed"]
categorical_features = ["region"]

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", model)])

# ============================
# Train Model
# ============================
pipeline.fit(X_train, y_train)

# ============================
# Evaluate Model
# ============================
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Training Complete!")
print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# ============================
# Save Trained Model
# ============================
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/model.pkl")

print("\n💾 Model saved successfully at: model/model.pkl")
print("🏷️ Classes learned:", pipeline.classes_)
