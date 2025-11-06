import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ========== Step 1: Load or create dataset ==========
data_path = "data/disaster_data.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

if not os.path.exists(data_path):
    print("⚠️ Dataset not found. Creating a sample dataset automatically...")
    import numpy as np
    np.random.seed(42)
    data = {
        "temperature": np.random.uniform(10, 45, 500),
        "humidity": np.random.uniform(20, 90, 500),
        "rainfall": np.random.uniform(0, 300, 500),
        "wind_speed": np.random.uniform(0, 100, 500),
        "region": np.random.choice(["North", "South", "East", "West"], 500),
        "disaster": np.random.choice([0, 1], 500, p=[0.8, 0.2])  # 0 = no disaster, 1 = disaster
    }
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    print("✅ Sample dataset created successfully!")
else:
    print("✅ Found existing dataset.")

# ========== Step 2: Read the dataset ==========
df = pd.read_csv(data_path)
print(f"📊 Dataset Loaded Successfully! Total Rows: {len(df)}")

# ========== Step 3: Preprocess the data ==========
df = pd.get_dummies(df, columns=["region"], drop_first=True)

X = df.drop("disaster", axis=1)
y = df["disaster"]

# ========== Step 4: Split the data ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("✂️ Data Split Completed.")

# ========== Step 5: Train Model ==========
print("🚀 Training RandomForest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ========== Step 6: Evaluate Model ==========
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# ========== Step 7: Save Model ==========
model_path = "model/model.pkl"
joblib.dump(model, model_path)
print(f"💾 Model saved successfully at: {model_path}")
