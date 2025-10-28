import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os

# ---- Load Dataset ----
print("📂 Loading dataset...")
data = pd.read_csv("earthquake.csv")  # ✅ your dataset file name

# ---- Detect or Define Target Column ----
target_col = None
for col in data.columns:
    if col.lower() in ['alert', 'target', 'label', 'class']:
        target_col = col
        break

# If not found automatically, set manually
if target_col is None:
    # ⚠️ Replace 'YOUR_TARGET_COLUMN' with actual column name
    target_col = "YOUR_TARGET_COLUMN"
    print(f"⚠️ Target column not auto-detected. Using manual target: {target_col}")

# ---- Separate Features and Target ----
X = data.select_dtypes(include=[np.number])
y = data[target_col]

print(f"✅ Using target column: {target_col}")
print(f"📊 Feature columns: {list(X.columns)}")

# ---- Preprocess (Impute + Scale) ----
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---- Train Model ----
print("🚀 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---- Evaluate Model ----
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("\n✅ Model Training Complete!")
print(f"📈 Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, preds))

# ---- Save Model and Preprocessors ----
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/rf_model.joblib")
joblib.dump(imputer, "model/imputer.joblib")
joblib.dump(scaler, "model/scaler.joblib")

meta = {"features": list(X.columns), "target": target_col}
with open("model/metadata.json", "w") as f:
    json.dump(meta, f)

print("\n🎉 Model and preprocessors saved in 'model/' folder!")
print("Files saved:")
print(" - model/rf_model.joblib")
print(" - model/imputer.joblib")
print(" - model/scaler.joblib")
print(" - model/metadata.json")
