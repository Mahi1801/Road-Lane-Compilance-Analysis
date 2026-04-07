import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

LOG_PATH   = "outputs/logs/traffic_log.csv"
MODEL_PATH = "models/congestion_predictor.pkl"

# ── Load and build features ───────────────────────────────────────────────────
print("\n" + "="*55)
print("  CONGESTION PREDICTOR — TRAINING")
print("="*55)

if not os.path.exists(LOG_PATH):
    print("ERROR: traffic_log.csv not found. Run main_pipeline.py first.")
    exit()

df = pd.read_csv(LOG_PATH)
df["violation"] = df["violation"].astype(str).str.lower().isin(["true","1"])
df["time_bin"]  = (df["frame_id"] // 150).astype(int)

print(f"\nLoaded {len(df)} records across {df['time_bin'].nunique()} time bins.")

# ── Build feature table (one row per time bin) ────────────────────────────────
g = df.groupby("time_bin")

features = pd.DataFrame({
    "total_records":   g["track_id"].count(),
    "unique_vehicles": g["track_id"].nunique(),
    "violation_count": g["violation"].sum(),
    "violation_ratio": g["violation"].mean(),
}).fillna(0)

# Vehicle type counts per bin
for cat in ["truck", "car", "bike", "cyclist"]:
    cat_counts = (
        df[df["category"] == cat]
        .groupby("time_bin")["track_id"]
        .count()
    )
    features[cat] = cat_counts
features = features.fillna(0)

# Congestion score = total vehicles * (1 + violation ratio)
features["congestion_score"] = (
    features["total_records"] * (1 + features["violation_ratio"])
).round(2)

# Target = NEXT bin's congestion score
features["target"] = features["congestion_score"].shift(-1)
features = features.dropna()

print(f"Feature table built: {features.shape[0]} samples x {features.shape[1]-1} features")
print(f"\nFeature columns:")
for col in features.columns:
    print(f"  - {col}")

# ── Train / test split ────────────────────────────────────────────────────────
X = features.drop(columns=["target", "congestion_score"])
y = features["target"]

if len(X) < 6:
    print("\nNot enough time bins to train yet.")
    print("Run main_pipeline.py with more frames (increase MAX_FRAMES to 3000).")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── Train model ───────────────────────────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)

print(f"\n{'='*55}")
print(f"  MODEL RESULTS")
print(f"{'='*55}")
print(f"  MAE (Mean Absolute Error) : {mae:.2f}")
print(f"  R2  Score                 : {r2:.3f}")
print(f"  (R2 closer to 1.0 = better fit)")

# ── Feature importances ───────────────────────────────────────────────────────
print(f"\n  FEATURE IMPORTANCES")
print(f"  (which factors most predict next congestion)")
print(f"  {'-'*40}")
importances = sorted(
    zip(X.columns, model.feature_importances_),
    key=lambda x: -x[1]
)
for feat, imp in importances:
    bar = "█" * int(imp * 40)
    print(f"  {feat:<20} {imp:.3f}  {bar}")

# ── Save model ────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"\n✅ Model saved to {MODEL_PATH}")

# ── Predict next congestion ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  LIVE PREDICTION DEMO")
print(f"{'='*55}")
last_bin  = X.iloc[-1:]
next_pred = model.predict(last_bin)[0]
current   = features["congestion_score"].iloc[-1]

print(f"  Current congestion score : {current:.2f}")
print(f"  Predicted next score     : {next_pred:.2f}")

if next_pred > current * 1.1:
    print("  ⚠️  WARNING: Congestion likely to INCREASE")
elif next_pred < current * 0.9:
    print("  ✅  Congestion likely to DECREASE")
else:
    print("  ➡️  Congestion likely to stay STABLE")

print("="*55)
print("\nNext step: run   python report.py   to generate your final HTML report")