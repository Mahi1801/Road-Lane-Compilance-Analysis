import pandas as pd
import os

LOG_PATH = "outputs/logs/traffic_log.csv"

# ── Load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(LOG_PATH):
    print("ERROR: traffic_log.csv not found. Run main_pipeline.py first.")
    exit()

df = pd.read_csv(LOG_PATH)
df["violation"] = df["violation"].astype(str).str.lower().isin(["true", "1"])

print("\n" + "="*55)
print("  INDIA LANE ANALYTICS — SUMMARY REPORT")
print("="*55)

# ── Basic counts ──────────────────────────────────────────────────────────────
total_frames    = df["frame_id"].nunique()
total_vehicles  = df["track_id"].nunique()
total_records   = len(df)
total_violators = df[df["violation"]]["track_id"].nunique()
overall_rate    = round(total_violators / total_vehicles * 100, 1) if total_vehicles > 0 else 0

print(f"\n📊 OVERVIEW")
print(f"   Frames analysed   : {total_frames}")
print(f"   Total records     : {total_records}")
print(f"   Unique vehicles   : {total_vehicles}")
print(f"   Violating vehicles: {total_violators}")
print(f"   Overall viol. rate: {overall_rate}%")

# ── Violation rate by lane ────────────────────────────────────────────────────
print(f"\n🛣️  VIOLATION RATE BY LANE")
print(f"   {'Lane':<10} {'Total':>8} {'Violators':>10} {'Rate %':>8}")
print(f"   {'-'*40}")

lane_labels = {1:"Cyclist", 2:"Bikes", 3:"Cars", 4:"Trucks"}
for lane_id in sorted(df["lane_id"].dropna().unique()):
    lane_df    = df[df["lane_id"] == lane_id]
    total      = lane_df["track_id"].nunique()
    violators  = lane_df[lane_df["violation"]]["track_id"].nunique()
    rate       = round(violators / total * 100, 1) if total > 0 else 0
    label      = lane_labels.get(int(lane_id), f"Lane{int(lane_id)}")
    print(f"   Lane {int(lane_id)} ({label:<8}): {total:>5} vehicles  {violators:>5} violators  {rate:>6}%")

# ── Violation rate by vehicle category ───────────────────────────────────────
print(f"\n🚗 VIOLATION RATE BY VEHICLE TYPE")
print(f"   {'Category':<12} {'Total':>8} {'Violators':>10} {'Rate %':>8}")
print(f"   {'-'*42}")

for cat in sorted(df["category"].dropna().unique()):
    cat_df    = df[df["category"] == cat]
    total     = cat_df["track_id"].nunique()
    violators = cat_df[cat_df["violation"]]["track_id"].nunique()
    rate      = round(violators / total * 100, 1) if total > 0 else 0
    print(f"   {cat:<12}: {total:>5} vehicles  {violators:>5} violators  {rate:>6}%")

# ── Congestion score over time ────────────────────────────────────────────────
print(f"\n⏱️  CONGESTION SCORE OVER TIME (every 150 frames)")
print(f"   {'Time Bin':<10} {'Vehicles':>10} {'Violations':>12} {'Viol Ratio':>12} {'Score':>8}")
print(f"   {'-'*56}")

df["time_bin"] = (df["frame_id"] // 150).astype(int)
for tbin, group in df.groupby("time_bin"):
    total      = len(group)
    violations = group["violation"].sum()
    v_ratio    = round(violations / total, 3) if total > 0 else 0
    score      = round(total * (1 + v_ratio), 2)
    print(f"   Bin {tbin:<6}: {total:>8} vehicles  {violations:>8} violations"
          f"  {v_ratio:>8}  {score:>8}")

# ── Top violating vehicles ────────────────────────────────────────────────────
print(f"\n🏆 TOP 10 MOST VIOLATING VEHICLES")
print(f"   {'Track ID':<10} {'Category':<12} {'Violation Frames':>16}")
print(f"   {'-'*42}")

top = (
    df[df["violation"]]
    .groupby(["track_id", "category"])["frame_id"]
    .count()
    .reset_index()
    .rename(columns={"frame_id": "violation_frames"})
    .sort_values("violation_frames", ascending=False)
    .head(10)
)
for _, row in top.iterrows():
    print(f"   ID {int(row['track_id']):<8}  {row['category']:<12}  {int(row['violation_frames']):>10} frames")

# ── Save summary to file ──────────────────────────────────────────────────────
os.makedirs("outputs/reports", exist_ok=True)
summary = {
    "total_frames":     total_frames,
    "total_vehicles":   total_vehicles,
    "total_violators":  total_violators,
    "overall_rate_pct": overall_rate
}
pd.DataFrame([summary]).to_csv("outputs/reports/summary.csv", index=False)
print(f"\n✅ Summary saved to outputs/reports/summary.csv")
print("="*55)
print("\nNext step: run   streamlit run dashboard.py   to see the visual dashboard")