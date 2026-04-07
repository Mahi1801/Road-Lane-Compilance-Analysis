import pandas as pd
import os

LOG_PATH    = "outputs/logs/traffic_log.csv"
OUTPUT_PATH = "outputs/reports/report.html"

if not os.path.exists(LOG_PATH):
    print("ERROR: Run main_pipeline.py first.")
    exit()

df = pd.read_csv(LOG_PATH)
df["violation"] = df["violation"].astype(str).str.lower().isin(["true","1"])

LANE_LABELS = {
    1:"Lane 1 - Cyclist",
    2:"Lane 2 - Bikes",
    3:"Lane 3 - Cars",
    4:"Lane 4 - Trucks"
}

# ── Compute stats ─────────────────────────────────────────────────────────────
total_vehicles  = df["track_id"].nunique()
total_frames    = df["frame_id"].nunique()
total_violators = df[df["violation"]]["track_id"].nunique()
overall_rate    = round(total_violators / total_vehicles * 100, 1)

# Lane table
lane_rows = ""
for lid in sorted(df["lane_id"].dropna().unique()):
    lane_df   = df[df["lane_id"] == lid]
    total     = lane_df["track_id"].nunique()
    violators = lane_df[lane_df["violation"]]["track_id"].nunique()
    rate      = round(violators / total * 100, 1) if total > 0 else 0
    label     = LANE_LABELS.get(int(lid), f"Lane {int(lid)}")
    color     = "#e74c3c" if rate > 50 else "#e67e22" if rate > 25 else "#27ae60"
    lane_rows += f"""
    <tr>
        <td>{label}</td>
        <td>{total}</td>
        <td>{violators}</td>
        <td style="color:{color}; font-weight:bold">{rate}%</td>
    </tr>"""

# Category table
cat_rows = ""
for cat in sorted(df["category"].dropna().unique()):
    cat_df    = df[df["category"] == cat]
    total     = cat_df["track_id"].nunique()
    violators = cat_df[cat_df["violation"]]["track_id"].nunique()
    rate      = round(violators / total * 100, 1) if total > 0 else 0
    color     = "#e74c3c" if rate > 50 else "#e67e22" if rate > 25 else "#27ae60"
    cat_rows += f"""
    <tr>
        <td>{cat}</td>
        <td>{total}</td>
        <td>{violators}</td>
        <td style="color:{color}; font-weight:bold">{rate}%</td>
    </tr>"""

# Top violators table
top = (
    df[df["violation"]]
    .groupby(["track_id","category"])["frame_id"]
    .count()
    .reset_index()
    .rename(columns={"frame_id":"violation_frames"})
    .sort_values("violation_frames", ascending=False)
    .head(15)
)
top_rows = ""
for _, row in top.iterrows():
    top_rows += f"""
    <tr>
        <td>{int(row['track_id'])}</td>
        <td>{row['category']}</td>
        <td>{int(row['violation_frames'])}</td>
    </tr>"""

# ── Build HTML ────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>India Lane Analytics Report</title>
<style>
  body      {{ font-family: Arial, sans-serif; max-width: 1000px;
               margin: 40px auto; color: #222; padding: 0 20px; }}
  h1        {{ color: #c0392b; border-bottom: 3px solid #c0392b; padding-bottom: 10px; }}
  h2        {{ color: #2c3e50; margin-top: 36px; border-left: 4px solid #c0392b;
               padding-left: 12px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
               margin: 24px 0; }}
  .kpi      {{ background: #f8f9fa; border-radius: 10px; padding: 20px;
               text-align: center; border: 1px solid #dee2e6; }}
  .kpi .num {{ font-size: 2.2em; font-weight: bold; color: #c0392b; }}
  .kpi .lbl {{ font-size: 0.85em; color: #666; margin-top: 6px; }}
  table     {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th        {{ background: #2c3e50; color: white; padding: 10px 14px;
               text-align: left; }}
  td        {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
  tr:hover  {{ background: #f5f5f5; }}
  .footer   {{ margin-top: 48px; color: #999; font-size: 0.8em;
               border-top: 1px solid #eee; padding-top: 16px; }}
</style>
</head>
<body>

<h1>🚦 India Road Lane Discipline — Analytics Report</h1>
<p>Computer vision based analysis of lane violations and their
   correlation with traffic congestion on Indian roads.</p>

<div class="kpi-grid">
  <div class="kpi"><div class="num">{total_vehicles}</div>
    <div class="lbl">Total Vehicles</div></div>
  <div class="kpi"><div class="num">{total_frames}</div>
    <div class="lbl">Frames Analysed</div></div>
  <div class="kpi"><div class="num">{total_violators}</div>
    <div class="lbl">Violating Vehicles</div></div>
  <div class="kpi"><div class="num">{overall_rate}%</div>
    <div class="lbl">Overall Violation Rate</div></div>
</div>

<h2>Violation Rate by Lane</h2>
<table>
  <tr><th>Lane</th><th>Total Vehicles</th>
      <th>Violators</th><th>Violation Rate</th></tr>
  {lane_rows}
</table>

<h2>Violation Rate by Vehicle Type</h2>
<table>
  <tr><th>Category</th><th>Total Vehicles</th>
      <th>Violators</th><th>Violation Rate</th></tr>
  {cat_rows}
</table>

<h2>Top 15 Violating Vehicles</h2>
<table>
  <tr><th>Track ID</th><th>Category</th><th>Violation Frames</th></tr>
  {top_rows}
</table>

<h2>Key Findings</h2>
<ul>
  <li>Overall <strong>{overall_rate}%</strong> of detected vehicles
      were in violation of lane rules.</li>
  <li>Total of <strong>{total_violators}</strong> unique vehicles
      recorded violating lane discipline.</li>
  <li>Analysis based on <strong>{total_frames}</strong> video frames
      with YOLOv8 + DeepSORT tracking.</li>
</ul>

<div class="footer">
  Generated by India Lane Analytics Pipeline |
  YOLOv8 + DeepSORT + OpenCV + Python
</div>
</body>
</html>"""

os.makedirs("outputs/reports", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Report saved to {OUTPUT_PATH}")
print("Open it in your browser to view the full report.")