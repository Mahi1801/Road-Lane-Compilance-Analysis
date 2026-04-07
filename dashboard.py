import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Indian Road Lane Discipline Analysis",
    page_icon="🚦",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
LOG_PATH = "outputs/logs/traffic_log.csv"

if not os.path.exists(LOG_PATH):
    st.error("No data found. Run main_pipeline.py first.")
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv(LOG_PATH)
    df["violation"] = df["violation"].astype(str).str.lower().isin(["true","1"])
    return df

df = load_data()

LANE_LABELS = {1:"Lane 1 - Cyclist", 2:"Lane 2 - Bikes",
               3:"Lane 3 - Cars",    4:"Lane 4 - Trucks"}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚦 India Road Lane Discipline Analysis")
st.caption("Real-time lane violation detection and traffic congestion analysis")
st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total_vehicles  = df["track_id"].nunique()
total_frames    = df["frame_id"].nunique()
total_violators = df[df["violation"]]["track_id"].nunique()
overall_rate    = round(total_violators / total_vehicles * 100, 1) if total_vehicles > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("🚗 Total Vehicles",      total_vehicles)
k2.metric("🎞️ Frames Analysed",     total_frames)
k3.metric("⚠️ Violating Vehicles",  total_violators)
k4.metric("📊 Violation Rate",       f"{overall_rate}%")

st.divider()

# ── Row 1: Lane violations + Category violations ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Violation Rate by Lane")
    rows = []
    for lane_id in sorted(df["lane_id"].dropna().unique()):
        lane_df   = df[df["lane_id"] == lane_id]
        total     = lane_df["track_id"].nunique()
        violators = lane_df[lane_df["violation"]]["track_id"].nunique()
        rate      = round(violators / total * 100, 1) if total > 0 else 0
        rows.append({
            "Lane": LANE_LABELS.get(int(lane_id), f"Lane {int(lane_id)}"),
            "Violation Rate %": rate,
            "Total Vehicles": total
        })
    lane_df_plot = pd.DataFrame(rows)
    fig1 = px.bar(
        lane_df_plot,
        x="Lane", y="Violation Rate %",
        color="Violation Rate %",
        color_continuous_scale="Reds",
        text="Violation Rate %"
    )
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig1.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Violation Rate by Vehicle Type")
    rows2 = []
    for cat in sorted(df["category"].dropna().unique()):
        cat_df    = df[df["category"] == cat]
        total     = cat_df["track_id"].nunique()
        violators = cat_df[cat_df["violation"]]["track_id"].nunique()
        rate      = round(violators / total * 100, 1) if total > 0 else 0
        rows2.append({"Category": cat, "Violation Rate %": rate, "Total": total})
    cat_df_plot = pd.DataFrame(rows2).sort_values("Violation Rate %", ascending=False)
    fig2 = px.bar(
        cat_df_plot,
        x="Category", y="Violation Rate %",
        color="Violation Rate %",
        color_continuous_scale="Oranges",
        text="Violation Rate %"
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Row 2: Congestion score over time ─────────────────────────────────────────
st.subheader("Congestion Score Over Time")
df["time_bin"] = (df["frame_id"] // 150).astype(int)
cong = df.groupby("time_bin").agg(
    total_vehicles=("track_id", "count"),
    violations=("violation", "sum")
).reset_index()
cong["violation_ratio"] = (cong["violations"] / cong["total_vehicles"]).round(3)
cong["congestion_score"] = (cong["total_vehicles"] * (1 + cong["violation_ratio"])).round(2)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=cong["time_bin"], y=cong["congestion_score"],
    mode="lines+markers", name="Congestion Score",
    line=dict(color="crimson", width=2)
))
fig3.add_trace(go.Bar(
    x=cong["time_bin"], y=cong["violations"],
    name="Violations", opacity=0.4,
    marker_color="orange"
))
fig3.update_layout(
    xaxis_title="Time Bin",
    yaxis_title="Score / Count",
    height=350,
    legend=dict(orientation="h")
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Row 3: Heatmap of violations ──────────────────────────────────────────────
st.subheader("Where Violations Happen — Position Heatmap")
viol_df = df[df["violation"]]
if len(viol_df) > 0:
    fig4 = px.density_heatmap(
        viol_df,
        x="centroid_x", y="centroid_y",
        nbinsx=40, nbinsy=30,
        color_continuous_scale="Hot",
    )
    fig4.update_yaxes(autorange="reversed")
    fig4.update_layout(height=380)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("No violations detected yet.")

st.divider()

# ── Row 4: Vehicle type distribution pie ──────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Vehicle Type Distribution")
    type_counts = df.groupby("category")["track_id"].nunique().reset_index()
    type_counts.columns = ["Category", "Count"]
    fig5 = px.pie(
        type_counts, names="Category", values="Count",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig5.update_layout(height=320)
    st.plotly_chart(fig5, use_container_width=True)

with col4:
    st.subheader("Violations vs OK by Lane")
    rows3 = []
    for lane_id in sorted(df["lane_id"].dropna().unique()):
        lane_df   = df[df["lane_id"] == lane_id]
        violators = lane_df[lane_df["violation"]]["track_id"].nunique()
        ok        = lane_df[~lane_df["violation"]]["track_id"].nunique()
        label     = LANE_LABELS.get(int(lane_id), f"Lane {int(lane_id)}")
        rows3.append({"Lane": label, "Status": "Violation", "Count": violators})
        rows3.append({"Lane": label, "Status": "OK",        "Count": ok})
    stacked_df = pd.DataFrame(rows3)
    fig6 = px.bar(
        stacked_df,
        x="Lane", y="Count", color="Status",
        barmode="stack",
        color_discrete_map={"Violation": "crimson", "OK": "seagreen"}
    )
    fig6.update_layout(height=320)
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── Top violators table ───────────────────────────────────────────────────────
st.subheader("Top 20 Most Violating Vehicles")
top = (
    df[df["violation"]]
    .groupby(["track_id","category"])["frame_id"]
    .count()
    .reset_index()
    .rename(columns={"frame_id":"violation_frames"})
    .sort_values("violation_frames", ascending=False)
    .head(20)
)
st.dataframe(top, use_container_width=True)

# ── Raw data expander ─────────────────────────────────────────────────────────
with st.expander("View Raw Log Data (last 500 rows)"):
    st.dataframe(df.tail(500), use_container_width=True)

st.caption("India Road Lane Discipline Analysis | Built with YOLOv8 + DeepSORT + Streamlit")
