import streamlit as st
import plotly.graph_objects as go
import pandas as pd
 
st.set_page_config(page_title="5-Run Average Comparison", layout="wide")
st.title("ML Library Comparison (Average)")
st.caption("Scikit-learn vs TensorFlow vs PyTorch")

# Data get after running each model in each library 3x3 = 9 runs 5 times, 9x5 = 45runs. See screenshots folder for avg
DATA = {
    "Housing": {
        "Scikit-learn": {"train": 0.0270, "predict": 0.0005, "memory": 2.264, "quality": 0.6488, "secondary": None},
        "TensorFlow":   {"train": 44.7189, "predict": 0.3942, "memory": 3.092, "quality": -3.1546, "secondary": None},
        "PyTorch":      {"train": 0.1576, "predict": 0.01696, "memory": 0.014, "quality": -3.1559, "secondary": None},
    },
    "Loan": {
        "Scikit-learn": {"train": 0.2600, "predict": 0.00102, "memory": 1.668, "quality": 0.8944, "secondary": 0.7586},
        "TensorFlow":   {"train": 78.2792, "predict": 0.2563, "memory": 7.77, "quality": 0.8938, "secondary": 0.7582},
        "PyTorch":      {"train": 0.1754, "predict": 0.00144, "memory": 0.01, "quality": 0.8451, "secondary": 0.7045},
    },
    "Churn": {
        "Scikit-learn": {"train": 0.0403, "predict": 0.0016, "memory": 0.88, "quality": 0.7925, "secondary": 0.5655},
        "TensorFlow":   {"train": 14.3636, "predict": 0.1137, "memory": 3.14, "quality": 0.7919, "secondary": 0.5754},
        "PyTorch":      {"train": 2.8843, "predict": 0.0077, "memory": 0.01, "quality": 0.2897, "secondary": 0.4208},
    },
}
 
LIBRARIES = ["Scikit-learn", "TensorFlow", "PyTorch"]
COLORS = {"Scikit-learn": "#2a78d6", "TensorFlow": "#eb6834", "PyTorch": "#1baf7a"}
QUALITY_LABEL = {"Housing": "R2 score (higher is better)", "Loan": "Accuracy", "Churn": "Accuracy"}
SECONDARY_LABEL = {"Housing": None, "Loan": "F1 score", "Churn": "F1 score"}
 
 
def normalize(values, invert=False):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.75 for _ in values]
    if invert:
        return [1 - (v - lo) / (hi - lo) for v in values]
    return [(v - lo) / (hi - lo) for v in values]
 
dataset = st.radio("Dataset", ["Housing", "Loan", "Churn"], horizontal=True)
rows = DATA[dataset]
 
train_vals = [rows[l]["train"] for l in LIBRARIES]
quality_vals = [rows[l]["quality"] for l in LIBRARIES]
memory_vals = [rows[l]["memory"] for l in LIBRARIES]
 
best_train = LIBRARIES[train_vals.index(min(train_vals))]
best_quality = LIBRARIES[quality_vals.index(max(quality_vals))]
best_memory = LIBRARIES[memory_vals.index(min(memory_vals))]
 
c1, c2, c3 = st.columns(3)
c1.metric("Fastest training (avg)", best_train, f"{min(train_vals):.4f}s")
c2.metric("Best quality (avg)", best_quality, f"{QUALITY_LABEL[dataset]}: {max(quality_vals):.4f}")
c3.metric("Lowest memory (avg)", best_memory, f"{min(memory_vals):.3f} MB")
 
st.divider()
 
st.subheader("Average training and prediction time (log scale)")
fig_speed = go.Figure()
fig_speed.add_trace(go.Bar(name="Train time (avg)", x=LIBRARIES, y=train_vals, marker_color="#2a78d6"))
fig_speed.add_trace(go.Bar(name="Predict time (avg)", x=LIBRARIES, y=[rows[l]["predict"] for l in LIBRARIES], marker_color="#eb6834"))
fig_speed.update_layout(barmode="group", yaxis_type="log", yaxis_title="Seconds (log scale)", height=380)
st.plotly_chart(fig_speed, use_container_width=True)
 
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("Average peak memory (MB)")
    fig_mem = go.Figure(go.Bar(x=LIBRARIES, y=memory_vals, marker_color=[COLORS[l] for l in LIBRARIES]))
    fig_mem.update_layout(yaxis_title="MB", height=340)
    st.plotly_chart(fig_mem, use_container_width=True)
 
with col2:
    st.subheader(f"Average {QUALITY_LABEL[dataset]}")
    fig_qual = go.Figure(go.Bar(x=LIBRARIES, y=quality_vals, marker_color=[COLORS[l] for l in LIBRARIES]))
    fig_qual.update_layout(height=340)
    fig_qual.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_qual, use_container_width=True)
 
if SECONDARY_LABEL[dataset]:
    st.subheader(f"Average {SECONDARY_LABEL[dataset]}")
    secondary_vals = [rows[l]["secondary"] for l in LIBRARIES]
    fig_sec = go.Figure(go.Bar(x=LIBRARIES, y=secondary_vals, marker_color=[COLORS[l] for l in LIBRARIES]))
    fig_sec.update_layout(height=340)
    st.plotly_chart(fig_sec, use_container_width=True)
 
st.divider()
 
st.subheader("Library scorecard — averaged (normalized, outer edge is better)")
 
train_score = normalize(train_vals, invert=True)
predict_score = normalize([rows[l]["predict"] for l in LIBRARIES], invert=True)
memory_score = normalize(memory_vals, invert=True)
quality_score = normalize(quality_vals, invert=False)
 
categories = ["Train speed", "Predict speed", "Memory efficiency", "Quality"]
 
fig_radar = go.Figure()
for i, lib in enumerate(LIBRARIES):
    values = [train_score[i], predict_score[i], memory_score[i], quality_score[i]]
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=lib,
        line_color=COLORS[lib],
    ))
 
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
    height=450,
    showlegend=True,
)
st.plotly_chart(fig_radar, use_container_width=True)
 
 