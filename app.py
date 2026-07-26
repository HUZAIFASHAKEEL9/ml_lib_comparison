import streamlit as st
import plotly.graph_objects as go
import pandas as pd
 
st.set_page_config(page_title="Single Run Comparison", layout="wide")
st.title("ML Library Comparison")
st.caption("Scikit-learn vs TensorFlow vs PyTorch")
 
data = {
    "Housing": {
        # secondary is f1 score, regression has no f1 score metric -> None
        "Scikit-learn": {"train": 0.1017, "predict": 0.0004, "memory": 3.80, "quality": 0.6488, "secondary": None},
        "TensorFlow":   {"train": 44.2915, "predict": 0.3888, "memory": 3.94, "quality": -3.1546, "secondary": None},
        "PyTorch":      {"train": 0.2291, "predict": 0.0006, "memory": 0.03, "quality": -3.1559, "secondary": None},
    },
    "Loan": {
        "Scikit-learn": {"train": 0.6177, "predict": 0.0020, "memory": 1.92, "quality": 0.8944, "secondary": 0.7586},
        "TensorFlow":   {"train": 77.5048, "predict": 0.2524, "memory": 6.87, "quality": 0.8933, "secondary": 0.7587},
        "PyTorch":      {"train": 0.1462, "predict": 0.0016, "memory": 0.01, "quality": 0.8630, "secondary": 0.7127},
    },
    "Churn": {
        "Scikit-learn": {"train": 0.0518, "predict": 0.0005, "memory": 1.15, "quality": 0.7925, "secondary": 0.5655},
        "TensorFlow":   {"train": 14.1908, "predict": 0.1142, "memory": 4.21, "quality": 0.7910, "secondary": 0.5764},
        "PyTorch":      {"train": 3.1002, "predict": 0.0109, "memory": 0.01, "quality": 0.2825, "secondary": 0.4056},
    },
}
 
libraries = ["Scikit-learn", "TensorFlow", "PyTorch"]
colors = {"Scikit-learn": "#2a78d6", "TensorFlow": "#eb6834", "PyTorch": "#1baf7a"}
quality_label = {"Housing": "R2 score (higher is better)", "Loan": "Accuracy", "Churn": "Accuracy"}
secondary_label = {"Housing": None, "Loan": "F1 score", "Churn": "F1 score"}
 
 
def normalize(values, invert=False):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.75 for _ in values]
    if invert:
        return [1 - (v - lo) / (hi - lo) for v in values]
    return [(v - lo) / (hi - lo) for v in values]
 
 
dataset = st.radio("Select Dataset", ["Housing", "Loan", "Churn"], horizontal=True)
rows = data[dataset]
 
train_vals = [rows[l]["train"] for l in libraries]
quality_vals = [rows[l]["quality"] for l in libraries]
memory_vals = [rows[l]["memory"] for l in libraries]
 
best_train = libraries[train_vals.index(min(train_vals))]
best_quality = libraries[quality_vals.index(max(quality_vals))]
best_memory = libraries[memory_vals.index(min(memory_vals))]
 
c1, c2, c3 = st.columns(3)
c1.metric("Fastest training", best_train, f"{min(train_vals):.4f}s")
c2.metric("Best quality", best_quality, f"{quality_label[dataset]}: {max(quality_vals):.4f}")
c3.metric("Lowest memory", best_memory, f"{min(memory_vals):.3f} MB")
 
st.divider()
 
st.subheader("Training and prediction time (log scale)")
fig_speed = go.Figure()
fig_speed.add_trace(go.Bar(name="Train time", x=libraries, y=train_vals, marker_color="#2a78d6"))
fig_speed.add_trace(go.Bar(name="Predict time", x=libraries, y=[rows[l]["predict"] for l in libraries], marker_color="#eb6834"))
fig_speed.update_layout(barmode="group", yaxis_type="log", yaxis_title="Seconds (log scale)", height=380)
st.plotly_chart(fig_speed, use_container_width=True)
 
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("Peak memory (MB)")
    fig_mem = go.Figure(go.Bar(x=libraries, y=memory_vals, marker_color=[colors[l] for l in libraries]))
    fig_mem.update_layout(yaxis_title="MB", height=340)
    st.plotly_chart(fig_mem, use_container_width=True)
 
with col2:
    st.subheader(quality_label[dataset])
    fig_qual = go.Figure(go.Bar(x=libraries, y=quality_vals, marker_color=[colors[l] for l in libraries]))
    fig_qual.update_layout(height=340)
    fig_qual.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_qual, use_container_width=True)
 
if secondary_label[dataset]:
    st.subheader(secondary_label[dataset])
    secondary_vals = [rows[l]["secondary"] for l in libraries]
    fig_sec = go.Figure(go.Bar(x=libraries, y=secondary_vals, marker_color=[colors[l] for l in libraries]))
    fig_sec.update_layout(height=340)
    st.plotly_chart(fig_sec, use_container_width=True)
 
st.divider()
 
st.subheader("Library scorecard (normalized — outer edge is better)")
 
train_score = normalize(train_vals, invert=True)
predict_score = normalize([rows[l]["predict"] for l in libraries], invert=True)
memory_score = normalize(memory_vals, invert=True)
quality_score = normalize(quality_vals, invert=False)
 
categories = ["Train speed", "Predict speed", "Memory efficiency", "Quality"]
 
fig_radar = go.Figure()
for i, lib in enumerate(libraries):
    values = [train_score[i], predict_score[i], memory_score[i], quality_score[i]]
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=lib,
        line_color=colors[lib],
    ))
 
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
    height=450,
    showlegend=True,
)
st.plotly_chart(fig_radar, use_container_width=True)
 