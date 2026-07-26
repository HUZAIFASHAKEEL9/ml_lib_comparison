import streamlit as st
st.set_page_config(page_title="ML libraries recommendation", layout="centered")
st.title("ML Lib Recommendation")
st.write("Select your project's requirements below, then click **Recommend** "
    "to see which library — Scikit-learn, TensorFlow, or PyTorch — best fits your needs.")
st.divider()
# This will tell score, when selected score is sumed up and final answer depends on the hioghest score
SCORES = {
    "dataset_size": {
        "Small":  {"Scikit-learn": 3, "TensorFlow": 1, "PyTorch": 1},
        "Medium": {"Scikit-learn": 2, "TensorFlow": 2, "PyTorch": 2},
        "Large":  {"Scikit-learn": 0, "TensorFlow": 3, "PyTorch": 3},
    },
    "hardware": {
        "CPU": {"Scikit-learn": 3, "TensorFlow": 1, "PyTorch": 1},
        "GPU": {"Scikit-learn": 0, "TensorFlow": 3, "PyTorch": 3},
    },
    "accuracy": {
        "Slightly Low": {"Scikit-learn": 3, "TensorFlow": 1, "PyTorch": 1},
        "Normal":       {"Scikit-learn": 2, "TensorFlow": 2, "PyTorch": 2},
        "High":         {"Scikit-learn": 0, "TensorFlow": 3, "PyTorch": 3},
    },
    "purpose": {
        "Traditional ML": {"Scikit-learn": 4, "TensorFlow": 0, "PyTorch": 0},
        "Deep Learning":  {"Scikit-learn": 0, "TensorFlow": 4, "PyTorch": 2},
        "Research":       {"Scikit-learn": 0, "TensorFlow": 1, "PyTorch": 4},
    },
    "performance": {
        "Slightly Low": {"Scikit-learn": 3, "TensorFlow": 1, "PyTorch": 1},
        "Good":         {"Scikit-learn": 2, "TensorFlow": 2, "PyTorch": 2},
        "Best":         {"Scikit-learn": 0, "TensorFlow": 3, "PyTorch": 3},
    },
    "training_speed": {
        "Low":    {"Scikit-learn": 1, "TensorFlow": 3, "PyTorch": 3},
        "Normal": {"Scikit-learn": 2, "TensorFlow": 2, "PyTorch": 2},
        "High":   {"Scikit-learn": 3, "TensorFlow": 1, "PyTorch": 1},
    },
}

LIBRARY_INFO = {
    "Scikit-learn": {
        "tagline": "Best for classical ML, small-to-medium tabular data, and fast prototyping.",
        "color": "#F7931E",
    },
    "TensorFlow": {
        "tagline": "Best for deep learning, production deployment, and large-scale training.",
        "color": "#FF6F00",
    },
    "PyTorch": {
        "tagline": "Best for research, flexibility, and custom deep learning architectures.",
        "color": "#EE4C2C",
    },
}
 
# This function will sum scores across all parameters and returns the winning library and full score. 
def recommend_library(selections: dict):

    totals = {"Scikit-learn": 0, "TensorFlow": 0, "PyTorch": 0}
 
    for param_key, chosen_option in selections.items():
        option_scores = SCORES[param_key][chosen_option]
        for lib, points in option_scores.items():
            totals[lib] += points
 
    winner = max(totals, key=totals.get)
    return winner, totals
 
 
# user interface
col1, col2 = st.columns(2)
 
with col1:
    dataset_size = st.selectbox("Size of Dataset", ["Small", "Medium", "Large"])
    accuracy = st.selectbox("Accuracy Needed", ["Slightly Low", "Normal", "High"])
    performance = st.selectbox("Overall Performance Needed", ["Slightly Low", "Good", "Best"])
 
with col2:
    hardware = st.selectbox("Hardware", ["CPU", "GPU"])
    purpose = st.selectbox("Purpose", ["Traditional ML", "Deep Learning", "Research"])
    training_speed = st.selectbox("Training Speed Priority", ["Low", "Normal", "High"])
 
st.divider()
 

button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
with button_col2:
    recommend_clicked = st.button(" Recommend Library", use_container_width=True)
 
# after all selections, these parameter should go to recommendation function and then calculte it score and finally find winner
if recommend_clicked:
    selections = {
        "dataset_size": dataset_size,
        "hardware": hardware,
        "accuracy": accuracy,
        "purpose": purpose,
        "performance": performance,
        "training_speed": training_speed,
    }
 
    winner, totals = recommend_library(selections)
    info = LIBRARY_INFO[winner]
 
    st.divider()
    st.markdown(
        f"""
        <div style="text-align:center; padding: 20px; border-radius: 12px;
                    border: 2px solid {info['color']}; background-color: rgba(0,0,0,0.02);">
            <h2 style="color:{info['color']};"> Recommended: {winner}</h2>
            <p style="font-size:16px;">{info['tagline']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    st.write("")
    st.subheader("Score Breakdown")
    st.bar_chart(totals)
 
    with st.expander("See raw scores"):
        st.json(totals)