import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="PhaseNet",
    page_icon="🔬",
    layout="wide"
)

# --- Targeted Visibility Fix ---
# --- High-Contrast Visibility Fix ---
# --- White Text Visibility Fix ---
st.markdown("""
    <style>

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #ffffff !important;
        opacity: 0.9;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 15px !important;
        border-radius: 10px !important;
    }
    </style>

    """, unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    return None

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item()

# --- Header Section ---
st.title("PhaseNet: Deep Learning for Atomic Structures")
st.markdown("""
    **Developer:** Anson Antony | **Institution:** TU Bergakademie Freiberg
    
    This platform leverages a **ResNet18 CNN** trained via Distributed Data Parallel (DDP) to classify molecular dynamics snapshots into 
    **Solid (Crystalline)** or **Liquid (Amorphous)** phases.
""")
st.divider()

# --- Main Interface ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("Upload Simulation Snapshot")
    uploaded_file = st.file_uploader("Drop a PNG/JPG snapshot here", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        # Use fixed width to prevent blurriness
        st.image(image, caption="Original Input (224x224)", width=400)

with col2:
    st.subheader(" Analysis Results")
    if uploaded_file:
        model = load_model("models/crystalline_classifier.pt")
        if model:
            with st.spinner('Running HPC Inference...'):
                label_idx, confidence = predict(image, model)
                
            labels = ["Liquid (Amorphous)", "Solid (Crystalline)"]
            result = labels[label_idx]
            
            # Display Metrics
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Predicted Phase", result)
            m_col2.metric("Confidence Score", f"{confidence*100:.2f}%")
            
            # Detailed Interpretation
            if label_idx == 1:
                st.success(" **Solid Phase Detected**: Significant FCC symmetry identified.")
            else:
                st.warning(" **Liquid Phase Detected**: System appears to be in a melted/disordered state.")
            
            # Progress bar for visual appeal
            st.write("Prediction Certainty:")
            st.progress(confidence)
    else:
        st.info("Waiting for image upload to begin analysis...")

# --- Extra Details & Technical Specs ---
st.divider()
with st.expander("View Technical Implementation Details"):
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.write("**Model Training Architecture**")
        st.markdown("""
        * **Base Model:** ResNet-18
        * **HPC Setup:** 2x NVIDIA Quadro P4000 GPUs
        * **Framework:** PyTorch Distributed Data Parallel (DDP)
        * **Optimization:** Mixed Precision (AMP) for faster throughput
        """)
    with t_col2:
        st.write("**Physics & Dataset**")
        st.markdown("""
        * **Material:** Copper (Cu)
        * **Simulation:** LAMMPS Molecular Dynamics
        * **Ground Truth:** Polyhedral Template Matching (PTM) via OVITO
        * **Input Size:** 224 x 224 RGB snapshots
        """)

# st.caption("© 2026 PhaseNet Project | ansonantony.tech")
# --- Redirectable Caption at the bottom ---
st.markdown(
    "<div style='text-align: center; color: white; opacity: 0.7; font-size: 0.8rem;'>"
    "© 2026 PhaseNet Project | <a href='https://ansonantony.tech' target='_blank' "
    "style='color: white; text-decoration: none;'>ansonantony.tech</a>"
    "</div>", 
    unsafe_allow_html=True
)