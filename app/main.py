import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

st.set_page_config(page_title="PhaseNet", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] p { color: #ffffff !important; opacity: 0.9; }
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 15px !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

CONFIDENCE_THRESHOLD = 0.80


@st.cache_resource
def load_model(model_path):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 2)
    )
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    return None


def is_simulation_snapshot(image: Image.Image):
    """
    Validates that the image is an OVITO molecular dynamics snapshot.

    OVITO Tachyon renders have a large uniform dark background with small
    colored atom blobs. Natural photos lack this large uniform region.

    Returns (is_valid: bool, error_message: str)
    """
    img = np.array(image.resize((224, 224))).astype(float)
    pixels = img.reshape(-1, 3)

    # Background pixels: near-black (Tachyon default) or near-white
    dark_bg  = np.all(pixels < 60,  axis=1).mean()
    white_bg = np.all(pixels > 210, axis=1).mean()
    bg_ratio = dark_bg + white_bg

    if bg_ratio < 0.20:
        return False, (
            "This does not appear to be a simulation snapshot. "
            "Please upload an OVITO-rendered molecular dynamics image "
            "(LAMMPS trajectory snapshot with PTM coloring)."
        )

    if bg_ratio > 0.99:
        return False, "Image appears to be blank or empty."

    return True, ""


def predict(image: Image.Image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item()


# --- UI ---
st.title("PhaseNet: Deep Learning for Atomic Structures")
st.markdown("""
    **Developer:** Anson Antony | **Institution:** TU Bergakademie Freiberg

    ResNet-18 CNN trained via Distributed Data Parallel (DDP) to classify molecular dynamics
    snapshots into **Solid (Crystalline FCC)** or **Liquid (Amorphous)** phases.
""")
st.divider()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("Upload Simulation Snapshot")
    uploaded_file = st.file_uploader("Drop a PNG/JPG snapshot here", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", width=400)

with col2:
    st.subheader("Analysis Results")

    if uploaded_file:
        # Step 1: Validate image
        valid, error_msg = is_simulation_snapshot(image)

        if not valid:
            st.error(f"**Invalid Input**\n\n{error_msg}")
            st.info(
                "Accepted inputs: OVITO-rendered snapshots of molecular dynamics simulations "
                "(e.g., LAMMPS trajectory frames rendered with Tachyon)."
            )
        else:
            model = load_model("models/crystalline_classifier.pt")
            if model:
                with st.spinner('Running inference...'):
                    label_idx, confidence = predict(image, model)

                # Step 2: Confidence threshold check
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(
                        f"**Low Confidence ({confidence*100:.1f}%)**\n\n"
                        "The model is not confident enough to classify this image. "
                        "Please ensure the snapshot is a properly rendered OVITO simulation frame."
                    )
                else:
                    labels = ["Liquid (Amorphous)", "Solid (Crystalline)"]
                    result = labels[label_idx]

                    m1, m2 = st.columns(2)
                    m1.metric("Predicted Phase", result)
                    m2.metric("Confidence", f"{confidence*100:.2f}%")

                    if label_idx == 1:
                        st.success("**Solid Phase Detected**: Significant FCC symmetry identified.")
                    else:
                        st.warning("**Liquid Phase Detected**: System appears to be in a disordered/melted state.")

                    st.write("Prediction Certainty:")
                    st.progress(confidence)
    else:
        st.info("Waiting for image upload to begin analysis...")

st.divider()
with st.expander("View Technical Implementation Details"):
    t1, t2 = st.columns(2)
    with t1:
        st.write("**Model Training Architecture**")
        st.markdown("""
        * **Base Model:** ResNet-18 (partial fine-tune: layer3 + layer4 + FC)
        * **HPC Setup:** 2× NVIDIA RTX 4000 GPUs
        * **Framework:** PyTorch Distributed Data Parallel (DDP)
        * **Optimization:** Mixed Precision (AMP) + Early Stopping
        * **Dataset:** 1,111 LAMMPS simulation snapshots
        """)
    with t2:
        st.write("**Physics & Dataset**")
        st.markdown("""
        * **Material:** Copper (Cu)
        * **Simulation:** LAMMPS Molecular Dynamics (10 runs, varied seeds)
        * **Ground Truth:** Polyhedral Template Matching (PTM) via OVITO
        * **Input Size:** 224 × 224 RGB snapshots
        * **Split:** 70% train / 15% val / 15% test
        """)

st.markdown(
    "<div style='text-align: center; color: white; opacity: 0.7; font-size: 0.8rem;'>"
    "© 2026 PhaseNet Project | <a href='https://ansonantony.tech' target='_blank' "
    "style='color: white; text-decoration: none;'>ansonantony.tech</a>"
    "</div>",
    unsafe_allow_html=True
)
