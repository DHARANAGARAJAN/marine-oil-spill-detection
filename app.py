import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Marine Oil Spill Detection",
    page_icon="🌊",
    layout="centered"
)

st.title("🌊 Marine Oil Spill Detection using SAR Images")
st.markdown(
    """
    **Problem:** Marine oil spills damage marine ecosystems and coastal life.  
    **Solution:** This system detects oil spill regions from SAR satellite images using deep learning.

    🟦 **Sky Blue** → Clean Sea  
    🟩 **Dark Green** → Oil Spill Area
    """
)

# -------------------------------
# Load Model safely
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "marine_oil_spill_model.h5",
        compile=False
    )

model = load_model()

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload SAR Image",
    type=["jpg", "jpeg", "png"]
)

threshold = st.slider(
    "🎚 Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input SAR Image", use_column_width=True)

    if st.button("🚨 Detect Oil Spill"):
        with st.spinner("Analyzing image..."):
            img = image.resize((128, 128))
            arr = np.array(img).astype("float32") / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr)[0, :, :, 0]
            mask = pred > threshold

            output = np.zeros_like(arr[0])
            output[~mask] = [135/255, 206/255, 235/255]  # Sky blue
            output[mask] = [0, 100/255, 0]               # Dark green

            result = Image.fromarray((output * 255).astype("uint8"))

        st.success("Oil Spill Detection Completed ✅")
        st.image(result, caption="Detected Oil Spill Area", use_column_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "🔬 **Tech Stack:** SAR Imaging | Deep Learning | TensorFlow | Streamlit  \n"
    "🌍 **Use Case:** Marine Pollution Monitoring & Environmental Protection"
)
