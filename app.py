import streamlit as st
import numpy as np
import cv2
from src.predict import load_model_for_inference

st.set_page_config(page_title="MRI Segmentation", layout="centered")
st.title("Medical Image Segmentation for Automated Diagnostics")

st.write("Upload an MRI image. The model will predict the anomaly mask.")

model = load_model_for_inference("models/best_unet.keras")

uploaded = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Could not read the uploaded image.")
    else:
        original = img.copy()

        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        x = img_resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, -1))

        pred = model.predict(x, verbose=0)[0, :, :, 0]
        pred_bin = (pred >= 0.5).astype(np.uint8)

        overlay = cv2.addWeighted(
            cv2.resize(original, (256, 256)),
            0.7,
            (pred_bin * 255).astype(np.uint8),
            0.3,
            0
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original MRI", use_container_width=True)
        with col2:
            st.image(pred_bin * 255, caption="Predicted Mask", use_container_width=True)

        st.subheader("Overlay")
        st.image(overlay, caption="MRI + Segmentation Overlay", use_container_width=True)