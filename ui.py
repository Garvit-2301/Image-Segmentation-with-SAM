import streamlit as st
from PIL import Image
import numpy as np
import time
from ultralytics import SAM

st.title("Segment Anything Model (SAM) Demo")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

@st.cache_resource
def load_model():
    start = time.time()
    model = SAM("sam_b.pt")
    return model, time.time() - start

model, model_load_time = load_model()
st.write(f"Model loaded in {model_load_time:.2f} seconds.")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", use_column_width=True)
    start = time.time()
    results = model(image)
    inference_time = time.time() - start
    masks = results.masks.data.cpu().numpy()
    mask = np.any(masks, axis=0)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    st.image(mask_img, caption=f"Segmented mask (Inference: {inference_time:.2f}s)")
    st.image(np.array(image) * np.expand_dims(mask, axis=2), caption="Masked Image")