import streamlit as st
import time
from PIL import Image
import numpy as np
from ultralytics import SAM

st.title("Segment Anything Model (SAM) Demo")

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["png", "jpg", "jpeg"])

@st.cache_resource
def load_model():
    start = time.time()
    model = SAM("sam_b.pt")
    model_load_time = time.time() - start
    return model, model_load_time

model, model_load_time = load_model()
st.info(f"SAM Model loaded in {model_load_time:.2f} seconds.")

if uploaded_file:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    orig_size = image.size

    # Resize large images for speed 
    max_side = 640
    if max(orig_size) > max_side:
        scale = max_side / max(orig_size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
        st.write(f"Image resized from {orig_size} to {new_size} for faster inference.")

    st.image(image, caption="Input Image", width='stretch')
    
    # Inference
    if st.button("Run SAM Segmentation"):
        st.write("Running segmentation...")
        start = time.time()
        results = model(image)[0]  
        inference_time = time.time() - start

        masks = results.masks.data.cpu().numpy() 
        if masks.ndim == 2:
            mask = masks
        else:
            mask = np.any(masks, axis=0)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        # Overlay visualization
        overlay = np.array(image).copy()
        overlay[mask > 0, :] = [255, 0, 0]  # Mark segmentation in red
        st.image(mask_img, caption=f"Segmented mask (Inference {inference_time:.2f}s)", use_container_width=True)
        st.image(overlay, caption="Segmentation Overlay", use_container_width=True)

        with st.expander("Details"):
            st.write(f"Original image size: {orig_size}")
            st.write(f"Model load time: {model_load_time:.2f} seconds")
            st.write(f"Inference time: {inference_time:.2f} seconds")
            st.write(f"Predicted masks: {masks.shape[0] if masks.ndim == 3 else 1}")
else:
    st.info("Please upload a JPG or PNG image to begin.")
