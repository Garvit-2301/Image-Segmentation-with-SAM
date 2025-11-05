# Mini Task: Image Segmentation Tool with SAM  
**For: Re-Imagining Photoshop – AI Editor of 2030 (ML Track)**

---

## 1. Objective

Build a **functional object segmentation tool** to isolate objects for editing, using the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), and demonstrate its performance across varied image editing scenarios.

---

## 2. Contents

- [Setup & Usage](#setup--usage)
- [Model Choice Rationale](#model-choice-rationale)
- [Results & Failure Analysis](#results--failure-analysis)
- [SAM Architecture (Explained)](#sam-architecture-explained)
- [Mobile Optimization Strategies](#mobile-optimization-strategies)
- [Comparison with Other Approaches](#comparison-with-other-approaches)
- [Best Practices & Limitations](#best-practices--limitations)
- [Streamlit App & Deployment](#streamlit-app--deployment)
- [References](#references)

---

## Setup & Usage

**Requirements:**  
- Python 3.8+
- `pip install torch ultralytics pillow matplotlib streamlit`

**Option 1 — Command Line Demo:**  
1. Download your evaluation/test images and save them as:  
   - `image1.jpeg`
   - `image2.jpg`
   - `image3.jpeg`
   - `image4.jpeg`
2. Run the script (e.g. `sam_file.py`):  
   ```
   python sam_file.py
   ```
3. Segmentation overlays will open for each case, and timings print in console.

**Option 2 — Streamlit Interactive App:**  
1. Make sure you have the main Streamlit app script (e.g. `ui.py` or `webui.py` as per your project).
2. Run in the terminal:
   ```
   streamlit run ui.py
   ```
   or, if your file is named differently:
   ```
   streamlit run webui.py
   ```
3. Upload your own `.jpg`/`.jpeg`/`.png` image in the browser UI.  
   - Wait for model to load (see sidebar/info in app).
   - Click the segmentation button to view mask overlays and results.

**Troubleshooting for Streamlit:**
- If you get a `libGL.so.1` missing error (common on Linux/Cloud), add a `packages.txt` file in your repo with the line:
  ```
  libgl1
  ```
  *This ensures OpenCV works on Streamlit Cloud or other hosted platforms.*
- Confirm your `requirements.txt` includes:  
  ```
  torch
  ultralytics
  pillow
  matplotlib
  streamlit
  ```

---

## Model Choice Rationale

### Why SAM?

- **Promptable:** Segments objects or regions based on points, boxes, or prior masks—ideal for interactive editing.
- **Generalizable:** Not constrained to pre-set classes (unlike Mask R-CNN); can segment “anything.”
- **Scalable:** Fast and accurate, even for large images and diverse real-world data.
- **Strong Industry Support:** Open-sourced, widely adopted, and actively optimized for edge/mobile with variants like MobileSAM.

### When NOT to use SAM?

- Needs boundary-level perfection for fine hair, fur, glass—SAM gives crisp masks, not soft alpha mattes (see [Matting Models](#comparison-with-other-approaches)).
- If only fixed, known classes are ever segmented (not a realistic future scenario).

---

## Results & Failure Analysis

### For each case, we measure:
- **Model load time**
- **Inference time**
- **Number of masks (instances) found**
- **Visual quality:** How well does the mask fit? Are boundaries clean? Did it miss objects?

#### **Typical Observations:**
- **Clear fg/bg:** Near-perfect mask, rapid inference.
- **Complex edges:** Decent segmentation, crisp mask, but fine hair/transparency will be jagged.
- **Multiple objects:** Can separate distinct items, but closely packed or overlapping items may yield merged/split masks.
- **Failure:** Low-contrast, tiny, or occluded regions may be clumped with background or missed completely.

#### Why do failures happen?
- **Contrast boundary is weak**
- **Too small or too many similar objects**
- **Model is trained for general segments, not pixel-perfect alpha borders**

---

## SAM Architecture (Explained)

**Three main parts:**
1. **Image Encoder:** Converts image into a dense spatial feature embedding using Transformer-like blocks.
2. **Prompt Encoder:** Encodes user cues (point, box, mask) as “prompts” for guided segmentation.
3. **Mask Decoder:** Combines image and prompt embeddings, outputs object mask(s).

**Key advantage:**  
- *“Segment anything,” in a promptable/flexible way—not bound to classes, no retraining for new object types.*

---

## Mobile Optimization Strategies

For integration into a future mobile AI app:
- **Model Quantization:** Convert weights to INT8/FP16 for less memory and faster compute.
- **Model Pruning:** Remove redundant neurons/nodes and filters.
- **Mobile Variants:** Use [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), FastSAM, or LoRA adapters designed for low latency.
- **On-device Inference:** Run via ONNX/CoreML/NNAPI for mobile hardware acceleration.
- **Progressive Inference:** Provide “fast preview” masks first, refine if needed.

---

## Comparison with Other Approaches

| Model         | Pros                           | Cons                             | Best Use                       |
|---------------|-------------------------------|----------------------------------|--------------------------------|
| **SAM**       | Promptable, robust, open domain| Coarse edges, can be slower (full-size)| All-purpose, user-driven segmentation |
| Mask R-CNN    | Instance separation, solid for known classes | Limited to trained classes, rigid | Cataloged objects (COCO-like classes) |
| Matting (MODNet etc.) | Fine semi-transparent boundary, hair/softness | Needs rough mask/trimap input, not robust for full image | Refine edges post-SAM         |

**Best Practice (modern editors):**  
- **SAM for region selection.**
- **Matting model to refine mask’s edges** for pro-quality composites.

---

## Best Practices & Limitations

**Good for:**
- Fast, flexible, user-prompted segmentation of “anything.”
- Integrating with generative fill, retouch, and creative pipelines.

**Limitations:**
- CPU inference can be slow; always prefer GPU/mobile-optimized models for deployment.
- Boundaries are “hard”—add a matting step if softness is needed.
- May miss objects in low-contrast, abnormal, or highly cluttered cases.

---

## Streamlit App & Deployment

- **Running locally:**  
  Simply run:
  ```
  streamlit run ui.py
  ```
- **Deploying to Streamlit Community Cloud:**
  1. Push your code to a **GitHub repository**.
  2. Add a `requirements.txt` file with:
      ```
      torch
      ultralytics
      pillow
      matplotlib
      streamlit
      ```
  3. (If needed for OpenCV) Add a `packages.txt` file with:
      ```
      libgl1
      ```
  4. In Streamlit Cloud, click "New App", point to your repo & `ui.py`.
  5. Launch, upload images, and run segmentations from anywhere!
  6. If you make further changes or fixes, just push to GitHub and your deployed app will update.

---

## References

- [SAM (Segment Anything Model) Paper](https://arxiv.org/abs/2304.02643)
- [Ultralytics SAM Docs](https://docs.ultralytics.com/models/sam/)
- [MobileSAM (Lightweight SAM for Edge Inference)](https://github.com/ChaoningZhang/MobileSAM)
- [MODNet: Real-Time Trimap-Free Human Matting](https://arxiv.org/abs/2011.11961)
- [Unsplash Creative Commons Images](https://unsplash.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Streamlit Docs: Deploying](https://docs.streamlit.io/streamlit-community-cloud)

---
