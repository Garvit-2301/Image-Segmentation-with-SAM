import time
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from ultralytics import SAM  

def load_image(img_path, target_size=512):
    image = Image.open(img_path).convert("RGB")
  
    if max(image.size) > target_size:
        ratio = target_size / float(max(image.size))
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image

def visualize_mask(image, mask, output_file=None, title=""):
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(title)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    plt.show()

def run_sam_on_image(model, img_path, save_dir="outputs/"):
    os.makedirs(save_dir, exist_ok=True)
    image = load_image(img_path)

    t0 = time.time()
    results = model(image)[0]  # Ultralytics returns a list of result(s)
    t1 = time.time()
    
    inference_time = t1 - t0
   
    masks = results.masks.data.cpu().numpy()
    mask_agg = np.any(masks, axis=0) if masks.ndim == 3 else masks

    image_np = np.array(image)
    visualize_mask(
        image_np, 
        mask_agg, 
        output_file=os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0]+"_seg.png"),
        title=f"{os.path.basename(img_path)}\nInference: {inference_time:.2f}s"
    )
  
    mask_img = Image.fromarray((mask_agg*255).astype(np.uint8))
    mask_img.save(os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0]+"_mask.png"))
    return inference_time, len(masks)

if __name__ == "__main__":
   
    model_path = "sam_b.pt"  
    print("Loading model")
    tM_start = time.time()
    model = SAM(model_path)
    tM_end = time.time()
    model_load_time = tM_end - tM_start
    print(f"Model load time: {model_load_time:.2f} sec")
    
    #the test images are considered based on the recommendations given by various LLMs in order to ensure best inference and analysis
 
    test_imgs = [
        r"C:\Users\sahni\OneDrive\Desktop\image1.jpeg",         # Clear foreground/background
        r"C:\Users\sahni\OneDrive\Desktop\image2.jpg",      # Complex edges (hair)
        r"C:\Users\sahni\OneDrive\Desktop\image3.jpeg",      # Multiple similar objects
        r"C:\Users\sahni\OneDrive\Desktop\image4.jpeg",       # Failure/ambiguous
    ]
    for idx, p in enumerate(test_imgs):
        print(f"\n=== Case {idx+1}: {p} ===")
        try:
            inf_time, mask_ct = run_sam_on_image(model, p)
            print(f"Inference on {p}: {inf_time:.2f}s, {mask_ct} mask(s)")
        except Exception as e:
            print(f"Failed to segment {p} | Reason: {e}")