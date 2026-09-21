import os
import sys
import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def apply_retinal_mask(heatmap_image, original_rgb):
    """
    Detects the circular fundus lens boundary and masks out all 
    background/non-eye regions outside the retina.
    """
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    
    # Threshold to identify the retinal region
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (the retinal fundus circle)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a single-channel mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Smooth mask edges to avoid harsh boundaries
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        
        # Convert single-channel mask to 3-channel RGB mask (normalized [0, 1])
        mask_3d = cv2.merge([mask, mask, mask]) / 255.0
        
        # Apply mask: keep heatmap inside eye region, set background outside to original dark image
        masked_heatmap = (heatmap_image * mask_3d) + (original_rgb * (1.0 - mask_3d))
        return np.uint8(masked_heatmap)
        
    return heatmap_image

def generate_gradcam_heatmap(model, input_tensor, original_image_rgb, targets=None):
    """
    Generates a Grad-CAM heatmap overlay strictly constrained to the eye region.
    """
    model.eval()
    
    target_layers = model.get_target_layer()
    if not isinstance(target_layers, list):
        target_layers = [target_layers]

    # Normalize image to float32 [0.0, 1.0] for pytorch_grad_cam
    rgb_img_float = np.float32(original_image_rgb)
    if rgb_img_float.max() > 1.0:
        rgb_img_float /= 255.0

    cam = GradCAM(model=model, target_layers=target_layers)

    with torch.set_grad_enabled(True):
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Raw Grad-CAM visualization
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    
    # Restrict heatmap strictly to the eye region by removing background noise
    final_heatmap = apply_retinal_mask(visualization, original_image_rgb)
    return final_heatmap