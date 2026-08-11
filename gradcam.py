import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
import logging

logger = logging.getLogger(__name__)

class NativeFeatureCAM:
    """
    Computes a Class Activation Map (CAM) directly from the dense features,
    bypassing the need for backward gradients. This perfectly leverages the
    high-resolution semantic patches distilled from DINOv3!
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor, target_class=0):
        with torch.no_grad():
            # Extract the raw dense spatial feature map
            # For mobilevit_xxs, this is typically shape (1, 320, 7, 7)
            spatial_features = self.model.forward_features(input_tensor)
            
            # Find the final classification linear layer weights
            fc_weight = None
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    fc_weight = module.weight
            
            if fc_weight is None:
                raise ValueError("Could not find a Linear classification head in the model.")
            
            features = spatial_features[0].cpu().numpy()  # Shape: (C, H, W)
            
            # Handle multi-class vs binary FC weights shape
            if len(fc_weight.shape) > 1 and fc_weight.shape[0] > 1:
                class_idx = min(target_class, fc_weight.shape[0] - 1)
                weights = fc_weight[class_idx].cpu().numpy()
            else:
                weights = fc_weight[0].cpu().numpy()
            
            # Compute Native CAM: Weighted sum of spatial features using the classifier's weights
            heatmap = np.zeros(features.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                heatmap += w * features[i]
                
            # Apply ReLU to keep only positive feature contributions
            heatmap = np.maximum(heatmap, 0)
            
            # Normalize to [0, 1]
            max_val = np.max(heatmap)
            if max_val > 0:
                heatmap /= max_val
                
            return heatmap


def get_gradcam_heatmap(model, img_tensor, last_conv_layer_name=None, pred_index=None):
    """
    Generates a Native Feature CAM heatmap.
    """
    target_class = pred_index if pred_index is not None else 0
    cam = NativeFeatureCAM(model)
    heatmap = cam.generate_heatmap(img_tensor, target_class=target_class)
    return heatmap


def apply_heatmap(heatmap, original_img, alpha=0.4):
    """
    Applies the heatmap overlay to the original image (BGR) and returns base64 JPEG string.
    """
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap (Returns BGR format)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Blend in BGR color space (original_img is also BGR)
    superimposed_img = jet.astype(np.float32) * alpha + original_img.astype(np.float32) * (1.0 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    
    # cv2.imencode expects BGR format natively. Do NOT convert to RGB before imencode!
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64