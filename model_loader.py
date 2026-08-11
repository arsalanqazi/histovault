import threading
import torch
import timm
from config import MODEL_PATHS, MODEL_GPU_MAP
import logging
import os

logger = logging.getLogger(__name__)

# Dictionary to store loaded models (cache)
# Initialize with None for lazy loading
_loaded_models = {k: None for k in MODEL_PATHS.keys()}

# Create a separate lock for EACH model to allow parallel loading requests (if needed)
_model_locks = {k: threading.Lock() for k in MODEL_PATHS.keys()}

# Export model cache for monitoring (used by health endpoint)
model_cache = _loaded_models

def get_student_architecture(device, num_classes=1):
    """Recreates the PyTorch MobileViT architecture to load weights into."""
    model = timm.create_model('mobilevit_xxs', pretrained=False, num_classes=num_classes)
    model.to(device)
    return model

def get_model(cancer_type):
    """
    Retrieves a loaded PyTorch model for the specified cancer type.
    If the model is not yet loaded, it loads it from disk.
    
    Args:
        cancer_type (str): Type of cancer model to load
    
    Returns:
        torch.nn.Module: Loaded model
    
    Raises:
        ValueError: If cancer_type is invalid
        Exception: If model loading fails
    """
    if cancer_type not in MODEL_PATHS:
        available = ', '.join(MODEL_PATHS.keys())
        raise ValueError(
            f"Unknown cancer type: '{cancer_type}'. "
            f"Available types: {available}"
        )
    
    # Double-checked locking pattern for performance + safety
    if _loaded_models[cancer_type] is None:
        with _model_locks[cancer_type]:
            if _loaded_models[cancer_type] is None:
                # Decide device from config
                device_str = MODEL_GPU_MAP.get(cancer_type, 'cpu')
                device = torch.device(device_str if device_str != 'cpu' or not torch.cuda.is_available() else 'cpu')
                
                logger.info(f"Loading PyTorch model for {cancer_type} on {device}...")
                
                try:
                    model_path = MODEL_PATHS[cancer_type]
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                        
                    num_classes = 8 if cancer_type == '8class' else 1
                    model = get_student_architecture(device, num_classes=num_classes)
                    # Load state dict
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                    model.eval() # Set to evaluation mode for inference
                    
                    _loaded_models[cancer_type] = model
                    
                    # Warm up the model with a dummy prediction
                    dummy_input = torch.randn(1, 3, 224, 224).to(device)
                    with torch.no_grad():
                        _ = _loaded_models[cancer_type](dummy_input)
                    
                    logger.info(f"PyTorch model for {cancer_type} loaded successfully.")
                
                except Exception as e:
                    logger.error(f"Failed to load model {cancer_type}: {e}")
                    raise Exception(f"Failed to load model: {str(e)}")
    
    return _loaded_models[cancer_type]


def unload_model(cancer_type):
    """Unload a model from cache to free memory."""
    if cancer_type in _loaded_models and _loaded_models[cancer_type] is not None:
        with _model_locks[cancer_type]:
            _loaded_models[cancer_type] = None
            logger.info(f"Unloaded model: {cancer_type}")
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
    
    return False


def get_loaded_models():
    """Get list of currently loaded models."""
    return [k for k, v in _loaded_models.items() if v is not None]


def clear_all_models():
    """Unload all models from cache."""
    count = 0
    for cancer_type in MODEL_PATHS.keys():
        if unload_model(cancer_type):
            count += 1
    
    logger.info(f"Cleared {count} models from cache")
    return count