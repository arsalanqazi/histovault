from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os


from config import TARGET_SIZE, MODEL_PATHS, MODEL_GPU_MAP, LABEL_MAPPINGS, ALLOWED_ORIGINS, MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from model_loader import get_model, get_loaded_models
from gradcam import get_gradcam_heatmap, apply_heatmap
from validation import validate_image
import time
import threading
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Secure CORS configuration
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# PyTorch handles memory growth dynamically, so no explicit config is needed.
if torch.cuda.is_available():
    logger.info(f"PyTorch detected {torch.cuda.device_count()} GPU(s)")
else:
    logger.warning("PyTorch did not detect any GPUs. Running on CPU.")

# --- Model Preloading Strategy ---
# Auto-detects available GPUs. If GPUs are present, models are distributed
# across GPU threads for parallel loading. If no GPUs, models are loaded
# sequentially on CPU. Compatible with both workstation and laptop.
loading_start_time = time.time()
models_ready = threading.Event()

def load_models_on_device(models, device_name):
    """Loads a list of models sequentially on a specific device."""
    logger.info(f"[{device_name}] Preloading Thread Started. Models queued: {models}")
    for cancer_type in models:
        try:
            logger.info(f"[{device_name}] Preloading {cancer_type}...")
            get_model(cancer_type)
            logger.info(f"[{device_name}] ✅ Successfully preloaded {cancer_type}")
        except Exception as e:
            logger.error(f"[{device_name}] ❌ Failed to preload {cancer_type}: {e}")
    
    elapsed = time.time() - loading_start_time
    logger.info(f"[{device_name}] Finished preloading in {elapsed:.2f}s")

def start_efficient_preloading():
    """Auto-detects GPUs and distributes model loading accordingly."""
    # Always load the validator model on CPU first
    try:
        logger.info("Preloading Histopathology Validator model on CPU...")
        from validation import get_validator_model
        get_validator_model()
    except Exception as e:
        logger.error(f"Failed to preload validator: {e}")

    # Auto-detect available GPUs
    available_gpus = torch.cuda.device_count() > 0
    all_model_keys = list(MODEL_GPU_MAP.keys())
    threads = []
    
    if available_gpus:
        # GPU mode: group models by their assigned GPU from config
        logger.info(f"🖥️ GPU mode: {torch.cuda.device_count()} GPU(s) detected. Distributing models across devices.")
        device_groups = {}
        for model_key, device in MODEL_GPU_MAP.items():
            device_groups.setdefault(device, []).append(model_key)
        
        for device, models in device_groups.items():
            t = threading.Thread(target=load_models_on_device, args=(models, device), daemon=True)
            threads.append(t)
            t.start()
    else:
        # CPU mode: load all models sequentially in a single background thread
        logger.info("💻 CPU mode: No GPU detected. Loading all models sequentially on CPU.")
        t = threading.Thread(target=load_models_on_device, args=(all_model_keys, "CPU:0"), daemon=True)
        threads.append(t)
        t.start()

    def mark_ready():
        for t in threads:
            t.join()
        models_ready.set()
        total_elapsed = time.time() - loading_start_time
        logger.info(f"🎉 All models fully loaded! Backend is 100% ready. Total time: {total_elapsed:.2f}s")
    
    threading.Thread(target=mark_ready, daemon=True).start()

# Start the background preloader immediately upon app start
start_efficient_preloading()


# --- Utility Functions ---

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_request():
    """Validate incoming request has required fields."""
    errors = []
    
    if 'image' not in request.files:
        errors.append("No image file provided")
    
    if 'cancer_type' not in request.form:
        errors.append("No cancer_type specified")
    
    if errors:
        return False, errors
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        errors.append("Empty filename")
    
    if not allowed_file(image_file.filename):
        errors.append(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    cancer_type = request.form.get('cancer_type')
    if cancer_type not in MODEL_PATHS:
        errors.append(f"Invalid cancer_type. Allowed: {', '.join(MODEL_PATHS.keys())}")
    
    return len(errors) == 0, errors


def preprocess_image(image_bytes, cancer_type, target_size=TARGET_SIZE):
    """
    Preprocess image bytes for PyTorch model prediction.
    """
    try:
        from PIL import Image
        import io
        pil_img = Image.open(io.BytesIO(image_bytes))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Keep a numpy BGR version for Grad-CAM display
        img_rgb = np.array(pil_img)
        original_img_resized = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), target_size)
    except Exception as e:
        logger.error(f"Pillow decoding failed: {e}")
        raise ValueError("Failed to decode image.")
    
    # PyTorch Transforms (Match training script)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess and add batch dimension
    tensor_img = transform(pil_img).unsqueeze(0)
    
    return tensor_img, original_img_resized


def predict_cancer(model, preprocessed_img, is_multi_class=False):
    """Run PyTorch model prediction on preprocessed image tensor."""
    device = next(model.parameters()).device
    preprocessed_img = preprocessed_img.to(device)
    
    with torch.no_grad():
        logits = model(preprocessed_img)
        
        if is_multi_class:
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predicted_class = [int(np.argmax(probs))]
            return predicted_class, probs
        else:
            # Apply sigmoid to get probability for binary classification
            prob = torch.sigmoid(logits).item()
            predicted_class = [1] if prob >= 0.5 else [0]
            return predicted_class, [[prob]]


def interpret_prediction(predicted_class, prediction, cancer_type):
    """
    Interpret prediction results with correct label mapping.
    
    Args:
        predicted_class: Predicted class index
        prediction: Raw prediction probabilities
        cancer_type: Type of cancer
    
    Returns:
        tuple: (predicted_label, cancerous_prob, non_cancerous_prob, top_class_prob)
    """
    if cancer_type == '8class':
        from config import EIGHT_CLASS_NAMES
        predicted_label = EIGHT_CLASS_NAMES[predicted_class[0]]
        
        cancerous_prob = float(prediction[1] + prediction[3] + prediction[5] + prediction[7])
        non_cancerous_prob = float(prediction[0] + prediction[2] + prediction[4] + prediction[6])
        top_class_prob = float(prediction[predicted_class[0]])
        return predicted_label, cancerous_prob, non_cancerous_prob, top_class_prob

    class_labels = LABEL_MAPPINGS.get(cancer_type, ['Non Cancerous', 'Cancerous'])
    predicted_label = class_labels[predicted_class[0]]
    
    # In our trained binary models, Class 1 is ALWAYS Cancerous
    cancerous_prob = float(prediction[0][0])
    non_cancerous_prob = 1.0 - cancerous_prob
    top_class_prob = max(cancerous_prob, non_cancerous_prob)
    
    return predicted_label, cancerous_prob, non_cancerous_prob, top_class_prob





# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    loaded_models = get_loaded_models()
    return jsonify({
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "total_models": len(MODEL_PATHS),
        "loaded_model_names": loaded_models
    }), 200


@app.route('/models', methods=['GET'])
def list_models():
    """List available cancer detection models."""
    loaded_models = get_loaded_models()
    return jsonify({
        "available_models": list(MODEL_PATHS.keys()),
        "loaded_models": loaded_models
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Expected form data:
        - cancer_type: str (e.g., 'oral', 'breast', 'gastrointestinal', 'colorectal', 'octa')
        - image: file (jpg, jpeg, png)
    
    Returns:
        JSON with prediction, probabilities, gradcam image, and timing
    """
    start_time = time.time()
    
    # Validate request
    is_valid, errors = validate_request()
    if not is_valid:
        return jsonify({"error": "; ".join(errors)}), 400
    
    cancer_type = request.form.get('cancer_type')
    image_file = request.files['image']
    
    # Read image bytes once
    try:
        image_bytes = image_file.read()
        
        # Check file size
        if len(image_bytes) > MAX_FILE_SIZE:
            return jsonify({
                "error": f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"
            }), 400
            
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        return jsonify({"error": "Failed to read image file"}), 400
    
    # --- Histopathology Image Validation (HistoVault v0.3) ---
    is_histo, validation_msg, confidence = validate_image(image_bytes)
    if not is_histo:
        logger.warning(f"Validation failed (Conf: {confidence:.2f}): {validation_msg}")
        return jsonify({
            "error": validation_msg,
            "validation_details": {
                "confidence": float(confidence),
                "is_histopathology": False
            }
        }), 400
    
    # Load model
    try:
        model = get_model(cancer_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    # Preprocess image
    try:
        preprocessed_img, original_img_resized = preprocess_image(
            image_bytes, 
            cancer_type
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return jsonify({"error": "Failed to preprocess image"}), 500
    
    # Run prediction
    try:
        is_multi_class = (cancer_type == '8class')
        predicted_class, prediction = predict_cancer(model, preprocessed_img, is_multi_class=is_multi_class)
        
        # Interpret results
        result, cancerous_prob, non_cancerous_prob, top_class_prob = interpret_prediction(
            predicted_class, prediction, cancer_type
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Failed to run prediction"}), 500
    
    # Generate Grad-CAM
    gradcam_image = None
    gradcam_error = None
    
    try:
        pred_idx = predicted_class[0] if predicted_class else 0
        heatmap = get_gradcam_heatmap(model, preprocessed_img, last_conv_layer_name=None, pred_index=pred_idx)
        gradcam_image = apply_heatmap(heatmap, original_img_resized)
        gradcam_image = f"data:image/jpeg;base64,{gradcam_image}"
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        gradcam_error = str(e)
    
    # Generate Web-Safe Original Image Preview (JPEG base64)
    # This is crucial for TIFFs which browsers can't render natively
    web_safe_image = None
    try:
        # Encode the BGR image to JPEG format in memory
        _, buffer = cv2.imencode('.jpg', original_img_resized)
        import base64
        web_safe_image = base64.b64encode(buffer).decode('utf-8')
        web_safe_image = f"data:image/jpeg;base64,{web_safe_image}"
    except Exception as e:
        logger.warning(f"Failed to create web-safe preview: {e}")

    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    response = {
        "prediction": result,
        "probability": {
            "cancerous": float(cancerous_prob),
            "non_cancerous": float(non_cancerous_prob),
            "top_class_confidence": float(top_class_prob)
        },
        "gradcam_image": gradcam_image,
        "original_image_preview": web_safe_image,
        "processing_time": f"{processing_time:.4f}s",
        "model_type": cancer_type
    }
    
    # Include gradcam error if it failed
    if gradcam_error:
        response["gradcam_error"] = gradcam_error
    
    return jsonify(response), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        "error": f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Set max content length
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    
    logger.info("Starting HistoVault Flask API...")
    app.run(host='0.0.0.0', port=5000, debug=False)