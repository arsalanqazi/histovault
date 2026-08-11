import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Constants (Must match training script)
IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models_pytorch', 'validator_mobilenet.h5')

_validator_model = None
_model_lock = __import__('threading').Lock()


def _build_validator_architecture():
    """Rebuild the validator MobileViT architecture from code.
    This is a fallback for when load_model fails due to Keras version mismatch.
    Architecture matches train_validator.py exactly.
    """
    from tensorflow.keras.layers import (
        Input, Conv2D, BatchNormalization, Activation, Add, Concatenate,
        LayerNormalization, Dense, GlobalAveragePooling2D, Dropout,
        MultiHeadAttention, Reshape
    )
    from tensorflow.keras.models import Model

    IMG_SIZE_VAL = (224, 224)

    def conv_block(x, filters, kernel_size=3, strides=1, activation='swish'):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x

    def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
        m = conv_block(x, expanded_channels, kernel_size=1, strides=1)
        m = Conv2D(expanded_channels, kernel_size=3, strides=strides, padding='same', groups=expanded_channels, use_bias=False)(m)
        m = BatchNormalization()(m)
        m = Activation('swish')(m)
        m = Conv2D(output_channels, kernel_size=1, strides=1, padding='same', use_bias=False)(m)
        m = BatchNormalization()(m)
        if strides == 1 and x.shape[-1] == output_channels:
            return Add()([x, m])
        return m

    def tf_multi_head_attention(query, key, value, num_heads):
        return MultiHeadAttention(num_heads=num_heads, key_dim=query.shape[-1] // num_heads)(query, value)

    def transformer_block(x, dim, num_heads=4, mlp_dim=256, dropout=0.1):
        res = x
        x = LayerNormalization(epsilon=1e-6)(x)
        x = tf_multi_head_attention(x, x, x, num_heads)
        x = Dropout(dropout)(x)
        x = Add()([res, x])
        res = x
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dense(mlp_dim, activation='swish')(x)
        x = Dropout(dropout)(x)
        x = Dense(dim)(x)
        x = Dropout(dropout)(x)
        x = Add()([res, x])
        return x

    def mobilevit_block(x, out_channels, tf_dim, num_heads=4, tf_blocks=2):
        local_features = conv_block(x, x.shape[-1], kernel_size=3)
        local_features = conv_block(local_features, tf_dim, kernel_size=1)
        _, h, w, c = local_features.shape
        sequence = Reshape((h * w, c))(local_features)
        for _ in range(tf_blocks):
            sequence = transformer_block(sequence, tf_dim, num_heads)
        global_features = Reshape((h, w, c))(sequence)
        fused = conv_block(global_features, out_channels, kernel_size=1)
        fused = Concatenate()([x, fused])
        out = conv_block(fused, out_channels, kernel_size=3)
        return out

    # Build model (matches train_validator.py)
    inputs = Input(shape=IMG_SIZE_VAL + (3,))
    x = conv_block(inputs, 16, strides=2)
    x = inverted_residual_block(x, 32, 24, strides=1)
    x = inverted_residual_block(x, 72, 48, strides=2)
    x = inverted_residual_block(x, 144, 48, strides=1)
    x = inverted_residual_block(x, 144, 64, strides=2)
    x = mobilevit_block(x, 64, tf_dim=96, tf_blocks=2)
    x = inverted_residual_block(x, 256, 80, strides=2)
    x = mobilevit_block(x, 80, tf_dim=120, tf_blocks=3)
    x = inverted_residual_block(x, 320, 96, strides=2)
    x = mobilevit_block(x, 96, tf_dim=144, tf_blocks=2)
    x = conv_block(x, 384, kernel_size=1)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model


def get_validator_model():
    """Lazy load the validator model"""
    global _validator_model
    if _validator_model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Validator model not found at {MODEL_PATH}")
            return None
        
        try:
            # Load on CPU to save GPU memory for main models
            with tf.device('/CPU:0'):
                # First, try direct loading (works if same Keras version)
                try:
                    _validator_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                except Exception as direct_err:
                    # If direct load fails (Keras version mismatch), rebuild architecture
                    # and load only weights. This matches train_validator.py exactly.
                    logger.warning(f"Direct model load failed: {direct_err}. Rebuilding architecture and loading weights...")
                    _validator_model = _build_validator_architecture()
                    _validator_model.load_weights(MODEL_PATH)
            logger.info("Histopathology Validator model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load validator model: {e}")
            return None
    return _validator_model

def validate_image(image_bytes):
    """
    Validate if the provided image bytes represent a histopathology slide.
    
    Returns:
        tuple: (is_valid: bool, message: str, confidence: float)
    """
    model = get_validator_model()
    if model is None:
        # If model loading failed, it's safer to fail-hard (reject) to prevent bypass.
        logger.error("Validator model unavailable. Rejecting image.")
        return False, "System Error: Histopathology Validator model failed to load. Please try again later.", 0.0

    try:
        # Decode and preprocess
        # Use Pillow for decoding as it's more robust for TIFFs
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if it's not already
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_rgb = np.array(pil_img)
        except Exception as e:
            logger.warning(f"Pillow decoding failed: {e}. Falling back to OpenCV.")
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return False, "Invalid image format or corrupted file", 0.0
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_batch = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

        # Predict
        # Class 0: Histopathology, Class 1: Non-Histopathology (based on training script)
        prediction = model.predict(img_batch, verbose=0)[0][0]
        
        # Binary Classification output is probability of Class 1 (Non-Histo)
        # So prob(Histo) = 1 - prediction
        prob_histo = 1.0 - prediction
        
        IS_HISTO_THRESHOLD = 0.5
        
        if prob_histo >= IS_HISTO_THRESHOLD:
            return True, "Valid histopathology image", float(prob_histo)
        else:
            return False, "This tool is meant for histopathology image related tasks. Please upload a valid histopathology image.", float(prob_histo)

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return True, f"Validation error: {str(e)}", 0.0
