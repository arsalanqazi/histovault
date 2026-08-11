# Configuration for HistoVault App v0.5 Preview (PyTorch Backend)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to the trained PyTorch models
MODEL_PATHS = {
    'colorectal': os.path.join(BASE_DIR, 'models_pytorch', 'mobilevit_colorectal.pt'),
    'breast': os.path.join(BASE_DIR, 'models_pytorch', 'mobilevit_breast.pt'),
    'gastrointestinal': os.path.join(BASE_DIR, 'models_pytorch', 'mobilevit_gastrointestinal.pt'),
    'oral': os.path.join(BASE_DIR, 'models_pytorch', 'mobilevit_oral.pt'),
    'octa': os.path.join(BASE_DIR, 'models_pytorch', 'mobilevit_octa.pt'),
    '8class': os.path.join(BASE_DIR, 'models_pytorch', 'mobilevit_8class.pt'),
}

# Device Distribution Strategy (PyTorch device strings)
MODEL_GPU_MAP = {
    'colorectal': 'cpu',
    'breast': 'cpu',
    'gastrointestinal': 'cpu',
    'oral': 'cpu',
    'octa': 'cpu',
    '8class': 'cpu',
}

# Image processing capability
TARGET_SIZE = (224, 224)

# Label mappings for each cancer type
# Format: [class_0_label, class_1_label]
LABEL_MAPPINGS = {
    'oral': ['Non Cancerous', 'Cancerous'],
    'breast': ['Non Cancerous', 'Cancerous'],
    'colorectal': ['Non Cancerous', 'Cancerous'],
    'gastrointestinal': ['Non Cancerous', 'Cancerous'],
    'octa': ['Non Cancerous', 'Cancerous'],
}

# 8-class label names (used if the 8-class model is exposed)
EIGHT_CLASS_NAMES = [
    'Breast Non-Cancerous', 'Breast Cancerous',
    'Colorectal Non-Cancerous', 'Colorectal Cancerous',
    'Gastrointestinal Non-Cancerous', 'Gastrointestinal Cancerous',
    'Oral Non-Cancerous', 'Oral Cancerous',
]

# CORS Configuration
ALLOWED_ORIGINS = "*"

# File Upload Limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# API Rate Limiting (if you implement it later)
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_PER_HOUR = 1000