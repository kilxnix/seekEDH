# src/config.py - Enhanced configuration with image storage support

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")

# API Configuration
SCRYFALL_API_URL = "https://api.scryfall.com"
API_HOST = "0.0.0.0"
API_PORT = 5000

# Embedding Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Database Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Image Storage Configuration
IMAGE_STORAGE_CONFIG = {
    # Image sizes to download from Scryfall
    "AVAILABLE_SIZES": ["small", "normal", "large", "png", "art_crop", "border_crop"],
    
    # Default size preferences (in order of preference)
    "SIZE_PREFERENCES": ["normal", "large", "small"],
    
    # Default size for new downloads
    "DEFAULT_SIZE": "normal",
    
    # Maximum concurrent downloads
    "MAX_DOWNLOAD_WORKERS": 5,
    
    # Download timeout in seconds
    "DOWNLOAD_TIMEOUT": 30,
    
    # Maximum concurrent uploads to storage
    "MAX_UPLOAD_WORKERS": 3,
    
    # Storage bucket name for Supabase
    "STORAGE_BUCKET_NAME": "mtg-card-images",
    
    # Local image directories
    "LOCAL_IMAGES_DIR": os.path.join(DATA_DIR, "images"),
    
    # Image file formats
    "SUPPORTED_FORMATS": [".jpg", ".jpeg", ".png", ".webp"],
    
    # Maximum file size for images (in bytes) - 10MB
    "MAX_FILE_SIZE": 10 * 1024 * 1024,
    
    # Cache control for stored images (in seconds) - 1 hour
    "CACHE_CONTROL_SECONDS": 3600,
    
    # Retry configuration
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 1,  # Base delay in seconds
    
    # Batch processing
    "DEFAULT_BATCH_SIZE": 50,
    "LARGE_BATCH_SIZE": 100,
    
    # Cleanup configuration
    "CLEANUP_KEEP_SIZES": ["normal"],  # Default sizes to keep during cleanup
    
    # Image quality settings
    "QUALITY_SETTINGS": {
        "preview": {"size": "small", "max_dimension": 200},
        "display": {"size": "normal", "max_dimension": 488},
        "detail": {"size": "large", "max_dimension": 672},
        "print": {"size": "png", "max_dimension": None}
    }
}

# Data Pipeline Configuration
PIPELINE_CONFIG = {
    # Whether to download images by default
    "DOWNLOAD_IMAGES_DEFAULT": True,
    
    # Whether to upload to storage by default
    "UPLOAD_TO_STORAGE_DEFAULT": True,
    
    # Whether to skip embeddings by default
    "SKIP_EMBEDDINGS_DEFAULT": False,
    
    # Force update threshold (hours)
    "FORCE_UPDATE_THRESHOLD_HOURS": 24,
    
    # Embedding batch size
    "EMBEDDING_BATCH_SIZE": 32,
    
    # Database import batch size
    "DB_IMPORT_BATCH_SIZE": 40,
    
    # Progress logging interval
    "PROGRESS_LOG_INTERVAL": 100
}

# Image Serving Configuration
IMAGE_SERVING_CONFIG = {
    # Enable local image serving
    "ENABLE_LOCAL_SERVING": True,
    
    # Enable storage image serving
    "ENABLE_STORAGE_SERVING": True,
    
    # Enable fallback to original URLs
    "ENABLE_FALLBACK_SERVING": True,
    
    # Default image serving preference order
    "SERVING_PREFERENCE": ["storage", "local", "original"],
    
    # Image serving cache headers
    "CACHE_HEADERS": {
        "Cache-Control": "public, max-age=3600",  # 1 hour
        "Expires": "3600"
    },
    
    # CORS settings for image serving
    "CORS_ORIGINS": ["*"],  # Allow all origins for images
    
    # Maximum image requests per minute per IP
    "RATE_LIMIT_PER_MINUTE": 120
}

# Monitoring and Logging Configuration
MONITORING_CONFIG = {
    # Enable detailed logging
    "DETAILED_LOGGING": True,
    
    # Log image operations
    "LOG_IMAGE_OPERATIONS": True,
    
    # Log level for image operations
    "IMAGE_LOG_LEVEL": "INFO",
    
    # Enable performance monitoring
    "ENABLE_PERFORMANCE_MONITORING": False,
    
    # Statistics collection interval (seconds)
    "STATS_COLLECTION_INTERVAL": 300,  # 5 minutes
    
    # Keep statistics for (days)
    "STATS_RETENTION_DAYS": 30
}

# Security Configuration
SECURITY_CONFIG = {
    # Enable image content validation
    "VALIDATE_IMAGE_CONTENT": True,
    
    # Maximum allowed image dimensions
    "MAX_IMAGE_DIMENSIONS": (2000, 2000),
    
    # Allowed MIME types
    "ALLOWED_MIME_TYPES": [
        "image/jpeg",
        "image/png", 
        "image/webp",
        "image/gif"
    ],
    
    # Scan for malicious content
    "SCAN_FOR_MALICIOUS_CONTENT": False,
    
    # Enable rate limiting
    "ENABLE_RATE_LIMITING": True,
    
    # API key requirements
    "REQUIRE_API_KEY_FOR_UPLOADS": False,
    "REQUIRE_API_KEY_FOR_MANAGEMENT": False
}

# Development/Testing Configuration
DEV_CONFIG = {
    # Enable development mode
    "DEVELOPMENT_MODE": os.getenv("DEVELOPMENT_MODE", "False").lower() == "true",
    
    # Test data directory
    "TEST_DATA_DIR": os.path.join(BASE_DIR, "tests", "data"),
    
    # Sample cards for testing
    "TEST_CARDS": [
        "Lightning Bolt",
        "Sol Ring", 
        "Black Lotus",
        "Kozilek, the Great Distortion",
        "Jace, the Mind Sculptor",
        "Tarmogoyf",
        "Snapcaster Mage",
        "Force of Will",
        "Dark Confidant",
        "Liliana of the Veil"
    ],
    
    # Mock external services in tests
    "MOCK_EXTERNAL_SERVICES": True,
    
    # Test image sizes
    "TEST_IMAGE_SIZES": ["small", "normal"],
    
    # Reduced timeouts for testing
    "TEST_TIMEOUT": 5,
    
    # Reduced batch sizes for testing
    "TEST_BATCH_SIZE": 5
}

# Feature Flags
FEATURE_FLAGS = {
    # Enable image management features
    "ENABLE_IMAGE_MANAGEMENT": True,
    
    # Enable dual-faced card support
    "ENABLE_DUAL_FACED_CARDS": True,
    
    # Enable advanced search features
    "ENABLE_ADVANCED_SEARCH": True,
    
    # Enable price embeddings
    "ENABLE_PRICE_EMBEDDINGS": True,
    
    # Enable RAG system
    "ENABLE_RAG_SYSTEM": True,
    
    # Enable storage integration
    "ENABLE_STORAGE_INTEGRATION": True,
    
    # Enable image optimization
    "ENABLE_IMAGE_OPTIMIZATION": False,  # Requires additional dependencies
    
    # Enable image CDN
    "ENABLE_IMAGE_CDN": False,  # For future implementation
    
    # Enable analytics
    "ENABLE_ANALYTICS": False
}

# Get environment-specific overrides
def get_config():
    """Get configuration with environment-specific overrides"""
    config = {
        "BASE_DIR": BASE_DIR,
        "DATA_DIR": DATA_DIR,
        "SCRYFALL_API_URL": SCRYFALL_API_URL,
        "API_HOST": API_HOST,
        "API_PORT": API_PORT,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_KEY": SUPABASE_KEY,
        "IMAGE_STORAGE": IMAGE_STORAGE_CONFIG,
        "PIPELINE": PIPELINE_CONFIG,
        "IMAGE_SERVING": IMAGE_SERVING_CONFIG,
        "MONITORING": MONITORING_CONFIG,
        "SECURITY": SECURITY_CONFIG,
        "DEV": DEV_CONFIG,
        "FEATURES": FEATURE_FLAGS
    }
    
    # Environment-specific overrides
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        config["FEATURES"]["ENABLE_ANALYTICS"] = True
        config["MONITORING"]["ENABLE_PERFORMANCE_MONITORING"] = True
        config["SECURITY"]["SCAN_FOR_MALICIOUS_CONTENT"] = True
        config["SECURITY"]["REQUIRE_API_KEY_FOR_MANAGEMENT"] = True
        config["IMAGE_STORAGE"]["MAX_DOWNLOAD_WORKERS"] = 10
        config["IMAGE_STORAGE"]["MAX_UPLOAD_WORKERS"] = 5
    
    elif env == "staging":
        config["MONITORING"]["ENABLE_PERFORMANCE_MONITORING"] = True
        config["IMAGE_STORAGE"]["MAX_DOWNLOAD_WORKERS"] = 8
        config["IMAGE_STORAGE"]["MAX_UPLOAD_WORKERS"] = 4
    
    elif env == "development":
        config["DEV"]["DEVELOPMENT_MODE"] = True
        config["MONITORING"]["DETAILED_LOGGING"] = True
        config["IMAGE_STORAGE"]["MAX_DOWNLOAD_WORKERS"] = 3
        config["IMAGE_STORAGE"]["MAX_UPLOAD_WORKERS"] = 2
    
    return config

# Validation functions
def validate_image_config():
    """Validate image configuration settings"""
    errors = []
    
    # Check required directories
    if not os.path.exists(DATA_DIR):
        try:
            Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create data directory: {e}")
    
    # Check image sizes
    available_sizes = IMAGE_STORAGE_CONFIG["AVAILABLE_SIZES"]
    size_preferences = IMAGE_STORAGE_CONFIG["SIZE_PREFERENCES"]
    
    for size in size_preferences:
        if size not in available_sizes:
            errors.append(f"Preferred size '{size}' not in available sizes")
    
    # Check batch sizes
    if IMAGE_STORAGE_CONFIG["DEFAULT_BATCH_SIZE"] <= 0:
        errors.append("DEFAULT_BATCH_SIZE must be positive")
    
    # Check timeout values
    if IMAGE_STORAGE_CONFIG["DOWNLOAD_TIMEOUT"] <= 0:
        errors.append("DOWNLOAD_TIMEOUT must be positive")
    
    # Check worker counts
    if IMAGE_STORAGE_CONFIG["MAX_DOWNLOAD_WORKERS"] <= 0:
        errors.append("MAX_DOWNLOAD_WORKERS must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

def get_image_url_template(size: str = "normal") -> str:
    """Get URL template for serving images"""
    if DEV_CONFIG["DEVELOPMENT_MODE"]:
        return f"http://localhost:{API_PORT}/api/images/serve/{{card_name}}?size={size}"
    else:
        return f"/api/images/serve/{{card_name}}?size={size}"

def get_storage_path_template(size: str = "normal") -> str:
    """Get storage path template for organizing images"""
    return f"images/{size}/{{card_id}}.{{ext}}"

# Initialize configuration on import
try:
    validate_image_config()
    print("Image configuration validated successfully")
except Exception as e:
    print(f"Warning: Image configuration validation failed: {e}")

# Export commonly used values
__all__ = [
    "BASE_DIR",
    "DATA_DIR", 
    "SCRYFALL_API_URL",
    "API_HOST",
    "API_PORT",
    "EMBEDDING_MODEL",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "IMAGE_STORAGE_CONFIG",
    "PIPELINE_CONFIG",
    "IMAGE_SERVING_CONFIG",
    "MONITORING_CONFIG",
    "SECURITY_CONFIG",
    "DEV_CONFIG",
    "FEATURE_FLAGS",
    "get_config",
    "validate_image_config",
    "get_image_url_template",
    "get_storage_path_template"
]