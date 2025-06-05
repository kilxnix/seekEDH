import os
import json
import logging
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get('DATA_DIR', BASE_DIR / 'data')
LOGS_DIR = os.environ.get('LOGS_DIR', BASE_DIR / 'logs')

# Ensure directories exist
Path(DATA_DIR).mkdir(exist_ok=True)
Path(LOGS_DIR).mkdir(exist_ok=True)
Path(DATA_DIR, 'raw').mkdir(exist_ok=True)
Path(DATA_DIR, 'processed').mkdir(exist_ok=True)

# API Configuration
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', 5000))

# Database Configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

# Card data source
SCRYFALL_API_URL = 'https://api.scryfall.com'

# Embedding model
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# Logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FILE = os.path.join(LOGS_DIR, 'mtg_ai.log')

# Initialize logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Store configuration
CONFIG = {
    'base_dir': str(BASE_DIR),
    'data_dir': str(DATA_DIR),
    'logs_dir': str(LOGS_DIR),
    'api_host': API_HOST,
    'api_port': API_PORT,
    'supabase_url': SUPABASE_URL,
    'supabase_key': SUPABASE_KEY,
    'scryfall_api_url': SCRYFALL_API_URL,
    'embedding_model': EMBEDDING_MODEL,
    'log_level': LOG_LEVEL,
    'log_file': LOG_FILE
}

def set_supabase_credentials(url, key):
    """Set Supabase credentials programmatically"""
    global SUPABASE_URL, SUPABASE_KEY, CONFIG
    SUPABASE_URL = url
    SUPABASE_KEY = key
    CONFIG['supabase_url'] = url
    CONFIG['supabase_key'] = key
    
    # Save credentials to a json file for persistence
    creds_file = os.path.join(BASE_DIR, 'credentials.json')
    try:
        with open(creds_file, 'w') as f:
            json.dump({'supabase_url': url, 'supabase_key': key}, f)
    except Exception as e:
        logging.error(f"Failed to save credentials: {e}")

# Try to load credentials from file if they exist
creds_file = os.path.join(BASE_DIR, 'credentials.json')
if os.path.exists(creds_file):
    try:
        with open(creds_file, 'r') as f:
            creds = json.load(f)
            SUPABASE_URL = creds.get('supabase_url', SUPABASE_URL)
            SUPABASE_KEY = creds.get('supabase_key', SUPABASE_KEY)
            CONFIG['supabase_url'] = SUPABASE_URL
            CONFIG['supabase_key'] = SUPABASE_KEY
    except Exception as e:
        logging.error(f"Failed to load credentials: {e}")

def get_config():
    """Get the current configuration"""
    return CONFIG
