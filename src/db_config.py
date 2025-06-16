import os
import json
import logging
from supabase import create_client

logger = logging.getLogger("DBConfig")

# Supabase configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://nhtnamzvwmzbqsvljvga.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')  # Will be set programmatically

# Save credentials to this file
CREDS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials.json')

def load_credentials():
    """Load credentials from file"""
    global SUPABASE_URL, SUPABASE_KEY
    
    if os.path.exists(CREDS_FILE):
        try:
            with open(CREDS_FILE, 'r') as f:
                creds = json.load(f)
                
                SUPABASE_URL = creds.get('supabase_url', SUPABASE_URL)
                SUPABASE_KEY = creds.get('supabase_key', SUPABASE_KEY)
                
                logger.info("Credentials loaded from file")
                return True
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
    
    return False

def save_credentials(url=None, key=None):
    """Save credentials to file"""
    global SUPABASE_URL, SUPABASE_KEY
    
    if url:
        SUPABASE_URL = url
    
    if key:
        SUPABASE_KEY = key
    
    try:
        creds = {
            'supabase_url': SUPABASE_URL,
            'supabase_key': SUPABASE_KEY
        }
        
        with open(CREDS_FILE, 'w') as f:
            json.dump(creds, f)
        
        logger.info("Credentials saved to file")
        return True
    except Exception as e:
        logger.error(f"Failed to save credentials: {e}")
        return False

def set_supabase_credentials(url=None, key=None):
    """Set Supabase credentials"""
    return save_credentials(url, key)

def get_supabase_client():
    """Get a Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Supabase URL or key not set")
        return None
    
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return None

# Load credentials on module import
load_credentials()