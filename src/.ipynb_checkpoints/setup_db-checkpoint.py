import os
import argparse
import logging
from src.db_config import set_supabase_credentials
from src.db_interface import DatabaseInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SetupDB")

def setup_database():
    """Set up the database connection with the Supabase key"""
    parser = argparse.ArgumentParser(description="Set up Supabase connection")
    parser.add_argument("--url", type=str, help="Supabase URL")
    parser.add_argument("--key", type=str, help="Supabase service role key")
    parser.add_argument("--init-schema", action="store_true", help="Initialize schema")
    
    args = parser.parse_args()
    
    # Get user input for missing parameters
    url = args.url or input("Supabase URL [https://nhtnamzvwmzbqsvljvga.supabase.co]: ") or "https://nhtnamzvwmzbqsvljvga.supabase.co"
    key = args.key or input("Supabase service role key: ")
    
    if not key:
        logger.error("Supabase key is required")
        return False
    
    # Save the credentials
    set_supabase_credentials(url, key)
    
    # Connect to the database
    db = DatabaseInterface()
    
    if not db.is_connected:
        logger.error("Failed to connect to Supabase")
        return False
    
    logger.info("Connected to Supabase successfully")
    
    # Initialize schema if requested
    if args.init_schema:
        logger.info("Initializing schema...")
        result = db.initialize_schema()
        
        if result:
            logger.info("Schema initialized successfully")
        else:
            logger.warning("Schema initialization might need to be done manually through the Supabase dashboard")
    
    return True

if __name__ == "__main__":
    setup_database()