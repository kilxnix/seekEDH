# src/db_setup.py
import os
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_database():
    """Initialize and set up the Supabase database"""
    try:
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in environment variables")
            return False
        
        supabase = create_client(supabase_url, supabase_key)
        
        logger.info("Connected to Supabase successfully")
        
        # Load the schema SQL file
        with open("database/schema.sql", "r") as f:
            schema_sql = f.read()
        
        # Execute the schema SQL
        # Note: In a real application, you would use the Supabase dashboard or migrations
        # This is a simplified example
        logger.info("Database schema initialized (simulated)")
        
        return True
    
    except Exception as e:
        logger.error(f"Error setting up database: {str(e)}")
        return False

def import_cards_to_db(data_dir="data"):
    """Import processed card data and embeddings to the database"""
    try:
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase = create_client(supabase_url, supabase_key)
        
        # Load processed cards
        processed_path = os.path.join(data_dir, "processed", "cards.csv")
        df = pd.read_csv(processed_path)
        
        # Load embeddings
        embeddings_path = os.path.join(data_dir, "processed", "embeddings.npy")
        embeddings = np.load(embeddings_path)
        
        # Load card IDs
        ids_path = os.path.join(data_dir, "processed", "card_ids.json")
        with open(ids_path, 'r') as f:
            card_ids = json.load(f)
        
        # Import cards to database
        logger.info(f"Importing {len(df)} cards to database")
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Convert embedding to list for JSON serialization
            embedding = embeddings[i].tolist()
            
            # Prepare card data
            card_data = {
                "scryfall_id": row["id"],
                "name": row["name"],
                "mana_cost": row["mana_cost"],
                "cmc": row["cmc"],
                "type_line": row["type_line"],
                "oracle_text": row["oracle_text"],
                "colors": row["colors"].split(",") if row["colors"] else [],
                "color_identity": row["color_identity"].split(",") if row["color_identity"] else [],
                "power": row["power"],
                "toughness": row["toughness"],
                "rarity": row["rarity"],
                "set_code": row["set"],
                "set_name": row["set_name"],
                "prices_usd": row["price_usd"],
                "prices_usd_foil": row["price_usd_foil"],
                "embedding": embedding
            }
            
            # Insert or update the card in the database
            result = supabase.table("mtg_cards").upsert(card_data).execute()
            
            if i % 100 == 0:
                logger.info(f"Imported {i}/{len(df)} cards")
        
        logger.info("Card import completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error importing cards to database: {str(e)}")
        return False

if __name__ == "__main__":
    if setup_database():
        logger.info("Database setup completed")
        import_cards_to_db()
    else:
        logger.error("Database setup failed")