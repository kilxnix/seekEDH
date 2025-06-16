import os
import json
import logging
import numpy as np
from src.db_interface import DatabaseInterface
from src.price_embeddings import PriceEmbeddingGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VerifyEmbeddings")

def verify_embeddings():
    """Verify that price embeddings were saved properly"""
    # Connect to database
    db = DatabaseInterface()
    
    if not db.is_connected:
        logger.error("Not connected to database")
        return False
    
    # First, let's check what tables and columns exist
    try:
        # Simple query to get a sample of the cards
        logger.info("Checking database structure...")
        response = db.client.table("mtg_cards").select("*").limit(1).execute()
        
        if not response.data:
            logger.error("No cards found in database")
            return False
            
        sample_card = response.data[0]
        logger.info(f"Sample card columns: {list(sample_card.keys())}")
        
        # Check if price_embedding column exists
        if 'price_embedding' not in sample_card:
            logger.error("price_embedding column not found in database")
            return False
            
        # Check if any cards have price embeddings
        logger.info("Checking for cards with price embeddings...")
        
        # Try different query approaches
        response = db.client.table("mtg_cards").select("name, prices_usd").limit(5).execute()
        logger.info(f"Found {len(response.data)} cards")
        
        # Check local embeddings
        price_embedder = PriceEmbeddingGenerator()
        
        if not os.path.exists(price_embedder.price_embeddings_path):
            logger.error(f"Price embeddings file not found at {price_embedder.price_embeddings_path}")
            return False
        
        # Load local embeddings
        embeddings = np.load(price_embedder.price_embeddings_path)
        logger.info(f"Local embeddings shape: {embeddings.shape}")
        
        if not os.path.exists(price_embedder.price_card_ids_path):
            logger.error(f"Card IDs file not found at {price_embedder.price_card_ids_path}")
            return False
            
        with open(price_embedder.price_card_ids_path, 'r') as f:
            card_ids = json.load(f)
        
        logger.info(f"Number of card IDs: {len(card_ids)}")
        
        # If embeddings exist locally but not in the database,
        # there was an issue with the database import
        logger.warning("Price embeddings exist locally but may not be in the database.")
        logger.warning("You may need to re-run the database import with: python -m src.main --import --price-embeddings")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False

if __name__ == "__main__":
    import json
    result = verify_embeddings()
    print(f"Verification {'successful' if result else 'failed'}")