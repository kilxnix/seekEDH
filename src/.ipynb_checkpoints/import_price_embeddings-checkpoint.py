import os
import json
import logging
import numpy as np
from src.db_interface import DatabaseInterface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ImportFixedEmbeddings")

def import_fixed_embeddings():
    """Import the fixed 20-dimensional price embeddings to the database"""
    # Connect to database
    db = DatabaseInterface()
    
    if not db.is_connected:
        logger.error("Not connected to database")
        return False
    
    # Load the fixed embeddings
    embeddings_path = os.path.join('data', 'processed', 'price_embeddings_20d.npy')
    if not os.path.exists(embeddings_path):
        logger.error(f"Fixed embeddings file not found at {embeddings_path}")
        logger.info("Please run src/fix_embeddings.py first")
        return False
    
    # Load the card IDs
    card_ids_path = os.path.join('data', 'processed', 'price_card_ids.json')
    if not os.path.exists(card_ids_path):
        logger.error(f"Card IDs file not found at {card_ids_path}")
        return False
    
    # Load embeddings and card IDs
    logger.info(f"Loading fixed embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    logger.info(f"Loading card IDs from {card_ids_path}")
    with open(card_ids_path, 'r') as f:
        card_ids = json.load(f)
    
    logger.info(f"Found {len(card_ids)} card IDs and {embeddings.shape[0]} embeddings")
    
    # Process in batches
    batch_size = 50
    total = len(card_ids)
    success_count = 0
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        logger.info(f"Processing batch {i}-{end} of {total}")
        
        batch_embeddings = embeddings[i:end]
        batch_ids = card_ids[i:end]
        
        for j, card_id in enumerate(batch_ids):
            try:
                # Find the card in the database by scryfall_id
                response = db.client.table("mtg_cards").select("id").eq("scryfall_id", card_id).limit(1).execute()
                
                if response.data:
                    db_id = response.data[0]['id']
                    
                    # Update the price_embedding
                    embedding = batch_embeddings[j].tolist()
                    db.client.table("mtg_cards").update({"price_embedding": embedding}).eq("id", db_id).execute()
                    success_count += 1
                else:
                    logger.warning(f"Card with scryfall_id {card_id} not found in database")
            
            except Exception as e:
                logger.error(f"Error updating card {card_id}: {e}")
        
        logger.info(f"Processed {end}/{total} cards, {success_count} updated successfully")
    
    logger.info(f"Import completed. Updated {success_count}/{total} cards with price embeddings")
    return success_count > 0

if __name__ == "__main__":
    result = import_fixed_embeddings()
    print(f"Import {'successful' if result else 'failed'}")