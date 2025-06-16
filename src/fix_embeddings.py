import os
import json
import logging
import numpy as np
from sklearn.decomposition import PCA
from src.price_embeddings import PriceEmbeddingGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FixEmbeddings")

def fix_embeddings():
    """Resize the price embeddings to 20 dimensions using PCA"""
    # Load the price embeddings
    price_embedder = PriceEmbeddingGenerator()
    
    if not os.path.exists(price_embedder.price_embeddings_path):
        logger.error(f"Price embeddings file not found at {price_embedder.price_embeddings_path}")
        return False
    
    logger.info(f"Loading price embeddings from {price_embedder.price_embeddings_path}")
    embeddings = np.load(price_embedder.price_embeddings_path)
    logger.info(f"Original embeddings shape: {embeddings.shape}")
    
    # Load card IDs
    with open(price_embedder.price_card_ids_path, 'r') as f:
        card_ids = json.load(f)
    logger.info(f"Card IDs count: {len(card_ids)}")
    
    # Reduce dimensions to 20 using PCA
    logger.info("Reducing embedding dimensions from 22 to 20 using PCA")
    pca = PCA(n_components=20)
    embeddings_20d = pca.fit_transform(embeddings)
    logger.info(f"Resized embeddings shape: {embeddings_20d.shape}")
    
    # Save the resized embeddings
    embeddings_20d_path = os.path.join(price_embedder.processed_dir, 'price_embeddings_20d.npy')
    np.save(embeddings_20d_path, embeddings_20d)
    logger.info(f"Saved 20-dimensional embeddings to {embeddings_20d_path}")
    
    # Return both the file path and the embeddings
    return {
        "path": embeddings_20d_path,
        "embeddings": embeddings_20d,
        "card_ids": card_ids
    }

if __name__ == "__main__":
    result = fix_embeddings()
    print(f"Embeddings fixed and saved to {result['path']}")