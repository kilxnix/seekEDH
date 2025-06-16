import os
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Import config
try:
    from src.config import DATA_DIR
except ImportError:
    DATA_DIR = "data"

logger = logging.getLogger("PriceEmbeddings")

class PriceEmbeddingGenerator:
    def __init__(self):
        """Initialize the price embedding generator"""
        self.processed_dir = os.path.join(DATA_DIR, 'processed')
        self.processed_data_path = os.path.join(self.processed_dir, 'cards.csv')
        self.price_embeddings_path = os.path.join(self.processed_dir, 'price_embeddings.npy')
        self.price_card_ids_path = os.path.join(self.processed_dir, 'price_card_ids.json')
        
        # Ensure directory exists
        Path(self.processed_dir).mkdir(exist_ok=True, parents=True)
    
    def load_card_data(self):
        """Load the processed card data"""
        try:
            df = pd.read_csv(self.processed_data_path)
            return df
        except Exception as e:
            logger.error(f"Error loading card data: {e}")
            return None
    
    def generate_price_embeddings(self):
        """Generate embeddings based on card prices and related attributes"""
        logger.info("Generating price-based embeddings")
        
        # Load card data
        df = self.load_card_data()
        if df is None:
            return None
        
        # Keep track of card IDs
        card_ids = df['id'].tolist()
        
        # Convert price columns to numeric
        df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')
        df['price_usd_foil'] = pd.to_numeric(df['price_usd_foil'], errors='coerce')
        
        # Fill NaN values with median or 0
        median_price = df['price_usd'].median()
        median_foil_price = df['price_usd_foil'].median()
        df['price_usd'] = df['price_usd'].fillna(median_price)
        df['price_usd_foil'] = df['price_usd_foil'].fillna(median_foil_price)
        
        # Create price ratio (foil to normal)
        df['price_ratio'] = df['price_usd_foil'] / df['price_usd']
        df['price_ratio'] = df['price_ratio'].replace([np.inf, -np.inf], 1.0)
        df['price_ratio'] = df['price_ratio'].fillna(1.0)
        
        # Extract rarity as numeric (common=0, uncommon=1, rare=2, mythic=3)
        rarity_map = {'common': 0, 'uncommon': 1, 'rare': 2, 'mythic': 3}
        df['rarity_num'] = df['rarity'].map(rarity_map).fillna(0)
        
        # Convert cmc to numeric
        df['cmc'] = pd.to_numeric(df['cmc'], errors='coerce').fillna(0)
        
        # Create binary features for card types
        type_features = []
        common_types = ['Creature', 'Instant', 'Sorcery', 'Artifact', 'Enchantment', 'Planeswalker', 'Land']
        for card_type in common_types:
            feature_name = f'is_{card_type.lower()}'
            df[feature_name] = df['type_line'].str.contains(card_type, case=False, na=False).astype(int)
            type_features.append(feature_name)
        
        # Create binary features for colors
        color_features = []
        for color in ['W', 'U', 'B', 'R', 'G']:
            feature_name = f'color_{color}'
            df[feature_name] = df['colors'].str.contains(color, na=False).astype(int)
            color_features.append(feature_name)
        
        # Create features for color identity
        color_id_features = []
        for color in ['W', 'U', 'B', 'R', 'G']:
            feature_name = f'color_id_{color}'
            df[feature_name] = df['color_identity'].str.contains(color, na=False).astype(int)
            color_id_features.append(feature_name)
        
        # Select features for embeddings
        feature_columns = [
            'price_usd', 'price_usd_foil', 'price_ratio', 
            'rarity_num', 'cmc'
        ] + type_features + color_features + color_id_features
        
        # Create the embedding matrix
        embeddings = df[feature_columns].values
        
        # Normalize the embeddings
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        
        # Save embeddings
        np.save(self.price_embeddings_path, embeddings)
        
        # Save card IDs for reference
        with open(self.price_card_ids_path, 'w') as f:
            json.dump(card_ids, f)
        
        # Save feature names for reference
        feature_names_path = os.path.join(self.processed_dir, 'price_embedding_features.json')
        with open(feature_names_path, 'w') as f:
            json.dump(feature_columns, f)
        
        logger.info(f"Price embeddings generated with shape {embeddings.shape} and saved to {self.price_embeddings_path}")
        return embeddings
    
    def get_similar_cards_by_price(self, card_name, top_n=10):
        """Find cards with similar price characteristics"""
        # Load card data
        df = self.load_card_data()
        if df is None:
            return None
        
        # Load embeddings
        if not os.path.exists(self.price_embeddings_path):
            logger.error("Price embeddings not found. Run generate_price_embeddings first.")
            return None
        
        embeddings = np.load(self.price_embeddings_path)
        
        # Load card IDs
        with open(self.price_card_ids_path, 'r') as f:
            card_ids = json.load(f)
        
        # Find the card index
        card_idx = df[df['name'] == card_name].index
        if len(card_idx) == 0:
            logger.error(f"Card '{card_name}' not found")
            return None
        
        card_idx = card_idx[0]
        
        # Get the card embedding
        card_embedding = embeddings[card_idx].reshape(1, -1)
        
        # Calculate distances
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(card_embedding, embeddings)[0]
        
        # Get top N similar cards (excluding the card itself)
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        
        # Get the card details
        similar_cards = []
        for idx in similar_indices:
            card_id = card_ids[idx]
            card_row = df[df['id'] == card_id].iloc[0]
            similar_cards.append({
                'name': card_row['name'],
                'price_usd': card_row['price_usd'],
                'price_usd_foil': card_row['price_usd_foil'],
                'rarity': card_row['rarity'],
                'set': card_row['set_name'],
                'similarity': similarities[idx]
            })
        
        return similar_cards
    
    def get_status(self):
        """Get the status of price embeddings"""
        status = {
            "price_embeddings_exist": os.path.exists(self.price_embeddings_path),
        }
        
        if status["price_embeddings_exist"]:
            status["price_embeddings_last_modified"] = datetime.fromtimestamp(
                os.path.getmtime(self.price_embeddings_path)
            ).isoformat()
            
            # Get embeddings shape - with type conversion for JSON serialization
            try:
                embeddings = np.load(self.price_embeddings_path)
                status["price_embeddings_shape"] = [int(dim) for dim in embeddings.shape]
                status["price_embeddings_size_mb"] = float(os.path.getsize(self.price_embeddings_path) / (1024 * 1024))
            except:
                status["price_embeddings_shape"] = "Error reading file"
        
        return status