import os
import json
import logging
import numpy as np
import pandas as pd

# Import database configuration
from src.db_config import get_supabase_client, set_supabase_credentials

logger = logging.getLogger("DBInterface")

class DatabaseInterface:
    def __init__(self):
        """Initialize the database interface"""
        # Try to connect to Supabase
        self.client = get_supabase_client()
        self.is_connected = self.client is not None
        
        if self.is_connected:
            logger.info("Connected to Supabase")
        else:
            logger.warning("Not connected to Supabase")
    
    def set_credentials(self, url=None, key=None):
        """Set Supabase credentials"""
        set_supabase_credentials(url, key)
        
        # Try to reconnect
        self.client = get_supabase_client()
        self.is_connected = self.client is not None
        
        if self.is_connected:
            logger.info("Connected to Supabase")
            return True
        else:
            logger.error("Failed to connect to Supabase")
            return False
    
    def initialize_schema(self):
        """Initialize the database schema (create tables)"""
        if not self.is_connected:
            logger.error("Cannot initialize schema: Not connected to database")
            return False
        
        try:
            # Enable the vector extension
            # Note: This may require elevated privileges and might need to be done
            # through the Supabase dashboard
            try:
                self.client.postgrest.rpc(
                    'create_extension_if_not_exists',
                    {'extension_name': 'vector'}
                ).execute()
                logger.info("Vector extension enabled")
            except Exception as e:
                logger.warning(f"Could not enable vector extension: {e}")
                logger.warning("You may need to enable it through the Supabase dashboard")
            
            # Enable the pg_trgm extension
            try:
                self.client.postgrest.rpc(
                    'create_extension_if_not_exists',
                    {'extension_name': 'pg_trgm'}
                ).execute()
                logger.info("pg_trgm extension enabled")
            except Exception as e:
                logger.warning(f"Could not enable pg_trgm extension: {e}")
            
            # Create the tables using SQL via the REST API
            # Note: Supabase doesn't provide a direct way to execute arbitrary SQL
            # through the client library. The schema would need to be created through
            # the Supabase dashboard or by connecting directly to the database.
            
            logger.info("Schema should be initialized through the Supabase dashboard.")
            logger.info("See the SQL schema in the README.md file.")
            
            # Check if the mtg_cards table exists
            try:
                response = self.client.table("mtg_cards").select("id", count="exact").limit(1).execute()
                logger.info("mtg_cards table exists")
                return True
            except Exception as e:
                logger.error(f"mtg_cards table does not exist: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            return False
    
    def import_cards(self, data_path=None, embeddings_path=None, card_ids_path=None, include_price_embeddings=False):
        """
        Import card data and embeddings to the database
        
        Args:
            data_path: Path to processed card data CSV
            embeddings_path: Path to embeddings NPY file
            card_ids_path: Path to card IDs JSON file
            include_price_embeddings: Whether to include price embeddings
        """
        if not self.is_connected:
            logger.error("Cannot import cards: Not connected to database")
            return False
        
        # Import here to avoid circular import
        from src.data_pipeline import MTGDataPipeline
        
        pipeline = MTGDataPipeline()
        
        # Use default paths from data pipeline if not provided
        data_path = data_path or pipeline.processed_data_path
        embeddings_path = embeddings_path or pipeline.embeddings_path
        card_ids_path = card_ids_path or pipeline.card_ids_path
        
        # Price embeddings paths
        price_embeddings_path = None
        price_card_ids_path = None
        
        if include_price_embeddings:
            try:
                from src.price_embeddings import PriceEmbeddingGenerator
                price_embedder = PriceEmbeddingGenerator()
                price_embeddings_path = price_embedder.price_embeddings_path
                price_card_ids_path = price_embedder.price_card_ids_path
            except ImportError:
                logger.warning("Price embeddings module not available")
        
        try:
            # Load card data
            logger.info(f"Loading card data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Check if text embeddings exist
            text_embeddings = None
            text_card_ids = None
            if os.path.exists(embeddings_path) and os.path.exists(card_ids_path):
                logger.info(f"Loading text embeddings from {embeddings_path}")
                text_embeddings = np.load(embeddings_path)
                
                logger.info(f"Loading text embedding card IDs from {card_ids_path}")
                with open(card_ids_path, 'r') as f:
                    text_card_ids = json.load(f)
            else:
                logger.warning(f"Text embeddings files not found")
                logger.warning("Cards will be imported without text embeddings")
            
            # Check if price embeddings exist
            price_embeddings = None
            price_card_ids = None
            if include_price_embeddings and price_embeddings_path and price_card_ids_path:
                if os.path.exists(price_embeddings_path) and os.path.exists(price_card_ids_path):
                    logger.info(f"Loading price embeddings from {price_embeddings_path}")
                    price_embeddings = np.load(price_embeddings_path)
                    
                    logger.info(f"Loading price embedding card IDs from {price_card_ids_path}")
                    with open(price_card_ids_path, 'r') as f:
                        price_card_ids = json.load(f)
                else:
                    logger.warning(f"Price embeddings files not found")
                    logger.warning("Cards will be imported without price embeddings")
            
            # Import cards in batches
            logger.info(f"Importing {len(df)} cards to database")
            batch_size = 50  # Smaller batch size for reliability
            
            for i in range(0, len(df), batch_size):
                try:
                    batch_df = df.iloc[i:i+batch_size]
                    batch_cards = []
                    
                    for j, (_, row) in enumerate(batch_df.iterrows()):
                        idx = i + j
                        card_id = row["id"]
                        
                        # Get text embedding if available
                        text_embedding = None
                        if text_embeddings is not None and text_card_ids is not None:
                            try:
                                text_idx = text_card_ids.index(card_id)
                                text_embedding = text_embeddings[text_idx].tolist()
                            except (ValueError, IndexError):
                                pass
                        
                        # Get price embedding if available
                        price_embedding = None
                        if price_embeddings is not None and price_card_ids is not None:
                            try:
                                price_idx = price_card_ids.index(card_id)
                                price_embedding = price_embeddings[price_idx].tolist()
                            except (ValueError, IndexError):
                                pass
                        
                        # Parse color arrays
                        colors = row["colors"].split(",") if pd.notna(row["colors"]) and row["colors"] else []
                        color_identity = row["color_identity"].split(",") if pd.notna(row["color_identity"]) and row["color_identity"] else []
                        
                        # Parse JSON fields
                        legalities = json.loads(row["legalities"]) if pd.notna(row["legalities"]) else {}
                        image_uris = json.loads(row["image_uris"]) if pd.notna(row["image_uris"]) else {}
                        
                        # Prepare card data
                        card_data = {
                            "scryfall_id": card_id,
                            "name": row["name"],
                            "mana_cost": row["mana_cost"] if pd.notna(row["mana_cost"]) else None,
                            "cmc": float(row["cmc"]) if pd.notna(row["cmc"]) else None,
                            "type_line": row["type_line"] if pd.notna(row["type_line"]) else None,
                            "oracle_text": row["oracle_text"] if pd.notna(row["oracle_text"]) else None,
                            "colors": colors,
                            "color_identity": color_identity,
                            "power": row["power"] if pd.notna(row["power"]) else None,
                            "toughness": row["toughness"] if pd.notna(row["toughness"]) else None,
                            "loyalty": row["loyalty"] if pd.notna(row["loyalty"]) else None,
                            "rarity": row["rarity"] if pd.notna(row["rarity"]) else None,
                            "set_code": row["set"] if pd.notna(row["set"]) else None,
                            "set_name": row["set_name"] if pd.notna(row["set_name"]) else None,
                            "prices_usd": float(row["price_usd"]) if pd.notna(row["price_usd"]) else None,
                            "prices_usd_foil": float(row["price_usd_foil"]) if pd.notna(row["price_usd_foil"]) else None,
                            "legalities": legalities,
                            "image_uris": image_uris
                        }
                        
                        # Add text embedding if available
                        if text_embedding:
                            card_data["text_embedding"] = text_embedding
                        
                        # Add price embedding if available
                        if price_embedding:
                            card_data["price_embedding"] = price_embedding
                        
                        batch_cards.append(card_data)
                    
                    # Upsert batch to database
                    self.client.table("mtg_cards").upsert(batch_cards).execute()
                    
                    if i % 500 == 0 or i + batch_size >= len(df):
                        logger.info(f"Imported {min(i + batch_size, len(df))}/{len(df)} cards")
                        
                except Exception as e:
                    logger.error(f"Error importing batch {i}-{i+batch_size}: {e}")
            
            logger.info(f"Successfully imported {len(df)} cards to database")
            return True
            
        except Exception as e:
            logger.error(f"Error importing cards to database: {e}")
            return False
    
    def get_card_count(self):
        """Get the total count of cards in the database"""
        if not self.is_connected:
            logger.error("Cannot get card count: Not connected to database")
            return 0
        
        try:
            response = self.client.table("mtg_cards").select("id", count="exact").execute()
            return response.count
        except Exception as e:
            logger.error(f"Error getting card count: {e}")
            return 0
    
    def get_status(self):
        """Get status information about the database"""
        status = {
            "connected": self.is_connected
        }
        
        if self.is_connected:
            try:
                card_count = self.get_card_count()
                status["card_count"] = card_count
            except Exception as e:
                status["error"] = str(e)
        
        return status