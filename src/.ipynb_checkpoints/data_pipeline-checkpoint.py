import os
import json
import requests
import logging
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import SentenceTransformer, but gracefully handle if it fails
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformer not available, some features will be disabled")

# Import config
try:
    from src.config import DATA_DIR, SCRYFALL_API_URL, EMBEDDING_MODEL
except ImportError:
    # Default values if config import fails
    DATA_DIR = "data"
    SCRYFALL_API_URL = "https://api.scryfall.com"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

logger = logging.getLogger("DataPipeline")

class MTGDataPipeline:
    def __init__(self):
        """Initialize the data pipeline"""
        self.raw_dir = os.path.join(DATA_DIR, 'raw')
        self.processed_dir = os.path.join(DATA_DIR, 'processed')
        self.api_base_url = SCRYFALL_API_URL
        
        # Ensure directories exist
        Path(self.raw_dir).mkdir(exist_ok=True, parents=True)
        Path(self.processed_dir).mkdir(exist_ok=True, parents=True)
        
        self.raw_data_path = os.path.join(self.raw_dir, 'oracle_cards.json')
        self.processed_data_path = os.path.join(self.processed_dir, 'cards.csv')
        self.embeddings_path = os.path.join(self.processed_dir, 'embeddings.npy')
        self.card_ids_path = os.path.join(self.processed_dir, 'card_ids.json')
        self.dfc_mapping_path = os.path.join(self.processed_dir, 'dual_faced_mapping.json')
    
    def fetch_card_data(self, force_update=False):
        """Fetch card data from Scryfall API"""
        # Check if we already have recent data
        if os.path.exists(self.raw_data_path) and not force_update:
            file_mtime = os.path.getmtime(self.raw_data_path)
            last_modified = datetime.datetime.fromtimestamp(file_mtime)
            now = datetime.datetime.now()
            if (now - last_modified).total_seconds() < 86400:  # 24 hours
                logger.info("Using existing card data (less than 24 hours old)")
                return True

        logger.info("Fetching card data from Scryfall API")
        
        try:
            # Get bulk data information
            bulk_data_url = f"{self.api_base_url}/bulk-data"
            response = requests.get(bulk_data_url)
            response.raise_for_status()
            
            # Find the Oracle Cards data
            bulk_data = response.json()
            oracle_cards = next(item for item in bulk_data['data'] if item['type'] == 'oracle_cards')
            download_url = oracle_cards['download_uri']
            
            # Download the data
            logger.info(f"Downloading Oracle Cards data from {download_url}")
            response = requests.get(download_url)
            response.raise_for_status()
            
            # Save the raw data
            with open(self.raw_data_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Card data saved to {self.raw_data_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching card data: {str(e)}")
            return False
    
    def preprocess_cards(self):
        """Process raw card data into a structured format with dual-faced card support"""
        logger.info("Preprocessing card data")
        
        try:
            # Load raw data
            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                cards = json.load(f)
            
            # Process cards
            processed_cards = []
            for card in cards:
                # Only include English cards
                if card.get('lang') != 'en':
                    continue
                
                # Parse card data
                processed_card = {
                    'id': card.get('id', ''),
                    'name': card.get('name', ''),
                    'mana_cost': card.get('mana_cost', ''),
                    'cmc': card.get('cmc', 0),
                    'type_line': card.get('type_line', ''),
                    'oracle_text': card.get('oracle_text', ''),
                    'colors': ','.join(card.get('colors', [])),
                    'color_identity': ','.join(card.get('color_identity', [])),
                    'keywords': ','.join(card.get('keywords', [])),
                    'power': card.get('power', ''),
                    'toughness': card.get('toughness', ''),
                    'loyalty': card.get('loyalty', ''),
                    'legalities': json.dumps(card.get('legalities', {})),
                    'set': card.get('set', ''),
                    'set_name': card.get('set_name', ''),
                    'rarity': card.get('rarity', ''),
                    'price_usd': card.get('prices', {}).get('usd'),
                    'price_usd_foil': card.get('prices', {}).get('usd_foil'),
                    'image_uris': json.dumps(card.get('image_uris', {})) if 'image_uris' in card else None,
                    # Add dual-faced card support
                    'card_faces': json.dumps(card.get('card_faces', [])) if 'card_faces' in card else None,
                    'layout': card.get('layout', '')
                }
                
                processed_cards.append(processed_card)
            
            # Create DataFrame
            df = pd.DataFrame(processed_cards)
            
            # Save processed data
            df.to_csv(self.processed_data_path, index=False)
            
            # Create dual-faced card mapping
            self.create_dual_faced_mapping(cards)
            
            logger.info(f"Processed {len(df)} cards and saved to {self.processed_data_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing cards: {str(e)}")
            raise
    
    def create_dual_faced_mapping(self, cards):
        """Create mapping between dual-faced cards and their faces"""
        dual_faced_mapping = {}
        dfc_layouts = ['transform', 'modal_dfc', 'double_faced_token', 'flip', 'adventure']
        
        for card in cards:
            if 'card_faces' in card and card.get('layout') in dfc_layouts:
                main_name = card.get('name')
                
                for face in card.get('card_faces', []):
                    face_name = face.get('name')
                    if face_name and face_name != main_name:
                        dual_faced_mapping[face_name] = main_name
        
        # Save mapping
        with open(self.dfc_mapping_path, 'w') as f:
            json.dump(dual_faced_mapping, f)
        
        logger.info(f"Created mapping for {len(dual_faced_mapping)} dual-faced cards")
    
    def generate_embeddings(self, model_name=None):
        """Generate embeddings for card text with dual-faced card support"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("SentenceTransformer not available. Skipping embedding generation.")
            return None
        
        model_name = model_name or EMBEDDING_MODEL
        logger.info(f"Generating embeddings using {model_name}")
        
        try:
            # Load processed cards
            df = pd.read_csv(self.processed_data_path)
            
            # Initialize the embedding model
            model = SentenceTransformer(model_name)
            
            # Prepare text for embedding
            texts = []
            card_ids = []
            
            for _, row in df.iterrows():
                card_id = row['id']
                name = row['name']
                type_line = row['type_line']
                oracle_text = row['oracle_text'] if pd.notna(row['oracle_text']) else ""
                
                # Main card text
                text = f"{name} {type_line} {oracle_text}"
                texts.append(text)
                card_ids.append(card_id)
                
                # Handle dual-faced cards
                if pd.notna(row['card_faces']) and row['card_faces'] != '[]':
                    try:
                        card_faces = json.loads(row['card_faces'])
                        for face in card_faces:
                            face_name = face.get('name', '')
                            face_type = face.get('type_line', '')
                            face_text = face.get('oracle_text', '')
                            
                            # Skip if it's the same as the main face
                            if face_name == name:
                                continue
                                
                            # Create face text
                            face_text = f"{face_name} {face_type} {face_text}"
                            texts.append(face_text)
                            # Use the same card ID but add a suffix for faces
                            card_ids.append(f"{card_id}_face_{face_name}")
                    except Exception as e:
                        logger.error(f"Error processing card faces for {name}: {str(e)}")
            
            # Generate embeddings with batching
            logger.info(f"Generating embeddings for {len(texts)} cards/faces")
            batch_size = 32  # Adjust based on GPU memory
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
                all_embeddings.append(batch_embeddings)
                
                if i % 1000 == 0 and i > 0:
                    logger.info(f"Processed {i}/{len(texts)} embeddings")
            
            # Combine all batches
            embeddings = np.vstack(all_embeddings)
            
            # Save embeddings
            np.save(self.embeddings_path, embeddings)
            
            # Save card IDs for reference
            with open(self.card_ids_path, 'w') as f:
                json.dump(card_ids, f)
            
            # Save embedding texts for reference
            embedding_texts_path = os.path.join(self.processed_dir, 'embedding_texts.json')
            with open(embedding_texts_path, 'w') as f:
                json.dump(texts, f)
            
            logger.info(f"Embeddings generated with shape {embeddings.shape} and saved to {self.embeddings_path}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            logger.warning("Continuing without embeddings")
            return None
    
    def get_dual_faced_mapping(self):
        """Load dual-faced card mapping from file"""
        if os.path.exists(self.dfc_mapping_path):
            try:
                with open(self.dfc_mapping_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading dual-faced mapping: {str(e)}")
                return {}
        else:
            logger.warning("Dual-faced mapping file not found")
            return {}
    
    def migrate_database_for_dual_faced_cards(self, db_interface):
        """Add card_faces column to database"""
        if not db_interface or not db_interface.is_connected:
            logger.error("Database interface not available or not connected")
            return False
            
        try:
            logger.info("Migrating database to support dual-faced cards")
            db_interface.client.query("""
                ALTER TABLE mtg_cards ADD COLUMN IF NOT EXISTS card_faces JSONB;
                ALTER TABLE mtg_cards ADD COLUMN IF NOT EXISTS layout VARCHAR(50);
            """).execute()
            logger.info("Database migration completed successfully")
            return True
        except Exception as e:
            logger.error(f"Database migration failed: {str(e)}")
            return False
    
    def run_pipeline(self, force_update=False, skip_embeddings=False):
        """Run the full data pipeline"""
        logger.info("Starting data pipeline")
        
        try:
            # Fetch card data
            if self.fetch_card_data(force_update):
                # Preprocess the data
                df = self.preprocess_cards()
                
                # Generate embeddings if not skipped
                embeddings = None
                if not skip_embeddings:
                    embeddings = self.generate_embeddings()
                else:
                    logger.info("Skipping embedding generation")
                
                logger.info("Data pipeline completed successfully")
                return {
                    "success": True,
                    "cards_processed": len(df),
                    "dual_faced_cards_mapped": len(self.get_dual_faced_mapping()),
                    "embeddings_generated": embeddings is not None,
                    "embeddings_shape": embeddings.shape if embeddings is not None else None
                }
            
            logger.warning("Data pipeline didn't run (data already up to date)")
            return {
                "success": False,
                "message": "Data already up to date"
            }
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_status(self):
        """Get the status of the data pipeline"""
        status = {
            "raw_data_exists": os.path.exists(self.raw_data_path),
            "processed_data_exists": os.path.exists(self.processed_data_path),
            "embeddings_exist": os.path.exists(self.embeddings_path),
            "dual_faced_mapping_exists": os.path.exists(self.dfc_mapping_path),
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        }
        
        if status["raw_data_exists"]:
            status["raw_data_last_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.raw_data_path)
            ).isoformat()
            
            # Get file size - Convert to regular float
            status["raw_data_size_mb"] = float(os.path.getsize(self.raw_data_path) / (1024 * 1024))
        
        if status["processed_data_exists"]:
            status["processed_data_last_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.processed_data_path)
            ).isoformat()
            
            # Get card count
            try:
                df = pd.read_csv(self.processed_data_path)
                status["card_count"] = int(len(df))  # Convert to regular int
                # Count dual-faced cards
                if 'card_faces' in df.columns:
                    status["dual_faced_count"] = int(df['card_faces'].notna().sum())  # Convert to regular int
            except:
                status["card_count"] = "Error reading file"
        
        if status["embeddings_exist"]:
            status["embeddings_last_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.embeddings_path)
            ).isoformat()
            
            # Get embeddings shape - Convert NumPy shape to list of ints
            try:
                embeddings = np.load(self.embeddings_path)
                # Convert NumPy shape tuple to regular Python list
                status["embeddings_shape"] = [int(dim) for dim in embeddings.shape]
                status["embeddings_size_mb"] = float(os.path.getsize(self.embeddings_path) / (1024 * 1024))
            except:
                status["embeddings_shape"] = "Error reading file"
        
        if status["dual_faced_mapping_exists"]:
            status["dual_faced_mapping_last_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.dfc_mapping_path)
            ).isoformat()
            
            # Get count of mapped cards
            try:
                mapping = self.get_dual_faced_mapping()
                status["dual_faced_mapping_count"] = int(len(mapping))  # Convert to regular int
            except:
                status["dual_faced_mapping_count"] = "Error reading file"
        
        return status