# src/data_pipeline.py - Enhanced Data Pipeline with Image Support
import os
import json
import requests
import logging
import datetime
import pandas as pd
import numpy as np
import hashlib
import time
import torch
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        """Initialize the data pipeline with comprehensive image storage support"""
        self.raw_dir = os.path.join(DATA_DIR, 'raw')
        self.processed_dir = os.path.join(DATA_DIR, 'processed')
        self.images_dir = os.path.join(DATA_DIR, 'images')
        self.api_base_url = SCRYFALL_API_URL
        
        # Enhanced image storage configuration
        self.image_sizes = ['small', 'normal', 'large', 'png', 'art_crop', 'border_crop']
        self.preferred_sizes = ['normal', 'large', 'small']  # Order of preference
        self.max_workers = 5  # For concurrent image downloads
        self.download_timeout = 30  # Seconds
        
        # Ensure directories exist
        Path(self.raw_dir).mkdir(exist_ok=True, parents=True)
        Path(self.processed_dir).mkdir(exist_ok=True, parents=True)
        Path(self.images_dir).mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for different image sizes
        for size in self.image_sizes:
            Path(os.path.join(self.images_dir, size)).mkdir(exist_ok=True, parents=True)
        
        # File paths
        self.raw_data_path = os.path.join(self.raw_dir, 'oracle_cards.json')
        self.processed_data_path = os.path.join(self.processed_dir, 'cards.csv')
        self.embeddings_path = os.path.join(self.processed_dir, 'embeddings.npy')
        self.card_ids_path = os.path.join(self.processed_dir, 'card_ids.json')
        self.dfc_mapping_path = os.path.join(self.processed_dir, 'dual_faced_mapping.json')
        self.image_manifest_path = os.path.join(self.processed_dir, 'image_manifest.json')
        self.failed_downloads_path = os.path.join(self.processed_dir, 'failed_image_downloads.json')
    
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
    
    def download_card_image(self, card_id, image_url, size='normal', max_retries=3):
        """Download a single card image with comprehensive error handling and retries"""
        if not image_url:
            return None
            
        try:
            # Create filename based on card ID and size
            file_extension = '.jpg'  # Most Scryfall images are JPG
            if 'png' in image_url.lower() or size == 'png':
                file_extension = '.png'
            
            filename = f"{card_id}_{size}{file_extension}"
            local_path = os.path.join(self.images_dir, size, filename)
            
            # Skip if file already exists and is valid
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                return {
                    'card_id': card_id,
                    'size': size,
                    'local_path': local_path,
                    'file_size': os.path.getsize(local_path),
                    'status': 'exists'
                }
            
            # Download with retries and exponential backoff
            for attempt in range(max_retries):
                try:
                    response = requests.get(
                        image_url, 
                        timeout=self.download_timeout,
                        stream=True,
                        headers={'User-Agent': 'MTG-AI-Framework/1.0'}
                    )
                    response.raise_for_status()
                    
                    # Save the image
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # Filter out keep-alive chunks
                                f.write(chunk)
                    
                    # Verify the file was written and has content
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        return {
                            'card_id': card_id,
                            'size': size,
                            'local_path': local_path,
                            'original_url': image_url,
                            'file_size': os.path.getsize(local_path),
                            'status': 'downloaded'
                        }
                    else:
                        raise Exception("Downloaded file is empty or corrupt")
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to download image for {card_id} (size: {size}) after {max_retries} attempts: {e}")
                        return {
                            'card_id': card_id,
                            'size': size,
                            'error': str(e),
                            'status': 'failed'
                        }
                    else:
                        wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(f"Download attempt {attempt + 1} failed for {card_id}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        
        except Exception as e:
            logger.error(f"Error downloading image for {card_id}: {e}")
            return {
                'card_id': card_id,
                'size': size,
                'error': str(e),
                'status': 'failed'
            }
    
    def download_card_images(self, cards_data, download_images=True, size_preference=None):
        """Download images for all cards with enhanced concurrent processing and progress tracking"""
        if not download_images:
            logger.info("Image downloading disabled")
            return {}
        
        size_preference = size_preference or self.preferred_sizes
        logger.info(f"Starting image download for {len(cards_data)} cards")
        
        # Prepare download tasks with intelligent size selection
        download_tasks = []
        
        for card in cards_data:
            card_id = card.get('id')
            if not card_id:
                continue
                
            # Get image URLs for main card
            image_uris = card.get('image_uris', {})
            card_faces = card.get('card_faces', [])
            
            # Add main card images (select best available size)
            if image_uris:
                for size in size_preference:
                    if size in image_uris:
                        download_tasks.append((card_id, image_uris[size], size))
                        break  # Only download one size per card (best available)
            
            # Add dual-faced card images
            for i, face in enumerate(card_faces):
                face_image_uris = face.get('image_uris', {})
                if not face_image_uris:
                    continue
                    
                face_name = face.get('name', f'face_{i}')
                # Create safe filename for face
                safe_face_name = ''.join(c for c in face_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                face_id = f"{card_id}_face_{safe_face_name.replace(' ', '_')}"
                
                for size in size_preference:
                    if size in face_image_uris:
                        download_tasks.append((face_id, face_image_uris[size], size))
                        break
        
        if not download_tasks:
            logger.warning("No download tasks generated")
            return {"message": "No images to download"}
        
        logger.info(f"Generated {len(download_tasks)} download tasks")
        
        # Download images concurrently with progress tracking
        download_results = []
        failed_downloads = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_task = {
                executor.submit(self.download_card_image, card_id, url, size): (card_id, url, size)
                for card_id, url, size in download_tasks
            }
            
            # Process completed downloads with progress tracking
            completed = 0
            total_tasks = len(future_to_task)
            
            for future in as_completed(future_to_task):
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        if result.get('status') == 'failed':
                            failed_downloads.append(result)
                        else:
                            download_results.append(result)
                    
                    # Progress logging every 100 downloads or at completion
                    if completed % 100 == 0 or completed == total_tasks:
                        success_count = len(download_results)
                        failed_count = len(failed_downloads)
                        logger.info(f"Progress: {completed}/{total_tasks} - Success: {success_count}, Failed: {failed_count}")
                        
                except Exception as e:
                    card_id, url, size = future_to_task[future]
                    logger.error(f"Error processing download for {card_id}: {e}")
                    failed_downloads.append({
                        'card_id': card_id,
                        'size': size,
                        'error': str(e),
                        'status': 'failed'
                    })
        
        # Create comprehensive image manifest
        image_manifest = {
            'download_date': datetime.datetime.now().isoformat(),
            'total_attempted': len(download_tasks),
            'successful_downloads': len(download_results),
            'failed_downloads': len(failed_downloads),
            'success_rate': len(download_results) / len(download_tasks) * 100 if download_tasks else 0,
            'size_preferences': size_preference,
            'images': {result['card_id']: result for result in download_results}
        }
        
        # Save manifest and failed downloads
        with open(self.image_manifest_path, 'w') as f:
            json.dump(image_manifest, f, indent=2)
            
        if failed_downloads:
            with open(self.failed_downloads_path, 'w') as f:
                json.dump(failed_downloads, f, indent=2)
        
        logger.info(f"Image download completed: {len(download_results)} successful ({image_manifest['success_rate']:.1f}%), {len(failed_downloads)} failed")
        return image_manifest
    
    def preprocess_cards(self, download_images=True):
        """Process raw card data into a structured format with comprehensive image support"""
        logger.info("Preprocessing card data with enhanced image support")
        
        try:
            # Load raw data
            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                cards = json.load(f)
            
            logger.info(f"Loaded {len(cards)} cards from raw data")
            
            # Download images if requested
            image_manifest = {}
            if download_images:
                image_manifest = self.download_card_images(cards, download_images)
            
            # Process cards with enhanced data structure
            processed_cards = []
            for card in cards:
                # Only include English cards
                if card.get('lang') != 'en':
                    continue
                
                # Get local image paths if available
                local_image_paths = {}
                card_id = card.get('id', '')
                
                if image_manifest and card_id in image_manifest.get('images', {}):
                    image_info = image_manifest['images'][card_id]
                    if image_info.get('status') in ['downloaded', 'exists']:
                        local_image_paths[image_info['size']] = image_info['local_path']
                
                # Parse card data with comprehensive field extraction
                processed_card = {
                    'id': card_id,
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
                    'local_image_paths': json.dumps(local_image_paths) if local_image_paths else None,
                    # Enhanced dual-faced card support
                    'card_faces': json.dumps(card.get('card_faces', [])) if 'card_faces' in card else None,
                    'layout': card.get('layout', ''),
                    # Additional metadata
                    'scryfall_uri': card.get('scryfall_uri', ''),
                    'rulings_uri': card.get('rulings_uri', ''),
                    'artist': card.get('artist', ''),
                    'border_color': card.get('border_color', ''),
                    'frame': card.get('frame', ''),
                    'full_art': card.get('full_art', False),
                    'textless': card.get('textless', False),
                    'booster': card.get('booster', False),
                    'story_spotlight': card.get('story_spotlight', False),
                    'promo': card.get('promo', False),
                    'digital': card.get('digital', False),
                    'flavor_text': card.get('flavor_text', ''),
                    'collector_number': card.get('collector_number', ''),
                    'released_at': card.get('released_at', ''),
                }
                
                processed_cards.append(processed_card)
            
            # Create DataFrame with enhanced data
            df = pd.DataFrame(processed_cards)
            
            # Save processed data
            df.to_csv(self.processed_data_path, index=False)
            
            # Create dual-faced card mapping
            self.create_dual_faced_mapping(cards)
            
            logger.info(f"Processed {len(df)} cards and saved to {self.processed_data_path}")
            
            if download_images and image_manifest:
                logger.info(f"Downloaded images for {image_manifest.get('successful_downloads', 0)} cards "
                           f"({image_manifest.get('success_rate', 0):.1f}% success rate)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing cards: {str(e)}")
            raise
    
    def create_dual_faced_mapping(self, cards):
        """Create comprehensive mapping between dual-faced cards and their faces"""
        dual_faced_mapping = {}
        dfc_layouts = ['transform', 'modal_dfc', 'double_faced_token', 'flip', 'adventure', 'split', 'meld']
        
        for card in cards:
            if 'card_faces' in card and card.get('layout') in dfc_layouts:
                main_name = card.get('name')
                card_id = card.get('id')
                
                # Store comprehensive mapping info
                face_info = {
                    'main_card_name': main_name,
                    'card_id': card_id,
                    'layout': card.get('layout'),
                    'faces': {}
                }
                
                for i, face in enumerate(card.get('card_faces', [])):
                    face_name = face.get('name')
                    if face_name and face_name != main_name:
                        face_info['faces'][face_name] = {
                            'face_index': i,
                            'mana_cost': face.get('mana_cost', ''),
                            'type_line': face.get('type_line', ''),
                            'oracle_text': face.get('oracle_text', ''),
                            'colors': face.get('colors', []),
                            'power': face.get('power'),
                            'toughness': face.get('toughness'),
                            'loyalty': face.get('loyalty'),
                            'flavor_text': face.get('flavor_text', ''),
                            'artist': face.get('artist', ''),
                            'image_uris': face.get('image_uris', {})
                        }
                        
                        # Also create reverse mapping
                        dual_faced_mapping[face_name] = main_name
                
                # Store comprehensive face information
                if face_info['faces']:
                    dual_faced_mapping[f"{main_name}_faces"] = face_info
        
        # Save mapping with enhanced structure
        with open(self.dfc_mapping_path, 'w') as f:
            json.dump(dual_faced_mapping, f, indent=2)
        
        # Count actual face mappings (excluding _faces entries)
        face_count = sum(1 for k in dual_faced_mapping.keys() if not k.endswith('_faces'))
        logger.info(f"Created mapping for {face_count} dual-faced card faces")
    
    def generate_embeddings(self, model_name=None):
        """Generate embeddings for card text with enhanced dual-faced card support"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("SentenceTransformer not available. Skipping embedding generation.")
            return None
        
        model_name = model_name or EMBEDDING_MODEL
        logger.info(f"Generating embeddings using {model_name}")
        
        try:
            # Load processed cards
            df = pd.read_csv(self.processed_data_path)

            # Initialize the embedding model with an explicit device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading embedding model on {device}")
            model = SentenceTransformer(model_name, device=device)
            
            # Prepare text for embedding with enhanced processing
            texts = []
            card_ids = []
            embedding_metadata = []
            
            for _, row in df.iterrows():
                card_id = row['id']
                name = row['name']
                type_line = row['type_line']
                oracle_text = row['oracle_text'] if pd.notna(row['oracle_text']) else ""
                flavor_text = row.get('flavor_text', '') if pd.notna(row.get('flavor_text', '')) else ""
                
                # Enhanced main card text with more context
                main_text = f"{name} {type_line} {oracle_text}"
                if flavor_text:
                    main_text += f" {flavor_text}"
                
                texts.append(main_text.strip())
                card_ids.append(card_id)
                embedding_metadata.append({
                    'card_id': card_id,
                    'card_name': name,
                    'embedding_type': 'main',
                    'face_name': None
                })
                
                # Handle dual-faced cards with comprehensive face processing
                if pd.notna(row['card_faces']) and row['card_faces'] != '[]':
                    try:
                        card_faces = json.loads(row['card_faces'])
                        for i, face in enumerate(card_faces):
                            face_name = face.get('name', '')
                            face_type = face.get('type_line', '')
                            face_text = face.get('oracle_text', '')
                            face_flavor = face.get('flavor_text', '')
                            
                            # Skip if it's the same as the main face
                            if face_name == name:
                                continue
                                
                            # Create comprehensive face text
                            face_full_text = f"{face_name} {face_type} {face_text}"
                            if face_flavor:
                                face_full_text += f" {face_flavor}"
                            
                            texts.append(face_full_text.strip())
                            # Use the same card ID but add face identifier
                            face_card_id = f"{card_id}_face_{i}_{face_name.replace(' ', '_')}"
                            card_ids.append(face_card_id)
                            embedding_metadata.append({
                                'card_id': card_id,
                                'card_name': name,
                                'embedding_type': 'face',
                                'face_name': face_name,
                                'face_index': i
                            })
                            
                    except Exception as e:
                        logger.error(f"Error processing card faces for {name}: {str(e)}")
            
            # Generate embeddings with enhanced batching and progress tracking
            logger.info(f"Generating embeddings for {len(texts)} cards/faces")
            batch_size = 32  # Adjust based on GPU memory
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                try:
                    batch_embeddings = model.encode(
                        batch_texts, 
                        show_progress_bar=False,  # We'll handle progress ourselves
                        convert_to_numpy=True,
                        normalize_embeddings=True  # Normalize for better similarity search
                    )
                    all_embeddings.append(batch_embeddings)
                    
                    if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(texts):
                        logger.info(f"Generated embeddings: {min(i + batch_size, len(texts))}/{len(texts)}")
                        
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {i}-{i+batch_size}: {e}")
                    # Create zero embeddings as fallback
                    fallback_embeddings = np.zeros((len(batch_texts), model.get_sentence_embedding_dimension()))
                    all_embeddings.append(fallback_embeddings)
            
            # Combine all batches
            embeddings = np.vstack(all_embeddings)
            
            # Save embeddings with enhanced metadata
            np.save(self.embeddings_path, embeddings)
            
            # Save card IDs for reference
            with open(self.card_ids_path, 'w') as f:
                json.dump(card_ids, f)
            
            # Save comprehensive embedding metadata
            embedding_texts_path = os.path.join(self.processed_dir, 'embedding_texts.json')
            with open(embedding_texts_path, 'w') as f:
                json.dump(texts, f)
            
            embedding_metadata_path = os.path.join(self.processed_dir, 'embedding_metadata.json')
            with open(embedding_metadata_path, 'w') as f:
                json.dump({
                    'model_name': model_name,
                    'embedding_dimension': embeddings.shape[1],
                    'total_embeddings': embeddings.shape[0],
                    'generation_date': datetime.datetime.now().isoformat(),
                    'metadata': embedding_metadata
                }, f, indent=2)
            
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
    
    def get_image_stats(self):
        """Get comprehensive statistics about downloaded images"""
        stats = {
            'manifest_exists': os.path.exists(self.image_manifest_path),
            'failed_downloads_exist': os.path.exists(self.failed_downloads_path),
            'local_images_count': 0,
            'total_size_mb': 0,
            'size_breakdown': {}
        }
        
        # Count local images and calculate total size by category
        if os.path.exists(self.images_dir):
            for size_dir in os.listdir(self.images_dir):
                size_path = os.path.join(self.images_dir, size_dir)
                if os.path.isdir(size_path):
                    size_count = 0
                    size_total_mb = 0
                    
                    for img_file in os.listdir(size_path):
                        img_path = os.path.join(size_path, img_file)
                        if os.path.isfile(img_path):
                            size_count += 1
                            size_total_mb += os.path.getsize(img_path)
                    
                    stats['local_images_count'] += size_count
                    stats['total_size_mb'] += size_total_mb
                    stats['size_breakdown'][size_dir] = {
                        'count': size_count,
                        'size_mb': round(size_total_mb / (1024 * 1024), 2)
                    }
        
        stats['total_size_mb'] = round(stats['total_size_mb'] / (1024 * 1024), 2)
        
        # Load manifest data if available
        if stats['manifest_exists']:
            try:
                with open(self.image_manifest_path, 'r') as f:
                    manifest = json.load(f)
                    stats['manifest_data'] = {
                        'download_date': manifest.get('download_date'),
                        'success_rate': manifest.get('success_rate', 0),
                        'total_attempted': manifest.get('total_attempted', 0),
                        'successful_downloads': manifest.get('successful_downloads', 0),
                        'failed_downloads': manifest.get('failed_downloads', 0)
                    }
            except Exception as e:
                logger.error(f"Error loading manifest: {e}")
        
        return stats
    
    def cleanup_images(self, keep_sizes=None):
        """Clean up downloaded images, keeping only specified sizes"""
        keep_sizes = keep_sizes or ['normal']
        logger.info(f"Cleaning up images, keeping sizes: {keep_sizes}")
        
        removed_count = 0
        removed_size = 0
        
        for size_dir in os.listdir(self.images_dir):
            if size_dir not in keep_sizes:
                size_path = os.path.join(self.images_dir, size_dir)
                if os.path.isdir(size_path):
                    for img_file in os.listdir(size_path):
                        img_path = os.path.join(size_path, img_file)
                        if os.path.isfile(img_path):
                            file_size = os.path.getsize(img_path)
                            try:
                                os.remove(img_path)
                                removed_count += 1
                                removed_size += file_size
                            except Exception as e:
                                logger.error(f"Error removing {img_path}: {e}")
                    
                    # Remove empty directory
                    try:
                        os.rmdir(size_path)
                        logger.info(f"Removed empty directory: {size_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove directory {size_path}: {e}")
        
        removed_size_mb = removed_size / (1024*1024)
        logger.info(f"Cleanup completed: removed {removed_count} images, freed {removed_size_mb:.2f} MB")
        return removed_count, removed_size
    
    def run_pipeline(self, force_update=False, skip_embeddings=False, download_images=True):
        """Run the comprehensive data pipeline with enhanced image support"""
        logger.info("Starting comprehensive MTG data pipeline with image support")
        
        try:
            # Step 1: Fetch card data
            if not self.fetch_card_data(force_update):
                return {
                    "success": False,
                    "error": "Failed to fetch card data"
                }
            
            # Step 2: Preprocess the data with image downloading
            df = self.preprocess_cards(download_images)
            
            # Step 3: Generate embeddings if not skipped
            embeddings = None
            if not skip_embeddings:
                embeddings = self.generate_embeddings()
            else:
                logger.info("Skipping embedding generation")
            
            # Step 4: Get comprehensive statistics
            image_stats = self.get_image_stats()
            dual_faced_mapping = self.get_dual_faced_mapping()
            
            # Compile comprehensive results
            result = {
                "success": True,
                "cards_processed": len(df),
                "dual_faced_cards_mapped": len([k for k in dual_faced_mapping.keys() if not k.endswith('_faces')]),
                "embeddings_generated": embeddings is not None,
                "embeddings_shape": embeddings.shape if embeddings is not None else None,
                "images_downloaded": image_stats.get('local_images_count', 0),
                "images_size_mb": image_stats.get('total_size_mb', 0),
                "image_success_rate": image_stats.get('manifest_data', {}).get('success_rate', 0),
                "image_stats": image_stats,
                "pipeline_completed_at": datetime.datetime.now().isoformat()
            }
            
            logger.info("Comprehensive data pipeline completed successfully")
            logger.info(f"Summary: {result['cards_processed']} cards, "
                       f"{result['images_downloaded']} images ({result['image_success_rate']:.1f}% success), "
                       f"{result['dual_faced_cards_mapped']} dual-faced cards")
            
            return result
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "failed_at": datetime.datetime.now().isoformat()
            }
    
    def get_status(self):
        """Get comprehensive status of the data pipeline"""
        status = {
            "raw_data_exists": os.path.exists(self.raw_data_path),
            "processed_data_exists": os.path.exists(self.processed_data_path),
            "embeddings_exist": os.path.exists(self.embeddings_path),
            "dual_faced_mapping_exists": os.path.exists(self.dfc_mapping_path),
            "image_manifest_exists": os.path.exists(self.image_manifest_path),
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "status_checked_at": datetime.datetime.now().isoformat()
        }
        
        # Raw data status
        if status["raw_data_exists"]:
            status["raw_data_last_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.raw_data_path)
            ).isoformat()
            status["raw_data_size_mb"] = float(os.path.getsize(self.raw_data_path) / (1024 * 1024))
        
        # Processed data status
        if status["processed_data_exists"]:
            status["processed_data_last_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.processed_data_path)
            ).isoformat()
            
            try:
                df = pd.read_csv(self.processed_data_path)
                status["card_count"] = int(len(df))
                
                # Enhanced statistics
                if 'card_faces' in df.columns:
                    status["dual_faced_count"] = int(df['card_faces'].notna().sum())
                if 'local_image_paths' in df.columns:
                    status["cards_with_images"] = int(df['local_image_paths'].notna().sum())
                if 'layout' in df.columns:
                    status["layout_distribution"] = df['layout'].value_counts().to_dict()
                    
            except Exception as e:
                status["card_count"] = f"Error reading file: {e}"
        
        # Embeddings status
        if status["embeddings_exist"]:
            status["embeddings_last_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.embeddings_path)
            ).isoformat()
            
            try:
                embeddings = np.load(self.embeddings_path)
                status["embeddings_shape"] = [int(dim) for dim in embeddings.shape]
                status["embeddings_size_mb"] = float(os.path.getsize(self.embeddings_path) / (1024 * 1024))
            except Exception as e:
                status["embeddings_shape"] = f"Error reading file: {e}"
        
        # Dual-faced mapping status
        if status["dual_faced_mapping_exists"]:
            try:
                mapping = self.get_dual_faced_mapping()
                status["dual_faced_mapping_count"] = int(len([k for k in mapping.keys() if not k.endswith('_faces')]))
            except Exception as e:
                status["dual_faced_mapping_count"] = f"Error reading file: {e}"
        
        # Enhanced image statistics
        image_stats = self.get_image_stats()
        status.update({
            "local_images_count": image_stats.get('local_images_count', 0),
            "local_images_size_mb": image_stats.get('total_size_mb', 0),
            "image_size_breakdown": image_stats.get('size_breakdown', {}),
            "image_manifest_data": image_stats.get('manifest_data', {}),
            "failed_downloads_exist": image_stats.get('failed_downloads_exist', False)
        })
        
        return status