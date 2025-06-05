import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("ImageManager")

@dataclass
class ImageInfo:
    """Data class for image information"""
    card_id: str
    size: str
    local_path: Optional[str] = None
    storage_url: Optional[str] = None
    original_url: Optional[str] = None
    file_size: Optional[int] = None
    status: str = 'unknown'
    error: Optional[str] = None

class MTGImageManager:
    """Centralized manager for MTG card images"""
    
    def __init__(self, db_interface=None, data_pipeline=None):
        """Initialize the image manager"""
        self.db = db_interface
        self.pipeline = data_pipeline
        
        # Import components if not provided
        if self.db is None:
            from src.db_interface import DatabaseInterface
            self.db = DatabaseInterface()
        
        if self.pipeline is None:
            from src.data_pipeline import MTGDataPipeline
            self.pipeline = MTGDataPipeline()
        
        self.preferred_sizes = ['normal', 'large', 'small']
        self.fallback_enabled = True
    
    def get_card_image_info(self, card_name_or_id: str, size: str = 'normal') -> Optional[ImageInfo]:
        """Get comprehensive image information for a card"""
        try:
            # Try to find card by name first, then by ID
            card_data = None
            
            if self.db.is_connected:
                # Try by name
                response = self.db.client.table("mtg_cards").select(
                    "scryfall_id, name, storage_image_urls, local_image_paths, image_uris"
                ).eq("name", card_name_or_id).limit(1).execute()
                
                if not response.data:
                    # Try by scryfall_id
                    response = self.db.client.table("mtg_cards").select(
                        "scryfall_id, name, storage_image_urls, local_image_paths, image_uris"
                    ).eq("scryfall_id", card_name_or_id).limit(1).execute()
                
                if response.data:
                    card_data = response.data[0]
            
            if not card_data:
                return None
            
            card_id = card_data['scryfall_id']
            image_info = ImageInfo(card_id=card_id, size=size)
            
            # Check storage URLs first
            if card_data.get('storage_image_urls'):
                try:
                    storage_urls = card_data['storage_image_urls']
                    if isinstance(storage_urls, str):
                        storage_urls = json.loads(storage_urls)
                    
                    if size in storage_urls:
                        image_info.storage_url = storage_urls[size]
                        image_info.status = 'storage'
                        return image_info
                except:
                    pass
            
            # Check local paths
            if card_data.get('local_image_paths'):
                try:
                    local_paths = card_data['local_image_paths']
                    if isinstance(local_paths, str):
                        local_paths = json.loads(local_paths)
                    
                    if size in local_paths:
                        local_path = local_paths[size]
                        if os.path.exists(local_path):
                            image_info.local_path = local_path
                            image_info.file_size = os.path.getsize(local_path)
                            image_info.status = 'local'
                            return image_info
                except:
                    pass
            
            # Fallback to original Scryfall URLs
            if card_data.get('image_uris'):
                try:
                    image_uris = card_data['image_uris']
                    if isinstance(image_uris, str):
                        image_uris = json.loads(image_uris)
                    
                    if size in image_uris:
                        image_info.original_url = image_uris[size]
                        image_info.status = 'original'
                        return image_info
                except:
                    pass
            
            # Try alternative sizes if requested size not found
            if self.fallback_enabled:
                for alt_size in self.preferred_sizes:
                    if alt_size != size:
                        alt_info = self.get_card_image_info(card_name_or_id, alt_size)
                        if alt_info and alt_info.status != 'unknown':
                            alt_info.size = alt_size  # Keep the actual size found
                            return alt_info
            
            image_info.status = 'not_found'
            return image_info
            
        except Exception as e:
            logger.error(f"Error getting image info for {card_name_or_id}: {e}")
            return ImageInfo(
                card_id=card_name_or_id,
                size=size,
                status='error',
                error=str(e)
            )
    
    def get_best_image_url(self, card_name_or_id: str, size: str = 'normal', 
                          prefer_storage: bool = True) -> Optional[str]:
        """Get the best available image URL for a card"""
        image_info = self.get_card_image_info(card_name_or_id, size)
        
        if not image_info:
            return None
        
        # Return URL based on preference and availability
        if prefer_storage and image_info.storage_url:
            return image_info.storage_url
        elif image_info.local_path and os.path.exists(image_info.local_path):
            # For local files, you might want to serve them through a local web server
            # or convert the path to a file:// URL
            return f"file://{os.path.abspath(image_info.local_path)}"
        elif image_info.original_url:
            return image_info.original_url
        else:
            return None
    
    def download_missing_images(self, card_names: List[str], 
                               sizes: List[str] = None) -> Dict[str, Any]:
        """Download images for cards that don't have them locally"""
        if sizes is None:
            sizes = ['normal']
        
        logger.info(f"Checking image availability for {len(card_names)} cards")
        
        # Find cards that need image downloads
        cards_to_download = []
        
        for card_name in card_names:
            for size in sizes:
                image_info = self.get_card_image_info(card_name, size)
                
                if not image_info or image_info.status in ['not_found', 'error']:
                    # Try to get the card's Scryfall data for download
                    if self.db.is_connected:
                        response = self.db.client.table("mtg_cards").select(
                            "scryfall_id, name, image_uris"
                        ).eq("name", card_name).limit(1).execute()
                        
                        if response.data:
                            card_data = response.data[0]
                            image_uris = card_data.get('image_uris')
                            
                            if image_uris:
                                if isinstance(image_uris, str):
                                    image_uris = json.loads(image_uris)
                                
                                if size in image_uris:
                                    cards_to_download.append({
                                        'id': card_data['scryfall_id'],
                                        'name': card_data['name'],
                                        'size': size,
                                        'url': image_uris[size]
                                    })
        
        if not cards_to_download:
            logger.info("No images need to be downloaded")
            return {"downloaded": 0, "failed": 0, "details": []}
        
        logger.info(f"Downloading {len(cards_to_download)} missing images")
        
        # Use the pipeline's download functionality
        download_results = []
        failed_downloads = []
        
        for card_info in cards_to_download:
            try:
                result = self.pipeline.download_card_image(
                    card_info['id'],
                    card_info['url'],
                    card_info['size']
                )
                
                if result and result.get('status') == 'downloaded':
                    download_results.append(result)
                else:
                    failed_downloads.append(result or card_info)
                    
            except Exception as e:
                logger.error(f"Error downloading image for {card_info['name']}: {e}")
                failed_downloads.append({
                    **card_info,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return {
            "downloaded": len(download_results),
            "failed": len(failed_downloads),
            "details": download_results,
            "failures": failed_downloads
        }
    
    def upload_to_storage(self, card_names: List[str] = None, 
                         sizes: List[str] = None, force_reupload: bool = False) -> Dict[str, Any]:
        """Upload local images to cloud storage - COMPLETELY FIXED VERSION"""
        if not self.db.image_storage_enabled:
            return {"error": "Image storage not enabled"}
        
        if sizes is None:
            sizes = ['normal']
        
        logger.info("Starting upload to cloud storage - scanning actual files")
        
        # Scan actual files on disk
        upload_tasks = []
        
        for size in sizes:
            size_dir = os.path.join(self.pipeline.images_dir, size)
            if not os.path.exists(size_dir):
                logger.warning(f"Size directory doesn't exist: {size_dir}")
                continue
            
            logger.info(f"Scanning {size} images in {size_dir}")
            
            # Get all image files
            image_files = []
            for f in os.listdir(size_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(size_dir, f)
                    if os.path.isfile(full_path):
                        image_files.append(f)
            
            logger.info(f"Found {len(image_files)} {size} images")
            
            for filename in image_files:
                local_path = os.path.join(size_dir, filename)
                
                try:
                    # Extract card ID from filename (format: cardid_size.jpg)
                    card_id = filename.split('_')[0]
                    
                    # If specific cards requested, filter
                    if card_names:
                        if self.db.is_connected:
                            response = self.db.client.table("mtg_cards").select("name").eq("scryfall_id", card_id).limit(1).execute()
                            if not response.data:
                                continue
                            card_name = response.data[0]['name']
                            if card_name not in card_names:
                                continue
                    
                    upload_tasks.append({
                        'card_id': card_id,
                        'local_path': local_path,
                        'size': size,
                        'filename': filename
                    })
                    
                except Exception as e:
                    logger.warning(f"Could not process filename {filename}: {e}")
                    continue
        
        if not upload_tasks:
            return {"uploaded": 0, "failed": 0, "message": "No images to upload"}
        
        logger.info(f"Prepared {len(upload_tasks)} images for upload")
        
        # Track existing files to avoid duplicate checks
        existing_files_cache = {}
        
        # Upload files
        upload_results = []
        failed_uploads = []
        skipped_count = 0
        
        for i, task in enumerate(upload_tasks):
            try:
                storage_path = f"images/{task['size']}/{task['filename']}"
                
                # Check existence only if not forcing reupload
                should_upload = force_reupload
                
                if not force_reupload:
                    # Check cache first
                    cache_key = f"{task['size']}/{task['filename']}"
                    
                    if cache_key not in existing_files_cache:
                        try:
                            # List files in the size directory to check existence
                            if task['size'] not in existing_files_cache:
                                files_in_size_dir = self.db.client.storage.from_(self.db.image_bucket).list(f"images/{task['size']}")
                                existing_files_cache[task['size']] = {f['name'] for f in files_in_size_dir}
                            
                            # Check if our file exists
                            file_exists = task['filename'] in existing_files_cache[task['size']]
                            existing_files_cache[cache_key] = file_exists
                            
                        except Exception as e:
                            logger.debug(f"Could not check existence, assuming file doesn't exist: {e}")
                            existing_files_cache[cache_key] = False
                    
                    should_upload = not existing_files_cache[cache_key]
                    
                    if not should_upload:
                        skipped_count += 1
                        if skipped_count % 1000 == 0:
                            logger.info(f"Skipped {skipped_count} existing files")
                        continue
                
                # Upload the file
                result = self.db.upload_image_to_storage(
                    task['local_path'],
                    storage_path,
                    overwrite=force_reupload
                )
                
                if result and result.get('status') in ['uploaded', 'exists']:
                    result.update(task)
                    upload_results.append(result)
                    
                    # Progress logging
                    if len(upload_results) % 100 == 0:
                        logger.info(f"Uploaded {len(upload_results)} images ({len(upload_results) + skipped_count}/{len(upload_tasks)} processed)")
                else:
                    failed_uploads.append({**task, **(result or {})})
                    
            except Exception as e:
                logger.error(f"Error uploading {task.get('filename', 'unknown')}: {e}")
                failed_uploads.append({
                    **task,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Update database with storage URLs
        if upload_results:
            logger.info(f"Updating database with {len(upload_results)} new storage URLs")
            try:
                upload_summary = {
                    'results': upload_results,
                    'failures': failed_uploads
                }
                self.db.update_card_storage_urls(upload_summary)
                logger.info("Database updated successfully")
            except Exception as e:
                logger.error(f"Error updating database: {e}")
        
        total_processed = len(upload_results) + len(failed_uploads) + skipped_count
        logger.info(f"Upload completed: {len(upload_results)} uploaded, {len(failed_uploads)} failed, {skipped_count} skipped")
        
        return {
            "uploaded": len(upload_results),
            "failed": len(failed_uploads),
            "skipped": skipped_count,
            "total_processed": total_processed,
            "details": upload_results,
            "failures": failed_uploads
        }
    
    def get_image_statistics(self) -> Dict[str, Any]:
        """Get comprehensive image statistics"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "local_images": {},
            "storage_images": {},
            "database_stats": {},
            "pipeline_stats": {}
        }
        
        # Get pipeline stats
        if self.pipeline:
            try:
                pipeline_stats = self.pipeline.get_image_stats()
                stats["pipeline_stats"] = pipeline_stats
            except Exception as e:
                logger.error(f"Error getting pipeline stats: {e}")
                stats["pipeline_stats"] = {"error": str(e)}
        
        # Get database stats
        if self.db and self.db.is_connected:
            try:
                db_stats = self.db.get_image_storage_stats()
                stats["database_stats"] = db_stats
            except Exception as e:
                logger.error(f"Error getting database stats: {e}")
                stats["database_stats"] = {"error": str(e)}
        
        # Analyze local image distribution
        try:
            if os.path.exists(self.pipeline.images_dir):
                for item in os.listdir(self.pipeline.images_dir):
                    size_path = os.path.join(self.pipeline.images_dir, item)
                    if os.path.isdir(size_path):
                        files = [f for f in os.listdir(size_path) if os.path.isfile(os.path.join(size_path, f))]
                        count = len(files)
                        
                        size_mb = 0
                        for f in files:
                            try:
                                size_mb += os.path.getsize(os.path.join(size_path, f))
                            except:
                                pass
                        size_mb = size_mb / (1024 * 1024)
                        
                        stats["local_images"][item] = {
                            "count": count,
                            "size_mb": round(size_mb, 2)
                        }
        except Exception as e:
            logger.error(f"Error analyzing local images: {e}")
            stats["local_images"] = {"error": str(e)}
        
        return stats
    
    def cleanup_images(self, cleanup_local: bool = False, cleanup_storage: bool = False,
                      keep_sizes: List[str] = None) -> Dict[str, Any]:
        """Clean up images based on specified criteria"""
        if keep_sizes is None:
            keep_sizes = ['normal']
        
        results = {
            "local_cleanup": {},
            "storage_cleanup": {}
        }
        
        if cleanup_local and self.pipeline:
            logger.info("Cleaning up local images")
            try:
                removed_count, removed_size = self.pipeline.cleanup_images(keep_sizes)
                results["local_cleanup"] = {
                    "success": True,
                    "removed_count": removed_count,
                    "removed_size_mb": round(removed_size / (1024 * 1024), 2)
                }
            except Exception as e:
                logger.error(f"Error cleaning up local images: {e}")
                results["local_cleanup"] = {"success": False, "error": str(e)}
        
        if cleanup_storage and self.db and self.db.image_storage_enabled:
            logger.info("Cleaning up storage images")
            try:
                success = self.db.cleanup_storage_images(keep_sizes)
                results["storage_cleanup"] = {"success": success}
            except Exception as e:
                logger.error(f"Error cleaning up storage images: {e}")
                results["storage_cleanup"] = {"success": False, "error": str(e)}
        
        return results
    
    def verify_image_integrity(self, card_names: List[str] = None) -> Dict[str, Any]:
        """Verify that images are accessible and not corrupted"""
        logger.info("Verifying image integrity")
        
        verification_results = {
            "verified": 0,
            "corrupted": 0,
            "missing": 0,
            "details": []
        }
        
        # Get cards to verify
        cards_to_verify = []
        
        if card_names is None:
            # Verify all cards with image data
            if self.db.is_connected:
                try:
                    response = self.db.client.table("mtg_cards").select(
                        "scryfall_id, name, local_image_paths, storage_image_urls"
                    ).execute()
                    cards_to_verify = response.data
                except Exception as e:
                    logger.error(f"Error getting cards from database: {e}")
                    return {"error": str(e)}
        else:
            # Verify specific cards
            for card_name in card_names:
                if self.db.is_connected:
                    try:
                        response = self.db.client.table("mtg_cards").select(
                            "scryfall_id, name, local_image_paths, storage_image_urls"
                        ).eq("name", card_name).limit(1).execute()
                        
                        if response.data:
                            cards_to_verify.extend(response.data)
                    except Exception as e:
                        logger.error(f"Error getting card {card_name}: {e}")
                        verification_results["details"].append({
                            "card_name": card_name,
                            "status": "error",
                            "reason": str(e)
                        })
        
        for card in cards_to_verify:
            card_id = card['scryfall_id']
            card_name = card['name']
            
            # Check local images
            if card.get('local_image_paths'):
                try:
                    local_paths = card['local_image_paths']
                    if isinstance(local_paths, str):
                        local_paths = json.loads(local_paths)
                    
                    for size, path in local_paths.items():
                        try:
                            if os.path.exists(path):
                                file_size = os.path.getsize(path)
                                if file_size > 0:
                                    verification_results["verified"] += 1
                                    verification_results["details"].append({
                                        "card_id": card_id,
                                        "card_name": card_name,
                                        "type": "local",
                                        "size": size,
                                        "status": "verified",
                                        "file_size": file_size
                                    })
                                else:
                                    verification_results["corrupted"] += 1
                                    verification_results["details"].append({
                                        "card_id": card_id,
                                        "card_name": card_name,
                                        "type": "local",
                                        "size": size,
                                        "status": "corrupted",
                                        "reason": "File size is 0"
                                    })
                            else:
                                verification_results["missing"] += 1
                                verification_results["details"].append({
                                    "card_id": card_id,
                                    "card_name": card_name,
                                    "type": "local",
                                    "size": size,
                                    "status": "missing",
                                    "reason": "File not found"
                                })
                        except Exception as e:
                            verification_results["corrupted"] += 1
                            verification_results["details"].append({
                                "card_id": card_id,
                                "card_name": card_name,
                                "type": "local",
                                "size": size,
                                "status": "error",
                                "reason": str(e)
                            })
                except Exception as e:
                    verification_results["corrupted"] += 1
                    verification_results["details"].append({
                        "card_id": card_id,
                        "card_name": card_name,
                        "type": "local",
                        "status": "error",
                        "reason": str(e)
                    })
        
        logger.info(f"Verification completed: {verification_results['verified']} verified, "
                   f"{verification_results['corrupted']} corrupted, "
                   f"{verification_results['missing']} missing")
        
        return verification_results
    
    def migrate_images_to_storage(self, batch_size: int = 100) -> Dict[str, Any]:
        """Migrate all local images to cloud storage"""
        if not self.db.image_storage_enabled:
            return {"error": "Image storage not enabled"}
        
        logger.info("Starting migration of all local images to storage")
        
        if not self.db.is_connected:
            return {"error": "Database not connected"}
        
        try:
            response = self.db.client.table("mtg_cards").select(
                "scryfall_id, name, local_image_paths"
            ).not_.is_("local_image_paths", "null").execute()
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}
        
        total_cards = len(response.data)
        logger.info(f"Found {total_cards} cards with local images")
        
        migration_results = {
            "total_cards": total_cards,
            "uploaded": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
        
        # Process in batches
        for i in range(0, total_cards, batch_size):
            batch = response.data[i:i + batch_size]
            card_names = [card['name'] for card in batch]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_cards + batch_size - 1)//batch_size}")
            
            try:
                # Upload batch
                result = self.upload_to_storage(card_names, force_reupload=False)
                
                migration_results["uploaded"] += result.get("uploaded", 0)
                migration_results["failed"] += result.get("failed", 0)
                migration_results["skipped"] += result.get("skipped", 0)
                migration_results["details"].extend(result.get("details", []))
                
                # Small delay between batches
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                migration_results["failed"] += len(batch)
        
        logger.info(f"Migration completed: {migration_results['uploaded']} uploaded, "
                   f"{migration_results['failed']} failed, {migration_results['skipped']} skipped")
        
        return migration_results