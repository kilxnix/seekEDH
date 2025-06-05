#!/usr/bin/env python3
"""
MTG Image Management CLI

This script provides command-line interface for managing MTG card images,
including downloading, uploading to storage, and maintenance operations.

Usage:
    python manage_images.py --help
    python manage_images.py download --cards "Lightning Bolt,Sol Ring" --sizes normal,large
    python manage_images.py upload-all --force
    python manage_images.py stats
    python manage_images.py cleanup --local --keep normal
    python manage_images.py verify --cards "Lightning Bolt"
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try to import our modules
try:
    from src.image_manager import MTGImageManager
    from src.db_interface import DatabaseInterface
    from src.data_pipeline import MTGDataPipeline
    IMAGE_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ImageCLI")

class ImageCLI:
    """Command-line interface for image management"""
    
    def __init__(self):
        """Initialize the CLI"""
        self.db = DatabaseInterface()
        self.pipeline = MTGDataPipeline()
        self.image_manager = MTGImageManager(self.db, self.pipeline)
        
        if not self.db.is_connected:
            logger.warning("Database not connected. Some features may not work.")
    
    def download_images(self, card_names: List[str], sizes: List[str]) -> None:
        """Download images for specified cards"""
        logger.info(f"Starting download for {len(card_names)} cards, sizes: {sizes}")
        
        result = self.image_manager.download_missing_images(card_names, sizes)
        
        print(f"\n=== Download Results ===")
        print(f"Downloaded: {result['downloaded']}")
        print(f"Failed: {result['failed']}")
        
        if result['failed'] > 0 and result['failures']:
            print(f"\nFailed downloads:")
            for failure in result['failures'][:10]:  # Show first 10 failures
                print(f"  - {failure.get('card_id', 'Unknown')}: {failure.get('error', 'Unknown error')}")
            
            if len(result['failures']) > 10:
                print(f"  ... and {len(result['failures']) - 10} more")
    
    def upload_to_storage(self, card_names: Optional[List[str]] = None, 
                         sizes: List[str] = None, force: bool = False) -> None:
        """Upload images to cloud storage"""
        if sizes is None:
            sizes = ['normal']
        
        if card_names:
            logger.info(f"Uploading images for {len(card_names)} cards to storage")
        else:
            logger.info("Uploading all available images to storage")
        
        result = self.image_manager.upload_to_storage(card_names, sizes, force)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\n=== Upload Results ===")
        print(f"Uploaded: {result['uploaded']}")
        print(f"Failed: {result['failed']}")
        
        if result['failed'] > 0 and result['failures']:
            print(f"\nFailed uploads:")
            for failure in result['failures'][:10]:  # Show first 10 failures
                print(f"  - {failure.get('card_id', 'Unknown')}: {failure.get('error', 'Unknown error')}")
    
    def show_statistics(self) -> None:
        """Show comprehensive image statistics"""
        stats = self.image_manager.get_image_statistics()
        
        print(f"\n=== Image Statistics ===")
        print(f"Timestamp: {stats['timestamp']}")
        
        # Pipeline stats
        if stats.get('pipeline_stats'):
            pipeline = stats['pipeline_stats']
            print(f"\nLocal Storage:")
            print(f"  - Total images: {pipeline.get('local_images_count', 0)}")
            print(f"  - Total size: {pipeline.get('total_size_mb', 0):.2f} MB")
            print(f"  - Manifest exists: {pipeline.get('manifest_exists', False)}")
        
        # Database stats
        if stats.get('database_stats'):
            db = stats['database_stats']
            print(f"\nDatabase:")
            print(f"  - Storage enabled: {db.get('storage_enabled', False)}")
            print(f"  - Cards with storage URLs: {db.get('cards_with_storage_urls', 0)}")
            print(f"  - Cards with local paths: {db.get('cards_with_local_paths', 0)}")
        
        # Local image breakdown by size
        if stats.get('local_images'):
            print(f"\nLocal Images by Size:")
            for size, info in stats['local_images'].items():
                print(f"  - {size}: {info['count']} images ({info['size_mb']:.2f} MB)")
    
    def cleanup_images(self, cleanup_local: bool = False, cleanup_storage: bool = False,
                      keep_sizes: List[str] = None) -> None:
        """Clean up images"""
        if keep_sizes is None:
            keep_sizes = ['normal']
        
        logger.info(f"Cleaning up images, keeping sizes: {keep_sizes}")
        
        result = self.image_manager.cleanup_images(cleanup_local, cleanup_storage, keep_sizes)
        
        print(f"\n=== Cleanup Results ===")
        
        if cleanup_local:
            local = result['local_cleanup']
            print(f"Local cleanup:")
            print(f"  - Removed: {local.get('removed_count', 0)} files")
            print(f"  - Freed: {local.get('removed_size_mb', 0):.2f} MB")
        
        if cleanup_storage:
            storage = result['storage_cleanup']
            print(f"Storage cleanup:")
            print(f"  - Success: {storage.get('success', False)}")
    
    def verify_integrity(self, card_names: Optional[List[str]] = None) -> None:
        """Verify image integrity"""
        if card_names:
            logger.info(f"Verifying integrity for {len(card_names)} cards")
        else:
            logger.info("Verifying integrity for all cards")
        
        result = self.image_manager.verify_image_integrity(card_names)
        
        print(f"\n=== Verification Results ===")
        print(f"Verified: {result['verified']}")
        print(f"Corrupted: {result['corrupted']}")
        print(f"Missing: {result['missing']}")
        
        if result['corrupted'] > 0:
            print(f"\nCorrupted files:")
            corrupted = [d for d in result['details'] if d['status'] == 'corrupted']
            for item in corrupted[:10]:  # Show first 10
                print(f"  - {item['card_name']}: {item.get('reason', 'Unknown')}")
        
        if result['missing'] > 0:
            print(f"\nMissing files:")
            missing = [d for d in result['details'] if d['status'] == 'missing']
            for item in missing[:10]:  # Show first 10
                print(f"  - {item['card_name']}: {item.get('reason', 'Unknown')}")
    
    def migrate_to_storage(self, batch_size: int = 100) -> None:
        """Migrate all local images to storage"""
        logger.info("Starting migration of all local images to storage")
        
        result = self.image_manager.migrate_images_to_storage(batch_size)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\n=== Migration Results ===")
        print(f"Total cards: {result['total_cards']}")
        print(f"Uploaded: {result['uploaded']}")
        print(f"Failed: {result['failed']}")
        print(f"Skipped: {result['skipped']}")
    
    def get_card_info(self, card_name: str, size: str = 'normal') -> None:
        """Get detailed information about a card's images"""
        image_info = self.image_manager.get_card_image_info(card_name, size)
        
        if not image_info:
            print(f"Card '{card_name}' not found")
            return
        
        print(f"\n=== Image Info for '{card_name}' ===")
        print(f"Card ID: {image_info.card_id}")
        print(f"Size: {image_info.size}")
        print(f"Status: {image_info.status}")
        
        if image_info.local_path:
            exists = os.path.exists(image_info.local_path)
            print(f"Local path: {image_info.local_path} {'(exists)' if exists else '(missing)'}")
            if exists and image_info.file_size:
                print(f"File size: {image_info.file_size:,} bytes")
        
        if image_info.storage_url:
            print(f"Storage URL: {image_info.storage_url}")
        
        if image_info.original_url:
            print(f"Original URL: {image_info.original_url}")
        
        if image_info.error:
            print(f"Error: {image_info.error}")
    
    def list_available_sizes(self) -> None:
        """List available image sizes"""
        sizes = self.pipeline.image_sizes
        print(f"\nAvailable image sizes: {', '.join(sizes)}")
        print(f"Preferred size order: {', '.join(self.pipeline.preferred_sizes)}")
    
    def download_random_sample(self, count: int = 10, sizes: List[str] = None) -> None:
        """Download images for a random sample of cards"""
        if sizes is None:
            sizes = ['normal']
        
        logger.info(f"Downloading images for {count} random cards")
        
        if not self.db.is_connected:
            print("Error: Database not connected")
            return
        
        # Get random cards
        response = self.db.client.table("mtg_cards").select("name").limit(count * 2).execute()
        
        if not response.data:
            print("No cards found in database")
            return
        
        import random
        card_names = [card['name'] for card in random.sample(response.data, min(count, len(response.data)))]
        
        print(f"Selected cards: {', '.join(card_names)}")
        
        # Download images
        self.download_images(card_names, sizes)

def parse_list_arg(value: str) -> List[str]:
    """Parse comma-separated list argument"""
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="MTG Image Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stats
  %(prog)s download --cards "Lightning Bolt,Sol Ring" --sizes normal,large
  %(prog)s upload-all --force
  %(prog)s info "Lightning Bolt"
  %(prog)s cleanup --local --keep normal
  %(prog)s verify --cards "Lightning Bolt"
  %(prog)s migrate --batch-size 50
  %(prog)s sample --count 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download images for specified cards')
    download_parser.add_argument('--cards', required=True, 
                               help='Comma-separated list of card names')
    download_parser.add_argument('--sizes', default='normal',
                               help='Comma-separated list of image sizes (default: normal)')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload images to storage')
    upload_parser.add_argument('--cards', 
                              help='Comma-separated list of card names (default: all)')
    upload_parser.add_argument('--sizes', default='normal',
                              help='Comma-separated list of image sizes (default: normal)')
    upload_parser.add_argument('--force', action='store_true',
                              help='Force re-upload even if already exists')
    
    # Upload all command
    upload_all_parser = subparsers.add_parser('upload-all', help='Upload all local images to storage')
    upload_all_parser.add_argument('--force', action='store_true',
                                  help='Force re-upload even if already exists')
    
    # Statistics command
    subparsers.add_parser('stats', help='Show image statistics')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up images')
    cleanup_parser.add_argument('--local', action='store_true',
                               help='Clean up local images')
    cleanup_parser.add_argument('--storage', action='store_true',
                               help='Clean up storage images')
    cleanup_parser.add_argument('--keep', default='normal',
                               help='Comma-separated list of sizes to keep (default: normal)')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify image integrity')
    verify_parser.add_argument('--cards',
                              help='Comma-separated list of card names (default: all)')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate all local images to storage')
    migrate_parser.add_argument('--batch-size', type=int, default=100,
                               help='Batch size for migration (default: 100)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get detailed info about a card\'s images')
    info_parser.add_argument('card_name', help='Name of the card')
    info_parser.add_argument('--size', default='normal',
                            help='Image size to check (default: normal)')
    
    # List sizes command
    subparsers.add_parser('sizes', help='List available image sizes')
    
    # Sample download command
    sample_parser = subparsers.add_parser('sample', help='Download images for random sample of cards')
    sample_parser.add_argument('--count', type=int, default=10,
                              help='Number of random cards (default: 10)')
    sample_parser.add_argument('--sizes', default='normal',
                              help='Comma-separated list of image sizes (default: normal)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    try:
        cli = ImageCLI()
    except Exception as e:
        print(f"Error initializing CLI: {e}")
        return
    
    # Execute commands
    try:
        if args.command == 'download':
            card_names = parse_list_arg(args.cards)
            sizes = parse_list_arg(args.sizes)
            cli.download_images(card_names, sizes)
        
        elif args.command == 'upload':
            card_names = parse_list_arg(args.cards) if args.cards else None
            sizes = parse_list_arg(args.sizes)
            cli.upload_to_storage(card_names, sizes, args.force)
        
        elif args.command == 'upload-all':
            cli.upload_to_storage(None, ['normal'], args.force)
        
        elif args.command == 'stats':
            cli.show_statistics()
        
        elif args.command == 'cleanup':
            keep_sizes = parse_list_arg(args.keep)
            cli.cleanup_images(args.local, args.storage, keep_sizes)
        
        elif args.command == 'verify':
            card_names = parse_list_arg(args.cards) if args.cards else None
            cli.verify_integrity(card_names)
        
        elif args.command == 'migrate':
            cli.migrate_to_storage(args.batch_size)
        
        elif args.command == 'info':
            cli.get_card_info(args.card_name, args.size)
        
        elif args.command == 'sizes':
            cli.list_available_sizes()
        
        elif args.command == 'sample':
            sizes = parse_list_arg(args.sizes)
            cli.download_random_sample(args.count, sizes)
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()