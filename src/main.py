import os
import sys
import argparse
import logging

# Ensure the src directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.data_pipeline import MTGDataPipeline
from src.db_interface import DatabaseInterface

# Import price embeddings
try:
    from src.price_embeddings import PriceEmbeddingGenerator
    PRICE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    PRICE_EMBEDDINGS_AVAILABLE = False
    logging.warning("Price embeddings not available")

# Try importing the API app, but handle if it fails
try:
    from src.api_server import app as api_app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    logging.warning("API server not available")

# Try importing config, but provide defaults if it fails
try:
    from src.config import API_HOST, API_PORT
except ImportError:
    API_HOST = "0.0.0.0"
    API_PORT = 5000

logger = logging.getLogger("Main")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MTG AI Data Pipeline and API")
    parser.add_argument("--data", action="store_true", help="Run data pipeline")
    parser.add_argument("--force", action="store_true", help="Force update data even if recent")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip text embedding generation")
    parser.add_argument("--price-embeddings", action="store_true", help="Generate price embeddings")
    parser.add_argument("--import", dest="import_data", action="store_true", help="Import data to database")
    parser.add_argument("--server", action="store_true", help="Run API server")
    parser.add_argument("--host", type=str, default=API_HOST, help="API server host")
    parser.add_argument("--port", type=int, default=API_PORT, help="API server port")
    parser.add_argument("--similar-to", type=str, help="Find cards with similar price to the specified card")
    parser.add_argument("--top-n", type=int, default=10, help="Number of similar cards to return")
    
    args = parser.parse_args()
    
    # If no args provided, run the server
    if not (args.data or args.import_data or args.server or args.price_embeddings or args.similar_to):
        args.server = True
    
    # Run data pipeline if requested
    if args.data:
        logger.info("Running data pipeline")
        pipeline = MTGDataPipeline()
        result = pipeline.run_pipeline(args.force, args.skip_embeddings)
        
        if isinstance(result, dict) and result.get('success', False):
            logger.info("Data pipeline completed successfully")
        else:
            logger.error(f"Data pipeline failed: {result.get('error', 'Unknown error') if isinstance(result, dict) else 'Unknown error'}")
            if not args.server and not args.price_embeddings and not args.similar_to:
                return
    
    # Generate price embeddings if requested
    if args.price_embeddings:
        if not PRICE_EMBEDDINGS_AVAILABLE:
            logger.error("Price embeddings not available")
            if not args.server and not args.import_data and not args.similar_to:
                return
        else:
            logger.info("Generating price embeddings")
            price_embedder = PriceEmbeddingGenerator()
            embeddings = price_embedder.generate_price_embeddings()
            
            if embeddings is not None:
                logger.info("Price embeddings generated successfully")
            else:
                logger.error("Failed to generate price embeddings")
                if not args.server and not args.import_data and not args.similar_to:
                    return
    
    # Find similar cards if requested
    if args.similar_to:
        if not PRICE_EMBEDDINGS_AVAILABLE:
            logger.error("Price embeddings not available")
            if not args.server and not args.import_data:
                return
        else:
            logger.info(f"Finding cards with similar price to '{args.similar_to}'")
            price_embedder = PriceEmbeddingGenerator()
            similar_cards = price_embedder.get_similar_cards_by_price(args.similar_to, args.top_n)
            
            if similar_cards:
                print(f"\nCards with similar price characteristics to '{args.similar_to}':")
                for i, card in enumerate(similar_cards):
                    print(f"{i+1}. {card['name']} - ${card['price_usd']} (Normal), ${card['price_usd_foil']} (Foil) - {card['rarity']} from {card['set']}")
            else:
                logger.error(f"Could not find similar cards to '{args.similar_to}'")
                if not args.server and not args.import_data:
                    return
    
    # Import data to database if requested
    if args.import_data:
        logger.info("Importing data to database")
        db = DatabaseInterface()
        
        if not db.is_connected:
            logger.error("Cannot import data: Not connected to database")
            if not args.server:
                return
        else:
            result = db.import_cards()
            
            if result:
                logger.info("Data imported successfully")
            else:
                logger.error("Failed to import data")
                if not args.server:
                    return
    
    # Run the API server if requested
    if args.server:
        if not API_AVAILABLE:
            logger.error("API server not available")
            return
            
        logger.info(f"Starting API server on {args.host}:{args.port}")
        api_app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()