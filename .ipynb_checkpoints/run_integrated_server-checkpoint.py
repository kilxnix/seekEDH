# run_integrated_server.py
import os
import sys
import logging
import argparse
from flask import send_from_directory

# Ensure the src directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import required modules
from src.api_server import app
import src.update_api_server  # This adds the RAG endpoints to the app

# Import our new deck endpoint
import api_deck_endpoint  # This adds the deck generation endpoints to the app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegratedServer")

@app.route('/mtg-ai-framework/static/<path:filename>')
def serve_static(filename):
    # Ensure you're using the correct absolute path to the static folder
    static_folder = os.path.join(current_dir, 'static')
    return send_from_directory(static_folder, filename)

def main():
    """Run the integrated server with both RAG and deck generation capabilities"""
    parser = argparse.ArgumentParser(description="Integrated MTG AI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Log startup information
    logger.info("Starting integrated MTG AI server")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Ensure directories exist
    os.makedirs('data/generated_decks', exist_ok=True)
    
    # Log available endpoints
    endpoints = []
    for rule in app.url_map.iter_rules():
        endpoints.append(f"{rule.rule} [{', '.join(rule.methods)}]")
    
    logger.info("Available endpoints:")
    for endpoint in sorted(endpoints):
        logger.info(f"  {endpoint}")
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()