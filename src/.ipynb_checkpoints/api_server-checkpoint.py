import os
import logging
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.config import API_HOST, API_PORT
from src.data_pipeline import MTGDataPipeline

# Import database interface
try:
    from src.db_interface import DatabaseInterface
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logging.warning("Database interface not available")

# Import price embeddings
try:
    from src.price_embeddings import PriceEmbeddingGenerator
    PRICE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    PRICE_EMBEDDINGS_AVAILABLE = False
    logging.warning("Price embeddings not available")

# Initialize logger
logger = logging.getLogger("APIServer")

# Initialize components
data_pipeline = MTGDataPipeline()
db_interface = DatabaseInterface() if DB_AVAILABLE else None
price_embedder = PriceEmbeddingGenerator() if PRICE_EMBEDDINGS_AVAILABLE else None

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    pipeline_status = data_pipeline.get_status()
    db_status = db_interface.get_status() if DB_AVAILABLE else {"available": False}
    price_embeddings_status = price_embedder.get_status() if PRICE_EMBEDDINGS_AVAILABLE else {"available": False}
    
    return jsonify({
        "status": "ok",
        "pipeline": pipeline_status,
        "database": db_status,
        "price_embeddings": price_embeddings_status
    })

@app.route('/api/config/database', methods=['POST'])
def set_database_config():
    """Set database credentials"""
    if not DB_AVAILABLE:
        return jsonify({"error": "Database interface not available"}), 400
    
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    url = data.get('url')
    key = data.get('key')
    
    if not url or not key:
        return jsonify({"error": "URL and key are required"}), 400
    
    success = db_interface.set_credentials(url, key)
    
    if success:
        return jsonify({"success": True, "message": "Database credentials updated"})
    else:
        return jsonify({"success": False, "error": "Failed to connect with provided credentials"}), 400

@app.route('/api/database/initialize', methods=['POST'])
def initialize_database():
    """Initialize database schema"""
    if not DB_AVAILABLE:
        return jsonify({"error": "Database interface not available"}), 400
    
    if not db_interface.is_connected:
        return jsonify({"error": "Not connected to database"}), 400
    
    success = db_interface.initialize_schema()
    
    if success:
        return jsonify({"success": True, "message": "Database schema initialized"})
    else:
        return jsonify({"success": False, "error": "Failed to initialize database schema"}), 500

@app.route('/api/data/update', methods=['POST'])
def update_data():
    """Run the data pipeline to fetch and process card data"""
    data = request.json or {}
    force_update = data.get('force', False)
    skip_embeddings = data.get('skip_embeddings', False)
    
    result = data_pipeline.run_pipeline(force_update, skip_embeddings)
    
    if result.get('success', False):
        return jsonify(result)
    else:
        return jsonify(result), 500

@app.route('/api/data/status', methods=['GET'])
def data_status():
    """Get status of data pipeline"""
    result = data_pipeline.get_status()
    return jsonify(result)

@app.route('/api/price-embeddings/generate', methods=['POST'])
def generate_price_embeddings():
    """Generate price embeddings"""
    if not PRICE_EMBEDDINGS_AVAILABLE:
        return jsonify({"error": "Price embeddings not available"}), 400
    
    embeddings = price_embedder.generate_price_embeddings()
    
    if embeddings is not None:
        return jsonify({
            "success": True, 
            "message": "Price embeddings generated successfully",
            "shape": embeddings.shape
        })
    else:
        return jsonify({
            "success": False, 
            "error": "Failed to generate price embeddings"
        }), 500

@app.route('/api/price-embeddings/similar', methods=['GET'])
def get_similar_cards():
    """Get cards with similar price characteristics"""
    if not PRICE_EMBEDDINGS_AVAILABLE:
        return jsonify({"error": "Price embeddings not available"}), 400
    
    card_name = request.args.get('card')
    top_n = int(request.args.get('top_n', 10))
    
    if not card_name:
        return jsonify({"error": "Card name is required"}), 400
    
    similar_cards = price_embedder.get_similar_cards_by_price(card_name, top_n)
    
    if similar_cards:
        return jsonify({
            "success": True,
            "card": card_name,
            "similar_cards": similar_cards
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Could not find similar cards to '{card_name}'"
        }), 404

@app.route('/api/database/import', methods=['POST'])
def import_to_database():
    """Import processed data to database"""
    if not DB_AVAILABLE:
        return jsonify({"error": "Database interface not available"}), 400
    
    if not db_interface.is_connected:
        return jsonify({"error": "Not connected to database"}), 400
    
    data = request.json or {}
    include_price_embeddings = data.get('include_price_embeddings', False)
    
    result = db_interface.import_cards(include_price_embeddings=include_price_embeddings)
    
    if result:
        return jsonify({"success": True, "message": "Data imported successfully"})
    else:
        return jsonify({"success": False, "error": "Failed to import data"}), 500

@app.route('/api/database/status', methods=['GET'])
def database_status():
    """Get database status"""
    if not DB_AVAILABLE:
        return jsonify({"available": False})
    
    result = db_interface.get_status()
    return jsonify(result)

@app.route('/api/price-embeddings/status', methods=['GET'])
def price_embeddings_status():
    """Get price embeddings status"""
    if not PRICE_EMBEDDINGS_AVAILABLE:
        return jsonify({"available": False})
    
    result = price_embedder.get_status()
    return jsonify(result)

def run_server():
    """Run the API server"""
    app.run(host=API_HOST, port=API_PORT)

if __name__ == "__main__":
    run_server()