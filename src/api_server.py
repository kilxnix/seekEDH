# src/api_server.py - Complete API Server with Image Support
import os
import logging
import json
from flask import Flask, request, jsonify, send_file, redirect
from io import StringIO
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

# Import image manager
try:
    from src.image_manager import MTGImageManager
    IMAGE_MANAGER_AVAILABLE = True
except ImportError:
    IMAGE_MANAGER_AVAILABLE = False
    logging.warning("Image manager not available")

# Import enhanced RAG system
try:
    from src.enhanced_rag_system import EnhancedMTGRetrievalSystem
    from src.rag_system import MTGRetrievalSystem
    ENHANCED_RAG_AVAILABLE = True
except ImportError:
    ENHANCED_RAG_AVAILABLE = False
    logging.warning("Enhanced RAG system not available")

# Import image embeddings system
try:
    from src.image_embeddings_system import MTGImageEmbeddingGenerator
    IMAGE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    IMAGE_EMBEDDINGS_AVAILABLE = False
    logging.warning("Image embeddings system not available")

# Initialize logger
logger = logging.getLogger("APIServer")

# Initialize components
data_pipeline = MTGDataPipeline()
db_interface = DatabaseInterface() if DB_AVAILABLE else None
price_embedder = PriceEmbeddingGenerator() if PRICE_EMBEDDINGS_AVAILABLE else None
image_manager = MTGImageManager(db_interface, data_pipeline) if IMAGE_MANAGER_AVAILABLE else None

# Initialize enhanced RAG system
enhanced_rag = None
image_embeddings_generator = None
universal_search_handler = None

def get_enhanced_rag():
    """Get or create enhanced RAG instance"""
    global enhanced_rag
    if enhanced_rag is None and ENHANCED_RAG_AVAILABLE and DB_AVAILABLE:
        try:
            rag_system = MTGRetrievalSystem(db_interface)
            enhanced_rag = EnhancedMTGRetrievalSystem(rag_system, image_manager)
            logger.info("Enhanced RAG system initialized")
        except Exception as e:
            logger.error(f"Error initializing enhanced RAG system: {e}")
    return enhanced_rag

def get_image_embeddings_generator():
    """Get or create image embeddings generator instance"""
    global image_embeddings_generator
    if image_embeddings_generator is None and IMAGE_EMBEDDINGS_AVAILABLE:
        try:
            image_embeddings_generator = MTGImageEmbeddingGenerator(db_interface, data_pipeline)
            logger.info("Image embeddings generator initialized")
        except Exception as e:
            logger.error(f"Error initializing image embeddings generator: {e}")
    return image_embeddings_generator

def get_universal_search_handler():
    """Get or create the universal search handler"""
    global universal_search_handler
    rag_system = get_enhanced_rag()
    if rag_system is None:
        return None
    if universal_search_handler is None:
        try:
            from src.enhanced_universal_search import EnhancedUniversalSearchHandler
            universal_search_handler = EnhancedUniversalSearchHandler(rag_system.rag)
            logger.info("Universal search handler initialized")
        except Exception as e:
            logger.error(f"Error initializing universal search handler: {e}")
            universal_search_handler = None
    return universal_search_handler

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    pipeline_status = data_pipeline.get_status()
    db_status = db_interface.get_status() if DB_AVAILABLE else {"available": False}
    price_embeddings_status = price_embedder.get_status() if PRICE_EMBEDDINGS_AVAILABLE else {"available": False}
    image_status = {"available": IMAGE_MANAGER_AVAILABLE}
    
    if IMAGE_MANAGER_AVAILABLE and image_manager:
        try:
            image_stats = image_manager.get_image_statistics()
            image_status["stats"] = image_stats
        except Exception as e:
            image_status["error"] = str(e)
    
    enhanced_rag_status = {"available": ENHANCED_RAG_AVAILABLE}
    if ENHANCED_RAG_AVAILABLE:
        try:
            rag = get_enhanced_rag()
            enhanced_rag_status["initialized"] = rag is not None
        except Exception as e:
            enhanced_rag_status["error"] = str(e)
    
    image_embeddings_status = {"available": IMAGE_EMBEDDINGS_AVAILABLE}
    if IMAGE_EMBEDDINGS_AVAILABLE:
        try:
            generator = get_image_embeddings_generator()
            if generator:
                image_embeddings_status.update(generator.get_status())
        except Exception as e:
            image_embeddings_status["error"] = str(e)
    
    return jsonify({
        "status": "ok",
        "pipeline": pipeline_status,
        "database": db_status,
        "price_embeddings": price_embeddings_status,
        "image_manager": image_status,
        "enhanced_rag": enhanced_rag_status,
        "image_embeddings": image_embeddings_status
    })

# ========== DATABASE ENDPOINTS ==========

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

@app.route('/api/database/import', methods=['POST'])
def import_to_database():
    """Import processed data to database with image support"""
    if not DB_AVAILABLE:
        return jsonify({"error": "Database interface not available"}), 400
    
    if not db_interface.is_connected:
        return jsonify({"error": "Not connected to database"}), 400
    
    data = request.json or {}
    include_price_embeddings = data.get('include_price_embeddings', False)
    upload_images_to_storage = data.get('upload_images_to_storage', True)
    verbose = data.get('verbose', False)

    log_stream = StringIO()
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)

    try:
        if upload_images_to_storage:
            result = db_interface.import_cards_with_images(
                include_price_embeddings=include_price_embeddings,
                upload_to_storage=True
            )
        else:
            result = db_interface.import_cards(include_price_embeddings=include_price_embeddings)
    finally:
        root_logger.removeHandler(stream_handler)
        stream_handler.flush()
    logs = log_stream.getvalue().splitlines()
    log_stream.close()

    response_data = {"success": bool(result)}
    if result:
        response_data["message"] = "Data imported successfully"
    else:
        response_data["error"] = "Failed to import data"
    if verbose:
        response_data["logs"] = logs

    if result:
        return jsonify(response_data)
    else:
        return jsonify(response_data), 500

@app.route('/api/database/status', methods=['GET'])
def database_status():
    """Get database status"""
    if not DB_AVAILABLE:
        return jsonify({"available": False})
    
    result = db_interface.get_status()
    return jsonify(result)

# ========== DATA PIPELINE ENDPOINTS ==========

@app.route('/api/data/update', methods=['POST'])
def update_data():
    """Run the data pipeline to fetch and process card data with image support"""
    data = request.json or {}
    force_update = data.get('force', False)
    skip_embeddings = data.get('skip_embeddings', False)
    download_images = data.get('download_images', True)
    
    result = data_pipeline.run_pipeline(force_update, skip_embeddings, download_images)
    
    if result.get('success', False):
        return jsonify(result)
    else:
        return jsonify(result), 500

@app.route('/api/data/status', methods=['GET'])
def data_status():
    """Get status of data pipeline"""
    result = data_pipeline.get_status()
    return jsonify(result)

# ========== PRICE EMBEDDINGS ENDPOINTS ==========

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

@app.route('/api/price-embeddings/status', methods=['GET'])
def price_embeddings_status():
    """Get price embeddings status"""
    if not PRICE_EMBEDDINGS_AVAILABLE:
        return jsonify({"available": False})
    
    result = price_embedder.get_status()
    return jsonify(result)

# ========== IMAGE MANAGEMENT ENDPOINTS ==========

@app.route('/api/images/card/<card_name>', methods=['GET'])
def get_card_image_info(card_name):
    """Get image information for a specific card"""
    if not IMAGE_MANAGER_AVAILABLE:
        return jsonify({"error": "Image manager not available"}), 400
    
    size = request.args.get('size', 'normal')
    
    try:
        image_info = image_manager.get_card_image_info(card_name, size)
        
        if not image_info:
            return jsonify({"error": f"Card '{card_name}' not found"}), 404
        
        return jsonify({
            "card_id": image_info.card_id,
            "size": image_info.size,
            "status": image_info.status,
            "local_path": image_info.local_path,
            "storage_url": image_info.storage_url,
            "original_url": image_info.original_url,
            "file_size": image_info.file_size,
            "error": image_info.error
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/images/card/<card_name>/url', methods=['GET'])
def get_card_image_url(card_name):
    """Get the best available image URL for a card"""
    if not IMAGE_MANAGER_AVAILABLE:
        return jsonify({"error": "Image manager not available"}), 400
    
    size = request.args.get('size', 'normal')
    prefer_storage = request.args.get('prefer_storage', 'true').lower() == 'true'
    
    try:
        image_url = image_manager.get_best_image_url(card_name, size, prefer_storage)
        
        if image_url:
            return jsonify({
                "card_name": card_name,
                "size": size,
                "image_url": image_url,
                "success": True
            })
        else:
            return jsonify({
                "card_name": card_name,
                "size": size,
                "image_url": None,
                "success": False,
                "message": "No image available"
            }), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/images/serve/<card_name>', methods=['GET'])
def serve_card_image(card_name):
    """Serve a card image file directly"""
    if not IMAGE_MANAGER_AVAILABLE:
        return jsonify({"error": "Image manager not available"}), 400
    
    size = request.args.get('size', 'normal')
    
    try:
        image_info = image_manager.get_card_image_info(card_name, size)
        
        if not image_info:
            return jsonify({"error": f"Card '{card_name}' not found"}), 404
        
        # Serve local file if available
        if image_info.local_path and os.path.exists(image_info.local_path):
            return send_file(image_info.local_path)
        
        # Redirect to storage URL if available
        elif image_info.storage_url:
            return redirect(image_info.storage_url)
        
        # Redirect to original Scryfall URL as fallback
        elif image_info.original_url:
            return redirect(image_info.original_url)
        
        else:
            return jsonify({"error": "No image available"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/images/statistics', methods=['GET'])
def get_image_statistics():
    """Get comprehensive image statistics"""
    if not IMAGE_MANAGER_AVAILABLE:
        return jsonify({"error": "Image manager not available"}), 400
    
    try:
        stats = image_manager.get_image_statistics()
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/images/upload-to-storage', methods=['POST'])
def upload_images_to_storage():
    """Upload local images to cloud storage"""
    if not IMAGE_MANAGER_AVAILABLE:
        return jsonify({"error": "Image manager not available"}), 400
    
    data = request.json or {}
    card_names = data.get('card_names')
    sizes = data.get('sizes', ['normal'])
    force_reupload = data.get('force_reupload', False)
    
    try:
        result = image_manager.upload_to_storage(card_names, sizes, force_reupload)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify({
            "success": True,
            "uploaded": result["uploaded"],
            "failed": result["failed"],
            "details": result["details"],
            "failures": result["failures"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== ENHANCED RAG ENDPOINTS ==========

@app.route('/api/rag/enhanced-search', methods=['POST'])
def enhanced_search():
    """Enhanced search with images"""
    try:
        data = request.json or {}
        query = data.get('query', '')
        filters = data.get('filters', {})
        top_k = min(data.get('top_k', 20), 50)
        image_size = data.get('image_size', 'normal')
        include_images = data.get('include_images', True)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        enhanced_rag_system = get_enhanced_rag()
        if not enhanced_rag_system:
            return jsonify({"error": "Enhanced RAG system not available"}), 400

        universal_handler = get_universal_search_handler()
        if not universal_handler:
            return jsonify({"error": "Universal search handler not available"}), 500

        # Use the universal search handler to process the query with optional context
        result = universal_handler.process_universal_query(query, filters)

        # Enhance returned cards with images if requested
        if result.get('success') and include_images and 'cards' in result:
            result['cards'] = enhanced_rag_system.enhance_cards_with_images(
                result['cards'], image_size
            )
            result['metadata'] = {
                "image_size": image_size,
                "includes_images": include_images
            }

        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/rag/card-synergies', methods=['POST'])
def enhanced_card_synergies():
    """Find card synergies with images"""
    try:
        data = request.json or {}
        seed_cards = data.get('seed_cards', [])
        top_k = min(data.get('top_k', 15), 30)
        image_size = data.get('image_size', 'normal')
        include_images = data.get('include_images', True)
        
        if not seed_cards:
            return jsonify({"error": "Seed cards are required"}), 400
        
        # Import synergy calculation
        try:
            from src.update_api_server import calculate_card_synergies
            synergy_results = calculate_card_synergies(seed_cards, top_k)
        except ImportError:
            return jsonify({"error": "Synergy calculation not available"}), 400
        
        # Extract cards and enhance with images if requested
        synergistic_cards = []
        for result in synergy_results:
            card = result.get('card', {})
            
            # Add synergy metadata
            card['synergy_score'] = result.get('total_score', 0)
            card['synergy_type'] = 'calculated'
            card['synergy_reason'] = 'Based on card mechanics and rules interactions'
            
            synergistic_cards.append(card)
        
        # Enhance with images
        if include_images:
            enhanced_rag_system = get_enhanced_rag()
            if enhanced_rag_system:
                synergistic_cards = enhanced_rag_system.enhance_cards_with_images(
                    synergistic_cards, image_size
                )
        
        return jsonify({
            "success": True,
            "seed_cards": seed_cards,
            "synergistic_cards": synergistic_cards,
            "metadata": {
                "image_size": image_size,
                "includes_images": include_images,
                "total_candidates_analyzed": len(synergy_results)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in card synergies: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/rag/universal-search', methods=['POST'])
def enhanced_universal_search():
    """Universal search with images"""
    try:
        data = request.json or {}
        query = data.get('query', '')
        context = data.get('context', {})
        image_size = data.get('image_size', 'normal')
        include_images = data.get('include_images', True)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Import universal search handler
        try:
            from src.enhanced_universal_search import EnhancedUniversalSearchHandler
        except ImportError:
            return jsonify({"error": "Universal search handler not available"}), 400
        
        enhanced_rag_system = get_enhanced_rag()
        if not enhanced_rag_system:
            return jsonify({"error": "Enhanced RAG system not available"}), 400
        
        universal_handler = EnhancedUniversalSearchHandler(enhanced_rag_system.rag)
        
        # Process query
        result = universal_handler.process_universal_query(query, context)
        
        # Enhance results with images if successful and images requested
        if result.get('success') and include_images and 'cards' in result:
            result['cards'] = enhanced_rag_system.enhance_cards_with_images(
                result['cards'], image_size
            )
            result['metadata'] = {
                "image_size": image_size,
                "includes_images": include_images
            }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in universal search: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/search-card', methods=['GET'])
def enhanced_search_card():
    """Search for a specific card with image"""
    try:
        card_name = request.args.get('name', '').strip()
        image_size = request.args.get('image_size', 'normal')
        include_images = request.args.get('include_images', 'true').lower() == 'true'
        
        if not card_name:
            return jsonify({"error": "Card name is required"}), 400
        
        enhanced_rag_system = get_enhanced_rag()
        
        if enhanced_rag_system:
            # Use enhanced system
            card = enhanced_rag_system.get_card_with_image(card_name, image_size)
            
            if card:
                return jsonify({
                    "success": True,
                    "query": card_name,
                    "exact_match": card,
                    "metadata": {
                        "image_size": image_size,
                        "includes_images": include_images
                    }
                })
            else:
                # Try similarity search
                similar_matches = enhanced_rag_system.retrieve_cards_by_text_with_images(
                    card_name, top_k=5, image_size=image_size
                )
                
                return jsonify({
                    "success": True,
                    "query": card_name,
                    "similar_matches": similar_matches,
                    "metadata": {
                        "image_size": image_size,
                        "includes_images": include_images
                    }
                })
        else:
            return jsonify({"error": "Enhanced RAG system not available"}), 400
        
    except Exception as e:
        logger.error(f"Error searching for card: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ========== IMAGE EMBEDDINGS ENDPOINTS ==========

@app.route('/api/image-embeddings/status', methods=['GET'])
def image_embeddings_status():
    """Get status of image embeddings system"""
    try:
        generator = get_image_embeddings_generator()
        if not generator:
            return jsonify({
                "available": False,
                "error": "Image embeddings system not available"
            }), 400
        
        status = generator.get_status()
        return jsonify({
            "available": True,
            **status
        })
        
    except Exception as e:
        logger.error(f"Error getting image embeddings status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/image-embeddings/generate', methods=['POST'])
def generate_image_embeddings():
    """Generate image embeddings for cards"""
    try:
        generator = get_image_embeddings_generator()
        if not generator:
            return jsonify({"error": "Image embeddings system not available"}), 400
        
        data = request.json or {}
        card_names = data.get('card_names')
        force_regenerate = data.get('force_regenerate', False)
        
        logger.info(f"Starting image embedding generation for {len(card_names) if card_names else 'all'} cards")
        
        result = generator.generate_embeddings_for_cards(card_names, force_regenerate)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating image embeddings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/image-embeddings/visual-similarity', methods=['GET'])
def find_visually_similar_cards():
    """Find cards with similar artwork"""
    try:
        generator = get_image_embeddings_generator()
        if not generator:
            return jsonify({"error": "Image embeddings system not available"}), 400
        
        card_name = request.args.get('card')
        top_k = min(int(request.args.get('top_k', 10)), 50)
        include_images = request.args.get('include_images', 'true').lower() == 'true'
        
        if not card_name:
            return jsonify({"error": "Card name is required"}), 400
        
        similar_cards = generator.find_visually_similar_cards(card_name, top_k)
        
        # Enhance with images if requested
        if include_images and similar_cards:
            try:
                enhanced_rag_system = get_enhanced_rag()
                if enhanced_rag_system:
                    similar_cards = enhanced_rag_system.enhance_cards_with_images(similar_cards)
            except:
                pass
        
        return jsonify({
            "success": True,
            "query_card": card_name,
            "visually_similar_cards": similar_cards,
            "total_found": len(similar_cards),
            "search_type": "visual_similarity"
        })
        
    except Exception as e:
        logger.error(f"Error finding visually similar cards: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/image-embeddings/search-by-description', methods=['POST'])
def search_by_image_description():
    """Search cards by describing their visual appearance"""
    try:
        generator = get_image_embeddings_generator()
        if not generator:
            return jsonify({"error": "Image embeddings system not available"}), 400
        
        data = request.json or {}
        description = data.get('description', '').strip()
        top_k = min(data.get('top_k', 10), 50)
        include_images = data.get('include_images', True)
        
        if not description:
            return jsonify({"error": "Description is required"}), 400
        
        matching_cards = generator.search_by_image_description(description, top_k)
        
        # Enhance with images if requested
        if include_images and matching_cards:
            try:
                enhanced_rag_system = get_enhanced_rag()
                if enhanced_rag_system:
                    matching_cards = enhanced_rag_system.enhance_cards_with_images(matching_cards)
            except:
                pass
        
        return jsonify({
            "success": True,
            "description": description,
            "matching_cards": matching_cards,
            "total_found": len(matching_cards),
            "search_type": "text_to_image"
        })
        
    except Exception as e:
        logger.error(f"Error searching by image description: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def run_server():
    """Run the API server"""
    app.run(host=API_HOST, port=API_PORT)

if __name__ == "__main__":
    run_server()