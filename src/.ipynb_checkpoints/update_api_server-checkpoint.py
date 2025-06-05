from flask import jsonify, request
from flask_cors import CORS
from src.rag_system import MTGRetrievalSystem
from src.api_server import app
import logging
from functools import lru_cache

# Configure logging
logger = logging.getLogger("UpdateAPIServer")

# Initialize the RAG system with embedding model
try:
    # Initialize with a small, efficient model
    rag_system = MTGRetrievalSystem(embedding_model_name="all-MiniLM-L6-v2")
    logger.info("RAG system initialized with embedding model")
except Exception as e:
    logger.error(f"Error initializing RAG system with embedding model: {e}")
    rag_system = MTGRetrievalSystem()
    logger.info("RAG system initialized without embedding model")

# Enable CORS
CORS(app)

# Helper functions
def validate_colors(colors_str):
    """Validate and normalize color parameters"""
    if not colors_str:
        return None
        
    valid_colors = ['W', 'U', 'B', 'R', 'G', 'C']  # Added 'C' for colorless
    colors = colors_str.split(',')
    return [c.upper() for c in colors if c.upper() in valid_colors]

def validate_format(format_str):
    """Validate format parameter"""
    if not format_str:
        return None
        
    valid_formats = ['standard', 'pioneer', 'modern', 'legacy', 'vintage', 'commander', 'brawl', 'historic', 'pauper']
    return format_str.lower() if format_str.lower() in valid_formats else None

def check_color_match(card_colors, filter_colors):
    """Check if a card's colors match the filter colors"""
    if 'C' in filter_colors and (not card_colors or len(card_colors) == 0):
        return True  # Card is colorless and filter includes colorless
    
    return any(color in filter_colors for color in card_colors) if card_colors else False

@lru_cache(maxsize=100)
def cached_card_search(keyword, limit):
    """Cache frequent searches to improve performance"""
    return rag_system.search_cards_by_keyword(keyword, limit)

@app.route('/api/rag/status', methods=['GET'])
def rag_status():
    """Get the status of the RAG system"""
    try:
        price_index_status = rag_system.price_index is not None
        text_index_status = rag_system.text_index is not None
        
        return jsonify({
            "success": True,
            "status": {
                "price_index": price_index_status,
                "text_index": text_index_status,
                "db_connection": rag_system.db.is_connected,
                "embedding_model": rag_system.embedding_available
            }
        })
    except Exception as e:
        logger.error(f"Error checking RAG status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/rag/similar-price', methods=['GET'])
def similar_price():
    """Get cards with similar price characteristics"""
    card = request.args.get('card', '')
    if not card:
        return jsonify({"success": False, "error": "No card provided"}), 400
    
    limit = min(int(request.args.get('limit', 10)), 100)  # Cap at 100
    page = int(request.args.get('page', 1))
    
    # Filters
    rarity = request.args.get('rarity')
    colors = validate_colors(request.args.get('colors'))
    format_legality = validate_format(request.args.get('format'))
    
    try:
        similar_cards = rag_system.retrieve_cards_by_price(card, limit * 2 if (colors or rarity or format_legality) else limit)
        
        # Apply filters
        if colors or rarity or format_legality:
            filtered_cards = []
            for card_data in similar_cards:
                # Apply color filter
                if colors and not check_color_match(card_data.get('colors', []), colors):
                    continue
                
                # Apply rarity filter
                if rarity and card_data.get('rarity') != rarity:
                    continue
                
                # Apply format legality filter
                if format_legality:
                    # Fetch legality if not available
                    legalities = card_data.get('legalities', {})
                    if isinstance(legalities, str):
                        try:
                            import json
                            legalities = json.loads(legalities)
                        except:
                            legalities = {}
                    
                    if not legalities.get(format_legality) in ['legal', 'restricted']:
                        continue
                
                filtered_cards.append(card_data)
            
            similar_cards = filtered_cards
        
        # Apply pagination
        start = (page - 1) * limit
        paginated_results = similar_cards[start:start + limit]
        
        return jsonify({
            "success": True,
            "card": card,
            "similar_cards": paginated_results,
            "total": len(similar_cards),
            "page": page,
            "limit": limit
        })
        
    except Exception as e:
        logger.error(f"Error finding similar price cards: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/rag/similar-text', methods=['GET'])
def similar_text():
    """Find cards with similar text content"""
    query = request.args.get('query')
    if not query:
        return jsonify({"success": False, "error": "No query provided"}), 400
    
    limit = min(int(request.args.get('limit', 10)), 100)  # Cap at 100
    page = int(request.args.get('page', 1))
    
    # Parse and validate filters
    colors = validate_colors(request.args.get('colors'))
    card_type = request.args.get('type')
    rarity = request.args.get('rarity')
    format_legality = validate_format(request.args.get('format'))
    
    try:
        # Get results
        results = rag_system.retrieve_cards_by_text(query, limit * 2 if (colors or card_type or rarity or format_legality) else limit)
        
        # Apply filters
        if colors or card_type or rarity or format_legality:
            filtered_results = []
            for card in results:
                # Apply color filter - match ANY of the specified colors or colorless
                if colors and not check_color_match(card.get('colors', []), colors):
                    continue
                
                # Apply type filter
                if card_type and card_type.lower() not in card.get('type_line', '').lower():
                    continue
                    
                # Apply rarity filter
                if rarity and card.get('rarity') != rarity:
                    continue
                
                # Apply format legality filter
                if format_legality:
                    # Fetch legality if not available
                    legalities = card.get('legalities', {})
                    if isinstance(legalities, str):
                        try:
                            import json
                            legalities = json.loads(legalities)
                        except:
                            legalities = {}
                    
                    if not legalities.get(format_legality) in ['legal', 'restricted']:
                        continue
                
                filtered_results.append(card)
            
            results = filtered_results
        
        # Apply pagination
        start = (page - 1) * limit
        paginated_results = results[start:start + limit]
        
        return jsonify({
            "success": True,
            "query": query,
            "similar_cards": paginated_results,
            "total": len(results),
            "page": page,
            "limit": limit
        })
        
    except Exception as e:
        logger.error(f"Error searching similar text: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/rag/deck-recommendation', methods=['POST'])
def deck_recommendation():
    """Get a deck recommendation based on strategy and constraints"""
    data = request.json
    
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    strategy = data.get('strategy', '')
    if not strategy:
        return jsonify({"success": False, "error": "No strategy provided"}), 400
    
    commander = data.get('commander', None)
    budget = data.get('budget', None)
    
    # Validate colors if provided
    colors_raw = data.get('colors', None)
    colors = validate_colors(','.join(colors_raw)) if isinstance(colors_raw, list) else validate_colors(colors_raw)
    
    # Additional options
    card_limit = min(int(data.get('card_limit', 60)), 100)
    include_lands = data.get('include_lands', True)
    format_legality = validate_format(data.get('format', 'commander'))
    
    try:
        deck = rag_system.generate_deck_recommendation(strategy, commander, budget, colors)
        
        # Apply format legality filter
        if format_legality and 'cards' in deck:
            legal_cards = []
            for card in deck['cards']:
                # Fetch legality if not available
                legalities = card.get('legalities', {})
                if isinstance(legalities, str):
                    try:
                        import json
                        legalities = json.loads(legalities)
                    except:
                        legalities = {}
                
                if legalities.get(format_legality) in ['legal', 'restricted']:
                    legal_cards.append(card)
            
            deck['cards'] = legal_cards
        
        # Apply additional customization
        if not include_lands and 'cards' in deck:
            deck['cards'] = [card for card in deck['cards'] if 'Land' not in card.get('type_line', '')]
        
        if 'cards' in deck:
            deck['cards'] = deck['cards'][:card_limit]
            deck['total_cards'] = len(deck['cards'])
        
        return jsonify({
            "success": True,
            "deck": deck
        })
        
    except Exception as e:
        logger.error(f"Error generating deck recommendation: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/rag/search-keyword', methods=['GET'])
def search_keyword():
    """Search cards by keyword with filtering options"""
    # Check both parameter names for backward compatibility
    keyword = request.args.get('keyword') or request.args.get('query')
    if not keyword:
        return jsonify({"success": False, "error": "No keyword provided"}), 400
    
    # Parse and validate parameters
    limit = min(int(request.args.get('limit', 10)), 100)  # Cap at 100
    page = int(request.args.get('page', 1))
    
    colors = validate_colors(request.args.get('colors'))
    card_type = request.args.get('type')
    rarity = request.args.get('rarity')
    format_legality = validate_format(request.args.get('format'))
    
    try:
        # Use cached results for frequent searches
        if len(keyword) < 20 and not colors and not card_type and not rarity and not format_legality:
            results = cached_card_search(keyword, limit)
        else:
            # Call your RAG system method
            results = rag_system.search_cards_by_keyword(keyword, limit * 2 if (colors or card_type or rarity or format_legality) else limit)
        
        # Apply filters
        if colors or card_type or rarity or format_legality:
            filtered_results = []
            for card in results:
                # Apply color filter - match ANY of the specified colors or colorless
                if colors and not check_color_match(card.get('colors', []), colors):
                    continue
                
                # Apply type filter
                if card_type and card_type.lower() not in card.get('type_line', '').lower():
                    continue
                    
                # Apply rarity filter
                if rarity and card.get('rarity') != rarity:
                    continue
                
                # Apply format legality filter
                if format_legality:
                    # Fetch legality if not available
                    legalities = card.get('legalities', {})
                    if isinstance(legalities, str):
                        try:
                            import json
                            legalities = json.loads(legalities)
                        except:
                            legalities = {}
                    
                    if not legalities.get(format_legality) in ['legal', 'restricted']:
                        continue
                
                filtered_results.append(card)
            
            results = filtered_results
        
        # Apply pagination
        start = (page - 1) * limit
        paginated_results = results[start:start + limit]
        
        return jsonify({
            "success": True,
            "keyword": keyword,
            "cards": paginated_results,
            "total": len(results),
            "page": page,
            "limit": limit
        })
        
    except Exception as e:
        logger.error(f"Error searching by keyword: {e}")
        return jsonify({"success": False, "error": str(e)}), 500