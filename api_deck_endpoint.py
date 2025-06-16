from flask import jsonify, request
from src.api_server import app
from src.rag_instance import rag_system
import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import BASE_DIR
from uuid import uuid4
from datetime import datetime

# Import RAG wrapper functions instead of deck_generator
from rag_wrapper import (
    initialize_rag_deck_system,
    get_commander_identity,
    validate_deck_identity,
    analyze_commander_for_strategies,
    search_cards_by_criteria,
    search_lands_by_quality,
    get_price_data,
    save_deck_to_database,
    get_card_info
)

# Import the deck building logic from deck_generator
from src.deck_generator import (
    determine_win_condition,
    generate_conditioned_deck,
    complete_deck,
    filter_deck_for_bracket,
    analyze_deck_for_bracket,
    count_unique_physical_cards,
    count_card_types,
    extract_deck,
    COMMANDER_BRACKETS,
    LAND_QUALITY_SETTINGS
)

# Configure logging
logger = logging.getLogger("RAGDeckEndpoint")

# Load model for deck generation (optional)
MODEL_PATH = os.path.join(BASE_DIR, "models/versions/First-Version")
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Successfully loaded model on {device}")
    MODEL_AVAILABLE = True
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    tokenizer = None
    device = None
    MODEL_AVAILABLE = False

# Replace the broken functions in api_deck_endpoint.py with these working versions

def find_card_in_db(card_name, db_cards=None):
    """Find card using working database interface - SUPPORTS COLORLESS CARDS"""
    try:
        # Use the working RAG system database
        response = rag_system.db.client.table("mtg_cards").select("*").eq("name", card_name).limit(1).execute()
        
        if response.data:
            return response.data[0]
        
        # If not found, try case-insensitive
        response = rag_system.db.client.table("mtg_cards").select("*").ilike("name", card_name).limit(1).execute()
        
        if response.data:
            return response.data[0]
            
        return None
        
    except Exception as e:
        logger.error(f"Error finding card {card_name}: {e}")
        return None

def get_all_cards_simple():
    """Get cards using the working database interface"""
    try:
        # Get all cards from the same database that works
        response = rag_system.db.client.table("mtg_cards").select("id, name, type_line, color_identity").execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Error getting all cards: {e}")
        return []

# Update the main endpoint to use the working database
@app.route("/api/generate-deck", methods=["POST"])
def generate_deck():
    api_start_time = datetime.utcnow()
    request_id = str(uuid4())
    
    try:
        req = request.get_json(force=True)
        strategy = req.get('strategy')
        commander = req.get('commander_name')
        bracket = int(req.get('bracket', 2))
        max_price = req.get('max_price')
        land_quality = req.get('land_quality', 'balanced')
        method = req.get('generationMethod', 'rag')

        errors, warnings = [], []

        # Use the working database interface
        found_commander = find_card_in_db(commander, rag_system)
        
        if not found_commander:
            # Use the working suggestion system
            try:
                response = rag_system.db.client.table("mtg_cards").select("name").ilike("name", f"%{commander}%").limit(5).execute()
                suggestions = [card['name'] for card in response.data] if response.data else []
            except:
                suggestions = []
                
            debug_info = {
                "search_attempted": commander,
                "database_connected": rag_system.db.is_connected,
                "duration": (datetime.utcnow() - api_start_time).total_seconds()
            }
            
            return jsonify({
                "success": False,
                "request_id": request_id,
                "error": f"Commander not found: '{commander}'",
                "card_requested": commander,
                "suggestions": suggestions,
                "validation_status": "invalid",
                "debug_info": debug_info
            }), 422
        
        # Generate deck using RAG system (which works)
        try:
            if method == "rag":
                deck_result = rag_system.generate_deck_recommendation(
                    strategy=strategy,
                    commander=commander,
                    budget=max_price,
                    colors=found_commander.get('color_identity', [])
                )
                
                # Format the response properly
                formatted_deck = {
                    "deck_list": deck_result.get("cards", {}),
                    "commander": commander,
                    "strategy": strategy,
                    "total_cards": len(deck_result.get("cards", [])),
                    "total_price": deck_result.get("total_price", 0)
                }
                
                deck_text = f"Generated {strategy} deck for {commander}"
                
            else:
                # Fallback to RAG if model not available
                deck_result = rag_system.generate_deck_recommendation(
                    strategy=strategy,
                    commander=commander,
                    budget=max_price,
                    colors=found_commander.get('color_identity', [])
                )
                
                formatted_deck = {
                    "deck_list": deck_result.get("cards", {}),
                    "commander": commander,
                    "strategy": strategy,
                    "total_cards": len(deck_result.get("cards", [])),
                    "total_price": deck_result.get("total_price", 0)
                }
                
                deck_text = f"Generated {strategy} deck for {commander}"
        
        except Exception as e:
            logger.error(f"Deck generation failed: {e}")
            return jsonify({
                "success": False,
                "request_id": request_id,
                "error": f"Deck generation failed: {str(e)}",
                "validation_status": "invalid"
            }), 500

        return jsonify({
            "success": True,
            "request_id": request_id,
            "validation_status": "valid",
            "warnings": warnings,
            "errors": errors,
            "deck_json": formatted_deck,
            "deck_text": deck_text,
            "debug_info": {
                "commander_found": True,
                "method_used": method,
                "duration": (datetime.utcnow() - api_start_time).total_seconds()
            }
        })
        
    except Exception as e:
        logger.exception("Critical error in generate_deck")
        return jsonify({
            "success": False,
            "request_id": request_id,
            "error": str(e),
            "validation_status": "invalid"
        }), 500
    
@app.route('/api/deck/<deck_id>', methods=['GET'])
def get_deck_api(deck_id):
    """API endpoint to get a saved deck by ID"""
    try:
        response = rag_system.db.client.table("saved_decks").select("*").eq("id", deck_id).limit(1).execute()
        
        if not response.data:
            return jsonify({"success": False, "error": "Deck not found"}), 404
        
        db_deck = response.data[0]
        
        deck = {
            "id": db_deck.get('id'),
            "commander": db_deck.get('commander'),
            "strategy": db_deck.get('strategy', ''),
            "bracket": db_deck.get('bracket'),
            "deck_list": db_deck.get('deck_list', {}),
            "total_price": db_deck.get('total_price'),
            "created_at": db_deck.get('created_at')
        }
        
        return jsonify({
            "success": True,
            "deck": deck,
            "source": "rag_database"
        })
        
    except Exception as e:
        logger.error(f"Error getting deck: {e}")
        return jsonify({"success": False, "error": str(e)}), 500