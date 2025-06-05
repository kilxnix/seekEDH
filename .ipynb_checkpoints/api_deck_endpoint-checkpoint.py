# api_deck_endpoint.py
from flask import jsonify, request, redirect, url_for
from src.api_server import app
from integrated_deck_service import IntegratedDeckService
import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import BASE_DIR
import re

# Configure logging
logger = logging.getLogger("APIDeckEndpoint")

# Initialize the integrated deck service
deck_service = IntegratedDeckService()

# Define COMMANDER_BRACKETS here for the model-based generation
COMMANDER_BRACKETS = {
    1: {'name': 'Exhibition', 'description': 'Casual, theme-focused decks'},
    2: {'name': 'Core', 'description': 'Balanced, precon-level decks'},
    3: {'name': 'Upgraded', 'description': 'Tuned, higher power decks'},
    4: {'name': 'Optimized', 'description': 'High-powered decks'},
    5: {'name': 'cEDH', 'description': 'Competitive EDH decks'}
}

# Load model for deck generation
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

@app.route('/api/generate-deck', methods=['POST'])
def generate_deck_api():
    """API endpoint to generate a deck using the integrated deck service"""
    try:
        # Parse request data
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Extract parameters
        strategy = data.get('strategy')
        if not strategy:
            return jsonify({"success": False, "error": "Strategy is required"}), 400
        
        # Fix parameter name to match what the service expects
        commander_name = data.get('commander_name')
        bracket = int(data.get('bracket', 2))
        max_price = float(data.get('max_price')) if data.get('max_price') else None
        land_quality = data.get('land_quality', 'balanced')
        
        # Require a commander
        if not commander_name:
            # Find legendary creatures matching the strategy
            query = f"legendary creature {strategy}"
            cards = deck_service.rag_system.retrieve_cards_by_text(query, top_k=10)
            suggested_commanders = []
            
            for card in cards:
                if "Legendary" in card.get("type_line", "") and "Creature" in card.get("type_line", ""):
                    suggested_commanders.append(card["name"])
            
            return jsonify({
                "success": False, 
                "error": "Commander is required",
                "suggested_commanders": suggested_commanders
            }), 400
        
        # Generate deck
        logger.info(f"Generating deck with strategy: {strategy}, commander: {commander_name}, bracket: {bracket}")
        result = deck_service.generate_deck(
            strategy=strategy,
            commander_name=commander_name,
            bracket=bracket,
            max_price=max_price,
            land_quality=land_quality
        )
        
        return jsonify({
            "success": True,
            "deck": result
        })
        
    except Exception as e:
        logger.error(f"Error generating deck: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/deck/<deck_id>', methods=['GET'])
def get_deck_direct(deck_id):
    """Direct endpoint to retrieve a deck by ID"""
    try:
        # Load the deck from file first (fastest option)
        file_path = os.path.join('data/generated_decks', f"{deck_id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return jsonify(json.load(f))
                
        # If file not found, query database
        response = deck_service.db.client.table("saved_decks").select("*").eq("id", deck_id).limit(1).execute()
        
        if not response.data:
            return jsonify({"success": False, "error": "Deck not found"}), 404
        
        # Return simplified database record
        deck_data = response.data[0]
        return jsonify({
            "id": deck_data["id"],
            "commander": deck_data["name"],
            "strategy": deck_data.get("description", ""),
            "total_price": deck_data.get("total_price", 0),
            "deck_list": deck_data.get("decklist", {})
        })
        
    except Exception as e:
        logger.error(f"Error retrieving deck: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/deck/<deck_id>', methods=['GET'])
def get_deck_api(deck_id):
    """API endpoint to get a saved deck by ID"""
    try:
        # First check if the deck exists in the database
        response = deck_service.db.client.table("saved_decks").select("*").eq("id", deck_id).limit(1).execute()
        
        if not response.data:
            # Try to load from file
            file_path = os.path.join('data/generated_decks', f"{deck_id}.json")
            
            if not os.path.exists(file_path):
                return jsonify({"success": False, "error": "Deck not found"}), 404
            
            # Load from file
            with open(file_path, 'r') as f:
                deck = json.load(f)
            
            return jsonify({
                "success": True,
                "deck": deck,
                "source": "file"
            })
        
        # Get deck from database
        db_deck = response.data[0]
        
        # Try to parse the decklist JSON
        try:
            # Make sure to properly parse the JSON string
            if db_deck.get('decklist') and isinstance(db_deck.get('decklist'), str):
                decklist = json.loads(db_deck.get('decklist', '{}'))
            else:
                decklist = db_deck.get('decklist', {})
        except Exception as e:
            logger.error(f"Error parsing decklist JSON: {e}")
            decklist = {}
        
        # Format the response
        deck = {
            "id": db_deck.get('id'),
            "commander": db_deck.get('name'),
            "strategy": db_deck.get('description'),
            "bracket": db_deck.get('bracket'),
            "total_price": db_deck.get('total_price'),
            "deck_list": decklist,
            "created_at": db_deck.get('created_at')
        }
        
        return jsonify({
            "success": True,
            "deck": deck,
            "source": "database"
        })
        
    except Exception as e:
        logger.error(f"Error getting deck: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate-deck-with-model', methods=['POST'])
def generate_deck_with_model_api():
    """API endpoint to generate a deck using the pre-trained model"""
    try:
        # Parse request data
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        strategy = data.get('strategy')
        if not strategy:
            return jsonify({"success": False, "error": "Strategy is required"}), 400
        
        commander_name = data.get('commander_name')
        if not commander_name:
            return jsonify({"success": False, "error": "Commander name is required"}), 400
        
        bracket = int(data.get('bracket', 2))
        max_price = float(data.get('max_price')) if data.get('max_price') else None
        land_quality = data.get('land_quality', 'balanced')
        
        # Check if model is available
        if not MODEL_AVAILABLE:
            logger.warning("Model not available, redirecting to RAG-based generation")
            return jsonify({"success": False, "error": "Model not available", "redirect": "/api/generate-deck"})
        
        # Get commander color identity
        response = deck_service.db.client.table("mtg_cards").select(
            "color_identity"
        ).eq("name", commander_name).limit(1).execute()
        
        commander_identity = []
        if response.data:
            commander_identity = response.data[0].get('color_identity', [])
        
        # Generate prompt for the model
        prompt = f"""You are a Magic: The Gathering deck builder.
CREATE a Commander deck with this strategy: {strategy}
Commander: {commander_name}
Power Level: {bracket}/5

GENERATE the following for a 100-card Commander deck:
- 1 Commander: {commander_name}
- 25-30 Creatures that support a {strategy} theme
- 8-12 Removal/interaction spells
- 8-10 Card draw spells
- 6-8 Ramp artifacts
- 36-38 Lands including basics
- Other support cards

All cards must be within {', '.join(commander_identity)} color identity.
Basic lands can have multiple copies; all other cards must be singleton.

RESPOND ONLY WITH THIS JSON:
{{
  "deck_list": {{
    "Commander": ["{commander_name}"],
    "Creatures": ["REAL MTG CREATURE NAME 1", "REAL MTG CREATURE NAME 2", ...],
    "Artifacts": ["REAL MTG ARTIFACT NAME 1", "REAL MTG ARTIFACT NAME 2", ...],
    "Enchantments": ["REAL MTG ENCHANTMENT NAME 1", ...],
    "Instants": ["REAL MTG INSTANT NAME 1", ...],
    "Sorceries": ["REAL MTG SORCERY NAME 1", ...],
    "Planeswalkers": ["REAL MTG PLANESWALKER NAME 1", ...],
    "Lands": ["REAL MTG LAND NAME 1", ...]
  }}
}}
"""
        logger.info(f"Generating deck with model using prompt length: {len(prompt)}")
        
        # Generate text with the model
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(
            input_ids,
            max_length=5600,
            temperature=0.9,  # Increased temperature
            top_p=0.95,       # Increased top_p
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode and process
        result_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Model generated text of length: {len(result_text)}")
        
        # Try to extract JSON from text
        try:
            # Log the full result text for debugging
            logger.info(f"Full model output: {result_text}")
            
            # First try normal JSON parsing
            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON format found, attempting direct extraction")
                deck_json = extract_deck_from_text(result_text)
            else:
                deck_json_str = result_text[start_idx:end_idx]
                fixed_json_str = fix_json_string(deck_json_str)
                
                try:
                    deck_json = json.loads(fixed_json_str)
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}, attempting direct extraction")
                    deck_json = extract_deck_from_text(result_text)
            
            logger.info(f"Successfully extracted deck JSON with {sum(len(cards) for category, cards in deck_json.get('deck_list', {}).items())} cards")
            
            # Process model output
            processed_deck = extract_card_counts(deck_json.get("deck_list", {}))
            
            # Calculate price
            total_price = deck_service.calculate_deck_price(processed_deck)
            
            # Format the deck for storage
            formatted_deck = deck_service.format_deck(
                processed_deck,
                commander_name,
                strategy,
                bracket,
                total_price
            )
            
            # Save the processed deck
            deck_id = deck_service.save_deck(formatted_deck)
            
            # Return the result
            return jsonify({
                "success": True,
                "deck": {
                    "deck_id": deck_id,
                    "commander": commander_name,
                    "strategy": strategy,
                    "bracket": bracket,
                    "bracket_name": COMMANDER_BRACKETS[bracket]['name'],
                    "deck_list": deck_json.get("deck_list", {}),
                    "total_price": total_price,
                    "card_count": sum(len(cards) for category, cards in deck_json.get('deck_list', {}).items())
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing model output: {e}")
            # Fall back to RAG-based generation
            return jsonify({
                "success": False, 
                "error": f"Error processing model output: {e}", 
                "redirect": "/api/generate-deck"
            })
        
    except Exception as e:
        logger.error(f"Error generating deck with model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
        
def fix_json_string(json_str):
    """Attempt to fix common JSON formatting issues in model output"""
    # Replace single quotes with double quotes (common model error)
    json_str = json_str.replace("'", '"')
    
    # Fix missing quotes around keys
    json_str = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', json_str)
    
    # Remove trailing commas before closing brackets
    json_str = re.sub(r',(\s*)}', r'\1}', json_str)
    json_str = re.sub(r',(\s*)]', r'\1]', json_str)
    
    json_str = re.sub(r'"([^"]+) // ([^"]+)"', r'"\1"', json_str)
    
    return json_str
# Add this function for more aggressive JSON fixing
def extract_deck_from_text(result_text):
    """Extract deck information from model output without relying on JSON parsing"""
    deck_list = {"Commander": [], "Creatures": [], "Artifacts": [], "Enchantments": [], 
                 "Instants": [], "Sorceries": [], "Planeswalkers": [], "Lands": []}
    
    # Log the first part of the text to understand the structure
    logger.info(f"Examining model output: {result_text[:500]}...")
    
    # Try to extract sections by looking for patterns
    sections = {}
    current_section = None
    
    for line in result_text.split('\n'):
        # Clean up the line
        clean_line = line.strip('"[],-: ')
        
        # Handle dual-faced format like "Card Name // Other Face"
        if " // " in clean_line:
            clean_line = clean_line.split(" // ")[0]
            
        if clean_line:
            sections[current_section].append(clean_line)
        
        # Skip empty lines
        if not line:
            continue
            
        # Check for section headers
        if line.endswith(':') and '"' not in line:
            current_section = line[:-1].strip()
            sections[current_section] = []
        elif current_section and (line.startswith('-') or line.startswith('"') or line.startswith('[')):
            # Clean up the line - remove quotes, brackets, commas
            clean_line = line.strip('"[],-: ')
            if clean_line:
                sections[current_section].append(clean_line)
    
    # Map sections to deck list categories
    if "Commander" in sections:
        deck_list["Commander"] = sections["Commander"]
    if "Creatures" in sections:
        deck_list["Creatures"] = sections["Creatures"]
    if "Artifacts" in sections:
        deck_list["Artifacts"] = sections["Artifacts"]
    if "Enchantments" in sections:
        deck_list["Enchantments"] = sections["Enchantments"]
    if "Instants" in sections:
        deck_list["Instants"] = sections["Instants"]
    if "Sorceries" in sections:
        deck_list["Sorceries"] = sections["Sorceries"]
    if "Planeswalkers" in sections:
        deck_list["Planeswalkers"] = sections["Planeswalkers"]
    if "Lands" in sections:
        deck_list["Lands"] = sections["Lands"]
    
    return {
        "deck_list": deck_list
    }


# Add these functions to handle dual-faced cards
def extract_card_counts(deck_list):
    """Extract card counts from deck list with dual-faced card support"""
    card_counts = {}
    
    # Get dual-faced card mapping
    flip_card_mapping = deck_service.flip_card_mapping
    
    for category, cards in deck_list.items():
        for card in cards:
            # Check if this is a face of a dual-faced card
            main_card = flip_card_mapping.get(card, card)
            
            if main_card in card_counts:
                card_counts[main_card] += 1
            else:
                card_counts[main_card] = 1
                
    return card_counts