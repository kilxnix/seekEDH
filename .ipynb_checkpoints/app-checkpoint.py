# app.py
from flask import Flask, request, jsonify
import os
import json
import logging
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.rag_system import MTGRetrievalSystem
from src.db_interface import DatabaseInterface
import config

# Configure logging
logger = logging.getLogger("__main__")

# Initialize Flask app
app = Flask(__name__)

# Load your existing RAG system
rag_system = MTGRetrievalSystem(embedding_model_name=config.EMBEDDING_MODEL)

# Load trained model (if available)
MODEL_PATH = os.path.join(config.BASE_DIR, "mtg-ai-framework/models/versions/First-Version")
if os.path.exists(MODEL_PATH):
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        tokenizer = None
else:
    logger.warning(f"Model path {MODEL_PATH} not found")
    model = None
    tokenizer = None

@app.route('/generate-deck', methods=['POST'])
def generate_dual_deck():
    try:
        data = request.json
        strategy = data.get('strategy', '')
        commander = data.get('commander', '')
        max_price = data.get('max_price', 200)
        budget_price = data.get('budget_price', 50)
        
        # Step 1: Generate deck with trained model or fallback to RAG
        if model and tokenizer:
            deck_json = generate_deck_with_model(strategy, commander)
        else:
            # Fallback to RAG-based generation if model isn't available
            deck_json = generate_deck_with_rag(strategy, commander)
        
        # Step 2: Simple validation
        improved_deck = validate_deck(deck_json, strategy)
        
        # Step 3: Create budget version with RAG
        budget_deck = create_budget_version(improved_deck, budget_price)
        
        # Step 4: Save both decks
        saved_original = save_deck(improved_deck, "original", strategy, max_price)
        saved_budget = save_deck(budget_deck, "budget", strategy, budget_price)
        
        return jsonify({
            "original_deck": {
                "id": saved_original['id'],
                "commander": improved_deck['commander'],
                "price": improved_deck.get('total_price', 0),
                "url": f"{config.SUPABASE_URL}/storage/v1/object/public/decks/{saved_original['id']}.json"
            },
            "budget_deck": {
                "id": saved_budget['id'],
                "commander": budget_deck['commander'],
                "price": budget_deck.get('total_price', 0),
                "url": f"{config.SUPABASE_URL}/storage/v1/object/public/decks/{saved_budget['id']}.json"
            }
        })
    
    except Exception as e:
        logger.error(f"Error generating decks: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/deck/<deck_id>', methods=['GET'])
def get_deck(deck_id):
    """Retrieve a deck by its ID"""
    try:
        # Query the database for the deck
        response = rag_system.db.client.table("saved_decks").select("*").eq("id", deck_id).execute()
        
        if not response.data or len(response.data) == 0:
            return jsonify({"error": "Deck not found"}), 404
        
        deck_data = response.data[0]
        
        # Try to get the full deck JSON from storage
        try:
            storage_response = rag_system.db.client.storage.from_("decks").download(f"{deck_id}.json")
            full_deck = json.loads(storage_response)
            return jsonify(full_deck)
        except:
            # If storage retrieval fails, return database record
            return jsonify({
                "id": deck_data["id"],
                "name": deck_data["name"],
                "description": deck_data.get("description", ""),
                "created_at": deck_data.get("created_at", "")
            })
        
    except Exception as e:
        logger.error(f"Error retrieving deck: {e}")
        return jsonify({"error": str(e)}), 500
        
def generate_deck_with_model(strategy, commander=None):
    """Generate a deck using the trained model with improved prompting"""
    try:
        # Create a more explicit prompt for JSON output
        prompt = f"""Generate an MTG Commander deck with the following strategy: {strategy}
        {"Commander: " + commander if commander else ""}
        
        Output the deck in JSON format like this:
        {{
          "commander": "Commander Name",
          "decklist": {{
            "Commander": ["Commander Name"],
            "Creatures": ["Creature 1", "Creature 2", ...],
            "Instants": ["Instant 1", "Instant 2", ...],
            "Sorceries": ["Sorcery 1", "Sorcery 2", ...],
            "Artifacts": ["Artifact 1", "Artifact 2", ...],
            "Enchantments": ["Enchantment 1", "Enchantment 2", ...],
            "Planeswalkers": ["Planeswalker 1", "Planeswalker 2", ...],
            "Lands": ["Land 1", "Land 2", ...]
          }},
          "total_price": 0
        }}
        """
        
        # Generate text with the model
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(
            input_ids,
            max_length=2048,  # Longer output
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,  # Fix padding warning
            attention_mask=torch.ones_like(input_ids)  # Fix attention mask warning
        )
        
        deck_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Try to extract JSON, or fallback to RAG if not possible
        try:
            deck_json_str = extract_json_from_text(deck_text)
            deck_json = json.loads(deck_json_str)
            logger.info(f"Successfully generated deck with model, commander: {deck_json.get('commander', 'None')}")
            return deck_json
        except:
            logger.warning("Model didn't output valid JSON, using RAG-based generation instead")
            return generate_deck_with_rag(strategy, commander)
    
    except Exception as e:
        logger.error(f"Error generating deck with model: {e}")
        return generate_deck_with_rag(strategy, commander)

def generate_deck_with_rag(strategy, commander=None):
    """Generate a deck using RAG system without OpenAI"""
    try:
        # Structure for a basic deck
        deck = {
            "commander": commander if commander else "Unknown Commander",
            "decklist": {
                "Commander": [commander] if commander else ["Unknown Commander"],
                "Creatures": [],
                "Instants": [],
                "Sorceries": [],
                "Artifacts": [],
                "Enchantments": [],
                "Planeswalkers": [],
                "Lands": []
            },
            "total_price": 0
        }
        
        # Find cards matching the strategy
        matching_cards = rag_system.retrieve_cards_by_text(strategy, top_k=60)
        
        # Sort cards into categories
        for card in matching_cards:
            card_type = card.get("type_line", "")
            
            if "Creature" in card_type and len(deck["decklist"]["Creatures"]) < 30:
                deck["decklist"]["Creatures"].append(card["name"])
            elif "Instant" in card_type and len(deck["decklist"]["Instants"]) < 10:
                deck["decklist"]["Instants"].append(card["name"])
            elif "Sorcery" in card_type and len(deck["decklist"]["Sorceries"]) < 10:
                deck["decklist"]["Sorceries"].append(card["name"])
            elif "Artifact" in card_type and len(deck["decklist"]["Artifacts"]) < 10:
                deck["decklist"]["Artifacts"].append(card["name"])
            elif "Enchantment" in card_type and len(deck["decklist"]["Enchantments"]) < 10:
                deck["decklist"]["Enchantments"].append(card["name"])
            elif "Planeswalker" in card_type and len(deck["decklist"]["Planeswalkers"]) < 5:
                deck["decklist"]["Planeswalkers"].append(card["name"])
            elif "Land" in card_type and len(deck["decklist"]["Lands"]) < 35:
                deck["decklist"]["Lands"].append(card["name"])
        
        # Add basic lands to reach 100 cards
        total_cards = sum(len(cards) for cards in deck["decklist"].values())
        lands_needed = 100 - total_cards
        
        if lands_needed > 0:
            colors = []
            if commander:
                # Try to get commander's color identity
                commander_response = rag_system.db.client.table("mtg_cards").select(
                    "color_identity"
                ).eq("name", commander).limit(1).execute()
                
                if commander_response.data:
                    colors = commander_response.data[0].get("color_identity", [])
            
            # Add basic lands based on colors
            basic_lands = []
            if "W" in colors:
                basic_lands.append("Plains")
            if "U" in colors:
                basic_lands.append("Island")
            if "B" in colors:
                basic_lands.append("Swamp")
            if "R" in colors:
                basic_lands.append("Mountain")
            if "G" in colors:
                basic_lands.append("Forest")
            
            # Default to all basics if no colors found
            if not basic_lands:
                basic_lands = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
            
            # Add lands evenly
            for i in range(lands_needed):
                deck["decklist"]["Lands"].append(basic_lands[i % len(basic_lands)])
        
        return deck
        
    except Exception as e:
        logger.error(f"Error in RAG fallback: {e}")
        # Return a minimal valid deck structure
        return {
            "commander": commander if commander else "Unknown Commander",
            "decklist": {
                "Commander": [commander] if commander else ["Unknown Commander"],
                "Creatures": [],
                "Instants": [],
                "Sorceries": [],
                "Artifacts": [],
                "Enchantments": [],
                "Planeswalkers": [],
                "Lands": ["Plains", "Island", "Swamp", "Mountain", "Forest"]
            },
            "total_price": 0
        }

def extract_json_from_text(text):
    """Extract JSON from generated text"""
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    
    if start_idx == -1 or end_idx == 0:
        raise ValueError("No JSON found in generated text")
    
    return text[start_idx:end_idx]

def validate_deck(deck_json, strategy):
    """Simple validation without OpenAI dependency"""
    try:
        # Check if we have a commander
        if not deck_json.get("commander") or deck_json["commander"] == "Unknown Commander":
            deck_json["commander"] = strategy.split()[0]  # Just use first word of strategy as placeholder
            
        # Make sure we have commander in the list
        if "Commander" not in deck_json["decklist"] or not deck_json["decklist"]["Commander"]:
            deck_json["decklist"]["Commander"] = [deck_json["commander"]]
        
        # Check if we have a reasonable number of cards
        total_cards = sum(len(cards) for cards in deck_json["decklist"].values())
        
        if total_cards < 90:
            # Add basic lands to reach 100 cards
            lands_needed = 100 - total_cards
            if "Lands" not in deck_json["decklist"]:
                deck_json["decklist"]["Lands"] = []
            
            deck_json["decklist"]["Lands"].extend(["Plains", "Island", "Swamp", "Mountain", "Forest"] * 
                                               ((lands_needed + 4) // 5))
            deck_json["decklist"]["Lands"] = deck_json["decklist"]["Lands"][:len(deck_json["decklist"]["Lands"]) + lands_needed]
        
        # Calculate total price
        total_price = calculate_deck_price(deck_json)
        deck_json["total_price"] = total_price
        
        return deck_json
    
    except Exception as e:
        logger.error(f"Error validating deck: {e}")
        return deck_json

def create_budget_version(original_deck, budget_price):
    """Create budget version of deck using RAG system"""
    try:
        budget_deck = original_deck.copy()
        budget_decklist = {}
        
        # Process each card category
        for category, cards in original_deck["decklist"].items():
            budget_cards = []
            
            # Keep commander as is
            if category == "Commander":
                budget_decklist[category] = cards
                continue
            
            for card in cards:
                # Find cheaper alternative
                cheaper_card = find_cheaper_alternative(card, category, budget_price/len(cards))
                budget_cards.append(cheaper_card)
            
            budget_decklist[category] = budget_cards
        
        budget_deck["decklist"] = budget_decklist
        
        # Calculate new total price
        total_price = calculate_deck_price(budget_deck)
        budget_deck["total_price"] = total_price
        
        logger.info(f"Created budget deck, commander: {budget_deck['commander']}, price: ${total_price}")
        return budget_deck
    
    except Exception as e:
        logger.error(f"Error creating budget version: {e}")
        return original_deck  # Return original if budget creation fails

def find_cheaper_alternative(card_name, card_type, price_target):
    """Find cheaper alternative for a card using RAG system"""
    try:
        # First get the original card to see its price and effects
        original_card_response = rag_system.db.client.table("mtg_cards").select(
            "id, name, oracle_text, type_line, prices_usd, color_identity"
        ).eq("name", card_name).limit(1).execute()
        
        if not original_card_response.data:
            logger.warning(f"Card not found: {card_name}")
            return card_name
        
        original_card = original_card_response.data[0]
        original_price = float(original_card.get("prices_usd", 0) or 0)
        
        # If card is already cheap, keep it
        if original_price <= price_target:
            return card_name
        
        # Use text similarity to find functionally similar cards
        similar_cards = rag_system.retrieve_cards_by_text(
            original_card.get("oracle_text", ""), 
            top_k=30
        )
        
        # Filter by color identity and type
        filtered_cards = []
        for card in similar_cards:
            # Skip the original card
            if card["name"] == card_name:
                continue
                
            # Check if type line contains the category (e.g., "Creature", "Instant")
            if card_type.rstrip("s") not in card.get("type_line", ""):
                continue
                
            # Check if card type is valid for the category
            card_price = float(card.get("prices_usd", 0) or 0)
            
            # Keep cards cheaper than original and below target
            if card_price < original_price and card_price <= price_target:
                filtered_cards.append(card)
        
        # Sort by similarity (most similar first)
        filtered_cards.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Return the best alternative, or original if no good alternative
        if filtered_cards:
            logger.info(f"Found cheaper alternative for {card_name}: {filtered_cards[0]['name']}")
            return filtered_cards[0]["name"]
        else:
            return card_name
    
    except Exception as e:
        logger.error(f"Error finding cheaper alternative for {card_name}: {e}")
        return card_name

def calculate_deck_price(deck):
    """Calculate total price of a deck"""
    total_price = 0
    
    for category, cards in deck["decklist"].items():
        for card in cards:
            card_response = rag_system.db.client.table("mtg_cards").select(
                "prices_usd"
            ).eq("name", card).limit(1).execute()
            
            if card_response.data:
                price = float(card_response.data[0].get("prices_usd", 0) or 0)
                total_price += price
    
    return round(total_price, 2)

def save_deck(deck, version, strategy, price_limit):
    """Save deck to Supabase"""
    try:
        # Insert into database with correct column names based on the schema
        response = rag_system.db.client.table("saved_decks").insert({
            "name": deck["commander"],  # Uses 'name' instead of 'commander'
            "description": strategy,    # Uses 'description' for strategy
            # Add any other fields that match your actual schema
        }).execute()
        
        if not response.data or len(response.data) == 0:
            raise Exception("Failed to save deck to database")
        
        saved_deck = response.data[0]
        
        # Try saving to storage (might not be available)
        try:
            rag_system.db.client.storage.from_("decks").upload(
                f"{saved_deck['id']}.json", 
                json.dumps(deck).encode("utf-8"),
                {"content-type": "application/json"}
            )
        except Exception as storage_e:
            logger.warning(f"Storage upload failed: {storage_e}, but deck saved to database")
        
        logger.info(f"Saved {version} deck to database with ID: {saved_deck['id']}")
        return saved_deck
    
    except Exception as e:
        logger.error(f"Error saving {version} deck: {e}")
        # Return a mock saved deck to continue execution
        return {"id": "temp-" + str(hash(str(deck["commander"]) + version))}

if __name__ == "__main__":
    app.run(host=config.API_HOST, port=config.API_PORT, debug=True)