# integration.py
import logging
from src.rag_system import MTGRetrievalSystem
from src.db_interface import DatabaseInterface

# Import the core deck building logic
from deck_generator import (
    validate_deck_identity,
    get_commander_identity,
    complete_deck,
    filter_deck_for_bracket,
    calculate_deck_price
)

class EnhancedDeckBuilder:
    def __init__(self):
        self.rag_system = MTGRetrievalSystem()
        self.db = DatabaseInterface()
        
    def generate_deck(self, commander_name, strategy, bracket=2, price_limit=None, land_quality='balanced'):
        """Generate a deck using enhanced capabilities"""
        # 1. Get commander identity
        commander_identity = get_commander_identity(commander_name)
        
        # 2. Use RAG to find relevant cards based on strategy
        strategy_cards = self.rag_system.retrieve_cards_by_text(strategy, top_k=50)
        
        # 3. Start building deck with commander
        initial_deck = {commander_name: 1}
        
        # 4. Add strategy-specific cards
        for card in strategy_cards:
            initial_deck[card['name']] = 1
        
        # 5. Validate deck against commander's color identity
        valid_deck, invalid_cards = validate_deck_identity(initial_deck, commander_identity)
        
        # 6. Complete the deck to have 100 cards
        completed_deck = complete_deck(valid_deck, commander_name, commander_identity, 
                                      land_quality=land_quality, max_price=price_limit)
        
        # 7. Filter for selected bracket
        filtered_deck, removed_cards = filter_deck_for_bracket(completed_deck, bracket)
        
        # 8. Calculate price
        total_price = calculate_deck_price(filtered_deck)
        
        # 9. Format for saving
        formatted_deck = self.format_for_database(filtered_deck, commander_name, strategy, bracket, total_price)
        
        # 10. Save to database
        deck_id = self.save_deck(formatted_deck)
        
        return {
            "deck_id": deck_id,
            "commander": commander_name,
            "decklist": formatted_deck,
            "total_price": total_price
        }
        
    def format_for_database(self, deck, commander, strategy, bracket, price):
        """Format deck for database storage"""
        # Group cards by type
        deck_by_type = {
            "Commander": [commander],
            "Creatures": [],
            "Artifacts": [],
            "Enchantments": [],
            "Instants": [],
            "Sorceries": [],
            "Planeswalkers": [],
            "Lands": []
        }
        
        # Categorize cards
        for card, qty in deck.items():
            if card == commander:
                continue
                
            card_info = self.rag_system.db.get_card_details(card)
            if not card_info:
                continue
                
            card_type = self.determine_card_type(card_info)
            if card_type in deck_by_type:
                for _ in range(qty):
                    deck_by_type[card_type].append(card)
        
        return {
            "commander": commander,
            "strategy": strategy,
            "bracket": bracket,
            "decklist": deck_by_type,
            "total_price": price
        }
    
    def determine_card_type(self, card_info):
        """Determine primary card type"""
        type_line = card_info.get("type_line", "")
        
        if "Land" in type_line:
            return "Lands"
        elif "Creature" in type_line:
            return "Creatures"
        elif "Artifact" in type_line:
            return "Artifacts"
        elif "Enchantment" in type_line:
            return "Enchantments"
        elif "Instant" in type_line:
            return "Instants"
        elif "Sorcery" in type_line:
            return "Sorceries"
        elif "Planeswalker" in type_line:
            return "Planeswalkers"
        else:
            return "Other"
    
    def save_deck(self, formatted_deck):
        """Save deck to database"""
        try:
            response = self.db.client.table("saved_decks").insert({
                "commander": formatted_deck["commander"],
                "decklist": formatted_deck["decklist"],
                "total_price": formatted_deck["total_price"]
            }).execute()
            
            if response.data:
                return response.data[0]["id"]
            return None
        except Exception as e:
            logging.error(f"Error saving deck: {e}")
            return None