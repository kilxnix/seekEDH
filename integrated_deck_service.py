# integrated_deck_service.py
import logging
import os
import json
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple

# Import RAG system components
from src.rag_system import MTGRetrievalSystem
from src.db_interface import DatabaseInterface

# Import deck generator components
from src.deck_generator import (
    validate_deck_identity,
    get_commander_identity,
    complete_deck,
    filter_deck_for_bracket,
    replace_removed_cards,
    count_card_types,
    optimize_for_cedh,
    analyze_deck_for_bracket,
    COMMANDER_BRACKETS
)
from app import (
    calculate_deck_price,
    validate_deck,  # instead of validate_deck_identity
    generate_deck_with_rag,  # instead of get_commander_identity
    create_budget_version,  # instead of complete_deck
    save_deck
)

logger = logging.getLogger("IntegratedDeckService")

# Define COMMANDER_BRACKETS here since it's not in app.py
COMMANDER_BRACKETS = {
    1: {'name': 'Exhibition', 'description': 'Casual, theme-focused decks'},
    2: {'name': 'Core', 'description': 'Balanced, precon-level decks'},
    3: {'name': 'Upgraded', 'description': 'Tuned, higher power decks'},
    4: {'name': 'Optimized', 'description': 'High-powered decks'},
    5: {'name': 'cEDH', 'description': 'Competitive EDH decks'}
}

class IntegratedDeckService:
    """Service that integrates the RAG system with deck generator functions"""
    
    def __init__(self):
        """Initialize the integrated deck service"""
        self.rag_system = MTGRetrievalSystem()
        self.db = self.rag_system.db
        self.flip_card_mapping = {}  # Added for dual-faced cards
        
        # Ensure the required directories exist
        os.makedirs('data/generated_decks', exist_ok=True)

        # Load dual-faced card mapping
        self.load_dual_faced_mapping()

    def load_dual_faced_mapping(self):
        """Load mapping between dual-faced cards and their faces"""
        try:
            # Query all cards with card_faces
            response = self.db.client.table("mtg_cards").select(
                "name, card_faces"
            ).not_.is_("card_faces", "null").execute()
            
            if response.data:
                for card in response.data:
                    main_name = card.get('name')
                    card_faces = card.get('card_faces')
                    
                    # Parse card_faces if it's a string
                    if isinstance(card_faces, str):
                        try:
                            card_faces = json.loads(card_faces)
                        except:
                            continue
                    
                    # Map each face to the main card
                    if isinstance(card_faces, list):
                        for face in card_faces:
                            if isinstance(face, dict) and 'name' in face:
                                face_name = face.get('name')
                                if face_name != main_name:
                                    self.flip_card_mapping[face_name] = main_name
                
                logger.info(f"Loaded {len(self.flip_card_mapping)} dual-faced card mappings")
            
        except Exception as e:
            logger.error(f"Error loading dual-faced card mapping: {e}")
    
    def count_unique_physical_cards(self, deck):
        """Count unique physical cards accounting for dual-faced cards"""
        physical_cards = set()
        
        for card in deck.keys():
            # If the card is a face, use the main card name
            if card in self.flip_card_mapping:
                physical_cards.add(self.flip_card_mapping[card])
            else:
                physical_cards.add(card)
                
        return len(physical_cards)
    def generate_deck(self, 
                     strategy: str, 
                     commander_name: Optional[str] = None,
                     bracket: int = 2, 
                     max_price: Optional[float] = None,
                     land_quality: str = 'balanced') -> Dict[str, Any]:
        """
        Generate a deck using the integrated approach
        
        Args:
            strategy: Description of the deck strategy
            commander_name: Optional commander name
            bracket: Power level bracket (1-5)
            max_price: Maximum price for the deck
            land_quality: Quality setting for lands ('competitive', 'balanced', or 'budget')
            
        Returns:
            Dictionary with the generated deck
        """
        logger.info(f"Generating deck with strategy: {strategy}, commander: {commander_name}")
        
        # 1. Determine color identity
        commander_identity = []
        if commander_name:
            # Get commander identity from database
            response = self.db.client.table("mtg_cards").select(
                "name, color_identity"
            ).eq("name", commander_name).limit(1).execute()
            
            if response.data:
                commander_identity = response.data[0].get('color_identity', [])
                logger.info(f"Found commander {commander_name} with color identity: {commander_identity}")
            else:
                logger.warning(f"Commander {commander_name} not found in database")
        
        # 2. Use RAG to find strategy-relevant cards
        relevant_cards = self.find_strategy_cards(strategy, commander_identity, max_price)
        logger.info(f"Found {len(relevant_cards)} relevant cards for strategy")
        
        # 3. Start building deck with commander
        initial_deck = {}
        if commander_name:
            initial_deck[commander_name] = 1
        
        # 4. Add strategy-specific cards
        for card in relevant_cards[:30]:  # Limit to top 30 cards initially
            if card['name'] != commander_name:  # Avoid duplicating commander
                initial_deck[card['name']] = 1
        
        # 5. Validate deck against commander's color identity
        valid_deck, invalid_cards = self.validate_deck_identity(initial_deck, commander_identity)
        if invalid_cards:
            logger.info(f"Removed {len(invalid_cards)} invalid cards")
        
        # 6. Complete the deck to have 100 cards
        completed_deck = self.complete_deck(valid_deck, commander_name, commander_identity, 
                                         strategy, land_quality, max_price)
        
        # 7. Apply bracket-specific optimizations
        if bracket == 5:  # cEDH
            completed_deck = self.optimize_for_cedh(completed_deck, commander_identity)
        
        # 8. Filter for selected bracket
        filtered_deck, removed_cards = self.filter_deck_for_bracket(completed_deck, bracket)
        
        # 9. Replace removed cards with bracket-appropriate alternatives
        if removed_cards:
            logger.info(f"Replacing {len(removed_cards)} cards removed due to bracket restrictions")
            filtered_deck = self.replace_removed_cards(filtered_deck, removed_cards, 
                                                    commander_identity, bracket, max_price)
        
        # 10. Calculate price
        total_price = self.calculate_deck_price(filtered_deck)
        
        # 11. Format and categorize deck
        formatted_deck = self.format_deck(filtered_deck, commander_name, strategy, bracket, total_price)
        
        # 12. Save deck to database
        deck_id = self.save_deck(formatted_deck)
        
        return {
            "deck_id": deck_id,
            "commander": commander_name,
            "strategy": strategy,
            "bracket": bracket,
            "bracket_name": COMMANDER_BRACKETS[bracket]['name'],
            "deck_list": formatted_deck['deck_list'],
            "total_price": total_price,
            "card_count": sum(qty for card, qty in filtered_deck.items())
        }
    
    def find_strategy_cards(self, strategy: str, color_identity: List[str], max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Find cards relevant to the strategy using RAG
        
        Args:
            strategy: Description of the deck strategy
            color_identity: Color identity to filter cards
            max_price: Maximum price for cards
            
        Returns:
            List of relevant cards
        """
        # Use semantic search if available
        if self.rag_system.embedding_available and self.rag_system.text_index is not None:
            cards = self.rag_system.retrieve_cards_by_text(strategy, top_k=50)
            
            # Filter by color identity
            if color_identity:
                cards = [card for card in cards if self.card_matches_identity(card, color_identity)]
            
            # Filter by price if needed
            if max_price is not None:
                cards = [card for card in cards if self.card_within_budget(card, max_price)]
                
            return cards
        else:
            # Fallback to keyword search
            logger.info("Semantic search not available, using keyword search")
            cards = []
            
            # Extract keywords from strategy
            keywords = [word for word in strategy.split() if len(word) >= 4]
            for keyword in keywords[:5]:  # Use top 5 keywords
                results = self.rag_system.search_cards_by_keyword(keyword, top_k=10)
                cards.extend(results)
            
            # Remove duplicates
            seen_ids = set()
            unique_cards = []
            for card in cards:
                if card['id'] not in seen_ids:
                    seen_ids.add(card['id'])
                    
                    # Filter by color identity
                    if color_identity and not self.card_matches_identity(card, color_identity):
                        continue
                        
                    # Filter by price
                    if max_price is not None and not self.card_within_budget(card, max_price):
                        continue
                        
                    unique_cards.append(card)
            
            return unique_cards
    
    def card_matches_identity(self, card: Dict[str, Any], color_identity: List[str]) -> bool:
        """Check if card's color identity is within commander's color identity"""
        # Get card's color identity from database if not in card object
        if 'color_identity' not in card:
            response = self.db.client.table("mtg_cards").select(
                "color_identity"
            ).eq("id", card['id']).limit(1).execute()
            
            if response.data:
                card_identity = response.data[0].get('color_identity', [])
            else:
                return True  # If we can't determine, assume it's legal
        else:
            card_identity = card['color_identity']
        
        # Check if all colors in card's identity are in commander's identity
        return all(color in color_identity for color in card_identity)
    
    def card_within_budget(self, card: Dict[str, Any], max_price: float) -> bool:
        """Check if card price is within budget"""
        # Get card price from database if not in card object
        if 'prices_usd' not in card:
            response = self.db.client.table("mtg_cards").select(
                "prices_usd"
            ).eq("id", card['id']).limit(1).execute()
            
            if response.data:
                price = response.data[0].get('prices_usd')
            else:
                return True  # If we can't determine, assume it's within budget
        else:
            price = card.get('prices_usd')
        
        # Check if price is within budget
        if price is None:
            return True
        
        try:
            return float(price) <= max_price
        except (ValueError, TypeError):
            return True
    
    def validate_deck_identity(self, deck: Dict[str, int], commander_identity: List[str]) -> Tuple[Dict[str, int], List[str]]:
        """
        Validate that all cards in the deck match the commander's color identity
        
        Args:
            deck: Dictionary of cards and quantities
            commander_identity: Color identity of the commander
            
        Returns:
            Tuple of (valid_deck, invalid_cards)
        """
        valid_deck = {}
        invalid_cards = []
        
        for card, qty in deck.items():
            # Skip validation for basic lands
            if card in ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes']:
                valid_deck[card] = qty
                continue
            
            # Check if card's color identity is within commander's
            response = self.db.client.table("mtg_cards").select(
                "color_identity"
            ).eq("name", card).limit(1).execute()
            
            if not response.data:
                invalid_cards.append(card)
                continue
                
            card_identity = response.data[0].get('color_identity', [])
            
            # Check if card's color identity is a subset of commander's identity
            if all(color in commander_identity for color in card_identity):
                valid_deck[card] = qty
            else:
                invalid_cards.append(card)
        
        return valid_deck, invalid_cards
    
    def complete_deck(self, partial_deck: Dict[str, int], commander_name: str, 
                    commander_identity: List[str], strategy: str,
                    land_quality: str = 'balanced', max_price: Optional[float] = None) -> Dict[str, int]:
        """
        Complete deck with appropriate cards to reach 100 cards total
        
        Args:
            partial_deck: Partial deck to complete
            commander_name: Name of the commander
            commander_identity: Color identity of the commander
            strategy: Deck strategy
            land_quality: Quality setting for lands
            max_price: Maximum price for cards
            
        Returns:
            Completed deck
        """
        deck = dict(partial_deck)
        
        # Calculate how many more cards we need
        total_cards = sum(deck.values())
        cards_needed = 100 - total_cards
        
        if cards_needed <= 0:
            return deck
            
        logger.info(f"Need to add {cards_needed} more cards to reach 100")
        
        # Calculate balance of card types to add
        lands_to_add = max(0, min(cards_needed, 36 - self.count_lands(deck)))
        creatures_to_add = max(0, min(cards_needed - lands_to_add, 25 - self.count_card_type(deck, "Creature")))
        other_to_add = cards_needed - lands_to_add - creatures_to_add
        
        # Add lands
        if lands_to_add > 0:
            logger.info(f"Adding {lands_to_add} lands")
            lands = self.find_lands(commander_identity, land_quality, max_price, lands_to_add)
            for land in lands:
                if land in deck:
                    deck[land] += 1
                else:
                    deck[land] = 1
        
        # Add creatures
        if creatures_to_add > 0:
            logger.info(f"Adding {creatures_to_add} creatures")
            creatures = self.find_creatures(strategy, commander_identity, max_price, creatures_to_add)
            for creature in creatures:
                if creature in deck:
                    deck[creature] += 1
                else:
                    deck[creature] = 1
        
        # Add other card types
        if other_to_add > 0:
            logger.info(f"Adding {other_to_add} other cards")
            other_cards = self.find_other_cards(strategy, commander_identity, max_price, other_to_add)
            for card in other_cards:
                if card in deck:
                    deck[card] += 1
                else:
                    deck[card] = 1
        
        # Ensure we have exactly 100 cards
        current_total = sum(deck.values())
        if current_total < 100:
            # Add basic lands to make up the difference
            lands_needed = 100 - current_total
            basic_lands = self.get_basic_lands(commander_identity)
            for i in range(lands_needed):
                land = basic_lands[i % len(basic_lands)]
                if land in deck:
                    deck[land] += 1
                else:
                    deck[land] = 1
        elif current_total > 100:
            # Remove excess cards (non-essential)
            excess = current_total - 100
            cards_to_remove = []
            
            # Prioritize removing non-land, non-commander cards
            for card, qty in deck.items():
                if card != commander_name and card not in ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes']:
                    for _ in range(qty):
                        cards_to_remove.append(card)
            
            # Shuffle to randomize which cards get removed
            random.shuffle(cards_to_remove)
            
            # Remove excess cards
            for i in range(excess):
                if i < len(cards_to_remove):
                    card = cards_to_remove[i]
                    if deck[card] > 1:
                        deck[card] -= 1
                    else:
                        del deck[card]
        
        return deck
    
    def find_lands(self, color_identity: List[str], land_quality: str, 
                max_price: Optional[float], count: int) -> List[str]:
        """Find appropriate lands for the deck"""
        lands = []
        
        # Get the basic lands for the color identity
        basic_lands = self.get_basic_lands(color_identity)
        
        # Handle special cases like colorless
        if not color_identity:
            lands.extend(['Wastes'] * count)
            return lands
        
        # Mix of basic and non-basic based on land quality
        if land_quality == 'competitive':
            # Premium lands (search up to 20 to find up to 'count' that meet criteria)
            premium_lands = self.search_premium_lands(color_identity, max_price, count * 2)
            lands.extend(premium_lands[:count])
        elif land_quality == 'budget':
            # Mostly basic lands
            basic_count = count * 3 // 4
            for i in range(basic_count):
                lands.append(basic_lands[i % len(basic_lands)])
            
            # Add some utility lands
            utility_lands = self.search_utility_lands(color_identity, max_price, count - basic_count)
            lands.extend(utility_lands)
        else:  # balanced
            # Mix of basic, utility, and premium
            basic_count = count // 3
            for i in range(basic_count):
                lands.append(basic_lands[i % len(basic_lands)])
            
            # Add utility and premium lands
            premium_lands = self.search_premium_lands(color_identity, max_price, (count - basic_count) // 2)
            utility_lands = self.search_utility_lands(color_identity, max_price, count - basic_count - len(premium_lands))
            
            lands.extend(premium_lands)
            lands.extend(utility_lands)
        
        # If we still need more lands, add basic lands
        while len(lands) < count:
            lands.append(basic_lands[len(lands) % len(basic_lands)])
        
        return lands[:count]
    
    def search_premium_lands(self, color_identity: List[str], max_price: Optional[float], count: int) -> List[str]:
        """Search for premium lands that match color identity and budget"""
        query = "type:land enters battlefield untapped"
        
        response = self.db.client.table("mtg_cards").select(
            "name, type_line, color_identity, prices_usd"
        ).like("type_line", "%Land%").execute()
        
        if not response.data:
            return []
        
        # Filter lands by color identity and budget
        valid_lands = []
        for land in response.data:
            # Check if land's color identity is within deck's color identity
            land_identity = land.get('color_identity', [])
            if not all(color in color_identity for color in land_identity):
                continue
                
            # Check if land is within budget
            if max_price is not None:
                price = land.get('prices_usd')
                if price is not None:
                    try:
                        if float(price) > max_price:
                            continue
                    except (ValueError, TypeError):
                        pass
            
            valid_lands.append(land['name'])
        
        # If we found enough lands, return them
        if len(valid_lands) >= count:
            return valid_lands[:count]
        
        # Otherwise, return what we found
        return valid_lands
    
    def search_utility_lands(self, color_identity: List[str], max_price: Optional[float], count: int) -> List[str]:
        """Search for utility lands that match color identity and budget"""
        response = self.db.client.table("mtg_cards").select(
            "name, type_line, color_identity, prices_usd"
        ).like("type_line", "%Land%").execute()
        
        if not response.data:
            return []
        
        # Filter lands by color identity and budget
        valid_lands = []
        for land in response.data:
            # Check if land's color identity is within deck's color identity
            land_identity = land.get('color_identity', [])
            if not all(color in color_identity for color in land_identity):
                continue
                
            # Check if land is within budget
            if max_price is not None:
                price = land.get('prices_usd')
                if price is not None:
                    try:
                        if float(price) > max_price:
                            continue
                    except (ValueError, TypeError):
                        pass
            
            valid_lands.append(land['name'])
        
        # Shuffle to randomize selection
        random.shuffle(valid_lands)
        
        # Return up to 'count' lands
        return valid_lands[:count]
    
    def get_basic_lands(self, color_identity: List[str]) -> List[str]:
        """Get basic lands appropriate for the color identity"""
        basic_lands = []
        
        if 'W' in color_identity:
            basic_lands.append('Plains')
        if 'U' in color_identity:
            basic_lands.append('Island')
        if 'B' in color_identity:
            basic_lands.append('Swamp')
        if 'R' in color_identity:
            basic_lands.append('Mountain')
        if 'G' in color_identity:
            basic_lands.append('Forest')
        
        # If no colors, use Wastes
        if not basic_lands:
            basic_lands.append('Wastes')
        
        return basic_lands
    
    def find_creatures(self, strategy: str, color_identity: List[str], 
                    max_price: Optional[float], count: int) -> List[str]:
        """Find creatures that match the strategy and color identity"""
        query = f"type:creature {strategy}"
        
        if self.rag_system.embedding_available and self.rag_system.text_index is not None:
            cards = self.rag_system.retrieve_cards_by_text(query, top_k=count * 2)
        else:
            cards = self.rag_system.search_cards_by_keyword("creature", top_k=count * 2)
        
        # Filter by color identity and type
        creature_names = []
        for card in cards:
            # Check if it's actually a creature
            if "type_line" not in card or "Creature" not in card["type_line"]:
                continue
                
            # Check color identity
            if not self.card_matches_identity(card, color_identity):
                continue
                
            # Check budget
            if max_price is not None and not self.card_within_budget(card, max_price):
                continue
                
            creature_names.append(card["name"])
            
            if len(creature_names) >= count:
                break
        
        # If we need more creatures, search for some generic good stuff
        if len(creature_names) < count:
            response = self.db.client.table("mtg_cards").select(
                "name, type_line, color_identity, prices_usd"
            ).like("type_line", "%Creature%").execute()
            
            if response.data:
                # Filter by color identity and budget
                for card in response.data:
                    # Skip if already in list
                    if card["name"] in creature_names:
                        continue
                        
                    # Check color identity
                    if not all(color in color_identity for color in card.get('color_identity', [])):
                        continue
                        
                    # Check budget
                    if max_price is not None:
                        price = card.get('prices_usd')
                        if price is not None:
                            try:
                                if float(price) > max_price:
                                    continue
                            except (ValueError, TypeError):
                                pass
                    
                    creature_names.append(card["name"])
                    
                    if len(creature_names) >= count:
                        break
        
        # Shuffle to randomize selection
        random.shuffle(creature_names)
        
        return creature_names[:count]
    
    def find_other_cards(self, strategy: str, color_identity: List[str], 
                       max_price: Optional[float], count: int) -> List[str]:
        """Find non-creature, non-land cards that match the strategy"""
        # Distribute among card types
        card_types = ["Artifact", "Enchantment", "Instant", "Sorcery", "Planeswalker"]
        cards_per_type = count // len(card_types)
        remainder = count % len(card_types)
        
        # Allocate remainder
        allocation = {card_type: cards_per_type for card_type in card_types}
        for i in range(remainder):
            allocation[card_types[i]] += 1
        
        # Find cards for each type
        other_cards = []
        for card_type, num_cards in allocation.items():
            if num_cards <= 0:
                continue
                
            query = f"type:{card_type.lower()} {strategy}"
            
            if self.rag_system.embedding_available and self.rag_system.text_index is not None:
                cards = self.rag_system.retrieve_cards_by_text(query, top_k=num_cards * 2)
            else:
                cards = self.rag_system.search_cards_by_keyword(card_type.lower(), top_k=num_cards * 2)
            
            # Filter by color identity and type
            for card in cards:
                # Check if it's the right type
                if "type_line" not in card or card_type not in card["type_line"]:
                    continue
                    
                # Check color identity
                if not self.card_matches_identity(card, color_identity):
                    continue
                    
                # Check budget
                if max_price is not None and not self.card_within_budget(card, max_price):
                    continue
                    
                other_cards.append(card["name"])
                
                if len(other_cards) >= count:
                    break
            
            if len(other_cards) >= count:
                break
        
        # If we still need more cards, get some generic good cards
        if len(other_cards) < count:
            query = f"type:instant OR type:sorcery {strategy}"
            
            if self.rag_system.embedding_available and self.rag_system.text_index is not None:
                cards = self.rag_system.retrieve_cards_by_text(query, top_k=(count - len(other_cards)) * 2)
            else:
                cards = self.rag_system.search_cards_by_keyword("instant", top_k=(count - len(other_cards)) * 2)
            
            # Filter by color identity
            for card in cards:
                # Skip if already in list
                if card["name"] in other_cards:
                    continue
                    
                # Check color identity
                if not self.card_matches_identity(card, color_identity):
                    continue
                    
                # Check budget
                if max_price is not None and not self.card_within_budget(card, max_price):
                    continue
                    
                other_cards.append(card["name"])
                
                if len(other_cards) >= count:
                    break
        
        # Shuffle to randomize selection
        random.shuffle(other_cards)
        
        return other_cards[:count]
    
    def count_lands(self, deck: Dict[str, int]) -> int:
        """Count the number of lands in the deck"""
        land_count = 0
        
        # Check each card
        for card, qty in deck.items():
            # Basic lands
            if card in ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes']:
                land_count += qty
                continue
            
            # Check if the card is a land
            response = self.db.client.table("mtg_cards").select(
                "type_line"
            ).eq("name", card).limit(1).execute()
            
            if response.data and "Land" in response.data[0].get("type_line", ""):
                land_count += qty
        
        return land_count
    
    def count_card_type(self, deck: Dict[str, int], card_type: str) -> int:
        """Count the number of cards of a specific type in the deck"""
        type_count = 0
        
        # Check each card
        for card, qty in deck.items():
            # Skip basic lands
            if card in ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes']:
                if card_type == "Land":
                    type_count += qty
                continue
            
            # Check if the card is of the specified type
            response = self.db.client.table("mtg_cards").select(
                "type_line"
            ).eq("name", card).limit(1).execute()
            
            if response.data and card_type in response.data[0].get("type_line", ""):
                type_count += qty
        
        return type_count
    
    def optimize_for_cedh(self, deck: Dict[str, int], commander_identity: List[str]) -> Dict[str, int]:
        """Optimize deck for competitive EDH"""
        # Define cEDH staples by color
        cedh_staples = {
            'W': ["Swords to Plowshares", "Path to Exile", "Enlightened Tutor", "Smothering Tithe"],
            'U': ["Force of Will", "Mana Drain", "Rhystic Study", "Cyclonic Rift"],
            'B': ["Demonic Tutor", "Vampiric Tutor", "Necropotence", "Dark Ritual"],
            'R': ["Jeska's Will", "Dockside Extortionist", "Deflecting Swat"],
            'G': ["Birds of Paradise", "Elves of Deep Shadow", "Carpet of Flowers"],
            'C': ["Mana Crypt", "Chrome Mox", "Mox Diamond", "Sol Ring", "Command Tower"]
        }
        
        # Identify staples based on color identity
        relevant_staples = []
        for color in commander_identity:
            if color in cedh_staples:
                relevant_staples.extend(cedh_staples[color])
        
        # Add colorless staples
        relevant_staples.extend(cedh_staples['C'])
        
        # Remove duplicates
        relevant_staples = list(set(relevant_staples))
        
        # Add staples to deck
        optimized_deck = dict(deck)
        for staple in relevant_staples:
            if staple not in optimized_deck:
                # Verify the card exists and matches color identity
                response = self.db.client.table("mtg_cards").select(
                    "color_identity"
                ).eq("name", staple).limit(1).execute()
                
                if response.data:
                    card_identity = response.data[0].get('color_identity', [])
                    if all(color in commander_identity for color in card_identity):
                        optimized_deck[staple] = 1
        
        # Ensure we still have 100 cards
        total_cards = sum(optimized_deck.values())
        if total_cards > 100:
            # Remove non-staple cards until we have 100
            cards_to_remove = []
            for card, qty in optimized_deck.items():
                if card not in relevant_staples:
                    for _ in range(qty):
                        cards_to_remove.append(card)
            
            # Shuffle to randomize which cards get removed
            random.shuffle(cards_to_remove)
            
            # Remove excess cards
            excess = total_cards - 100
            for i in range(excess):
                if i < len(cards_to_remove):
                    card = cards_to_remove[i]
                    if optimized_deck[card] > 1:
                        optimized_deck[card] -= 1
                    else:
                        del optimized_deck[card]
        
        return optimized_deck
    
    def filter_deck_for_bracket(self, deck: Dict[str, int], bracket: int) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Filter a deck to comply with the chosen bracket's restrictions
        
        Args:
            deck: Dictionary of cards and quantities
            bracket: Bracket to filter for (1-5)
            
        Returns:
            Tuple of (filtered_deck, removed_cards)
        """
        # If bracket is 4 or 5, no restrictions apply
        if bracket >= 4:
            return deck, {}
        
        filtered_deck = {}
        removed_cards = {}
        
        # Define banned cards per bracket
        banned_cards = {
            1: ["Mana Crypt", "Mana Vault", "Force of Will"],  # Exhibition
            2: ["Mana Crypt", "Demonic Tutor", "Dockside Extortionist"],  # Core
            3: ["Mana Crypt"]  # Upgraded
        }
        
        # Get relevant banned list
        bracket_banned = banned_cards.get(bracket, [])
        
        # Filter cards
        for card, qty in deck.items():
            if card in bracket_banned:
                removed_cards[card] = qty
            else:
                filtered_deck[card] = qty
        
        return filtered_deck, removed_cards
    
    def replace_removed_cards(self, deck: Dict[str, int], removed_cards: Dict[str, int], 
                           commander_identity: List[str], bracket: int,
                           max_price: Optional[float] = None) -> Dict[str, int]:
        """
        Replace cards that were removed due to bracket restrictions
        
        Args:
            deck: Current deck
            removed_cards: Cards that were removed
            commander_identity: Color identity of the commander
            bracket: Bracket to build for (1-5)
            max_price: Maximum price for cards
            
        Returns:
            Updated deck with replacement cards
        """
        updated_deck = dict(deck)
        
        # Calculate how many cards we need to replace
        cards_to_add = sum(removed_cards.values())
        
        if cards_to_add <= 0:
            return updated_deck
        
        # Find replacements
        replacements = []
        
        # Replace ramp with other ramp
        ramp_count = 0
        for card in removed_cards:
            if card in ["Mana Crypt", "Mana Vault", "Sol Ring"]:
                ramp_count += removed_cards[card]
        
        if ramp_count > 0:
            ramp_replacements = self.find_ramp(commander_identity, bracket, max_price, ramp_count)
            replacements.extend(ramp_replacements)
        
        # Replace tutors with card draw
        tutor_count = 0
        for card in removed_cards:
            if "Tutor" in card:
                tutor_count += removed_cards[card]
        
        if tutor_count > 0:
            draw_replacements = self.find_card_draw(commander_identity, bracket, max_price, tutor_count)
            replacements.extend(draw_replacements)
        
        # Replace remaining cards with appropriate types
        remaining = cards_to_add - len(replacements)
        if remaining > 0:
            other_replacements = self.find_other_cards("good value", commander_identity, max_price, remaining)
            replacements.extend(other_replacements)
        
        # Add replacements to deck
        for card in replacements:
            if card in updated_deck:
                updated_deck[card] += 1
            else:
                updated_deck[card] = 1
        
        return updated_deck
    
    def find_ramp(self, color_identity: List[str], bracket: int, 
                max_price: Optional[float], count: int) -> List[str]:
        """Find appropriate ramp for the bracket"""
        # Define ramp options by bracket
        ramp_options = {
            1: ["Arcane Signet", "Commander's Sphere", "Wayfarer's Bauble"],  # Exhibition
            2: ["Arcane Signet", "Talisman of Progress", "Fellwar Stone"],    # Core
            3: ["Arcane Signet", "Fellwar Stone", "Mind Stone"]               # Upgraded
        }
        
        bracket_ramp = ramp_options.get(bracket, ["Arcane Signet", "Sol Ring"])
        
        # Search for more ramp options
        query = "mana rock OR ramp"
        
        if self.rag_system.embedding_available and self.rag_system.text_index is not None:
            cards = self.rag_system.retrieve_cards_by_text(query, top_k=count * 2)
        else:
            cards = self.rag_system.search_cards_by_keyword("ramp", top_k=count * 2)
        
        # Filter by color identity and budget
        ramp_cards = []
        
        # Start with predefined options
        for card in bracket_ramp:
            if len(ramp_cards) >= count:
                break
                
            # Verify the card exists and matches color identity
            response = self.db.client.table("mtg_cards").select(
                "color_identity, prices_usd"
            ).eq("name", card).limit(1).execute()
            
            if response.data:
                card_identity = response.data[0].get('color_identity', [])
                
                # Check color identity
                if not all(color in color_identity for color in card_identity):
                    continue
                    
                # Check budget
                if max_price is not None:
                    price = response.data[0].get('prices_usd')
                    if price is not None:
                        try:
                            if float(price) > max_price:
                                continue
                        except (ValueError, TypeError):
                            pass
                
                ramp_cards.append(card)
        
        # Add more from search results
        for card in cards:
            if len(ramp_cards) >= count:
                break
                
            # Skip if already in list
            if card["name"] in ramp_cards:
                continue
                
            # Check color identity
            if not self.card_matches_identity(card, color_identity):
                continue
                
            # Check budget
            if max_price is not None and not self.card_within_budget(card, max_price):
                continue
                
            ramp_cards.append(card["name"])
        
        # Return up to 'count' cards
        return ramp_cards[:count]
    
    def find_card_draw(self, color_identity: List[str], bracket: int, 
                     max_price: Optional[float], count: int) -> List[str]:
        """Find appropriate card draw for the bracket"""
        query = "draw cards"
        
        if self.rag_system.embedding_available and self.rag_system.text_index is not None:
            cards = self.rag_system.retrieve_cards_by_text(query, top_k=count * 2)
        else:
            cards = self.rag_system.search_cards_by_keyword("draw", top_k=count * 2)
        
        # Filter by color identity and budget
        draw_cards = []
        for card in cards:
            # Skip if already in list
            if card["name"] in draw_cards:
                continue
                
            # Check color identity
            if not self.card_matches_identity(card, color_identity):
                continue
                
            # Check budget
            if max_price is not None and not self.card_within_budget(card, max_price):
                continue
                
            draw_cards.append(card["name"])
            
            if len(draw_cards) >= count:
                break
        
        # Return up to 'count' cards
        return draw_cards[:count]
    
    def calculate_deck_price(self, deck: Dict[str, int]) -> float:
        """
        Calculate the total price of a deck
        
        Args:
            deck: Dictionary of cards and quantities
            
        Returns:
            Total price of the deck
        """
        total_price = 0
        
        for card, qty in deck.items():
            # Get card price from database
            response = self.db.client.table("mtg_cards").select(
                "prices_usd"
            ).eq("name", card).limit(1).execute()
            
            if response.data:
                price = response.data[0].get('prices_usd')
                if price is not None:
                    try:
                        total_price += float(price) * qty
                    except (ValueError, TypeError):
                        pass
        
        return round(total_price, 2)
    
    def format_deck(self, deck: Dict[str, int], commander_name: str, strategy: str, 
                  bracket: int, total_price: float) -> Dict[str, Any]:
        """
        Format deck for database storage
        
        Args:
            deck: Dictionary of cards and quantities
            commander_name: Name of the commander
            strategy: Deck strategy
            bracket: Power level bracket
            total_price: Total price of the deck
            
        Returns:
            Formatted deck dictionary
        """
        # Group cards by type
        deck_by_type = {
            "Commander": [],
            "Creatures": [],
            "Artifacts": [],
            "Enchantments": [],
            "Instants": [],
            "Sorceries": [],
            "Planeswalkers": [],
            "Lands": [],
            "Other": []
        }
        
        # Categorize cards
        for card, qty in deck.items():
            # Check if this is a face of a dual-faced card
            main_card = self.flip_card_mapping.get(card, card)
            
            # Use the main card name for categorization
            if main_card == commander_name:
                deck_by_type["Commander"] = [main_card] * qty
                continue
                
            # Basic lands
            if card in ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes']:
                deck_by_type["Lands"].extend([card] * qty)
                continue
                
            # Get card type from database
            response = self.db.client.table("mtg_cards").select(
                "type_line"
            ).eq("name", card).limit(1).execute()
            
            if not response.data:
                continue
                
            type_line = response.data[0].get("type_line", "")
            
            # Determine card type
            card_type = self.determine_card_type(type_line)
            
            # Add to appropriate category
            deck_by_type[card_type].extend([card] * qty)
        
        # Count totals by type
        type_counts = {
            card_type: len(cards) for card_type, cards in deck_by_type.items()
        }
        
        # Format bracket info
        bracket_info = {
            "number": bracket,
            "name": COMMANDER_BRACKETS[bracket]['name'],
            "description": COMMANDER_BRACKETS[bracket]['description']
        }
        
        # Create formatted deck
        formatted_deck = {
            "commander": commander_name,
            "strategy": strategy,
            "bracket": bracket_info,
            "total_price": total_price,
            "deck_list": deck_by_type,
            "type_counts": type_counts,
            "total_cards": sum(type_counts.values())
        }
        # Add a count of unique physical cards
        formatted_deck["unique_physical_cards"] = self.count_unique_physical_cards(deck)
        
        return formatted_deck
    
    def determine_card_type(self, type_line: str) -> str:
        """Determine primary card type based on type line"""
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
    
    def save_deck(self, formatted_deck: Dict[str, Any]) -> str:
        """
        Save deck to database and local file
        
        Args:
            formatted_deck: Formatted deck dictionary
            
        Returns:
            Deck ID
        """
        try:
            # Create a simplified version for database
            db_deck = {
                "name": formatted_deck["commander"],
                "description": formatted_deck["strategy"],
                "decklist": formatted_deck["deck_list"],
                "total_price": formatted_deck["total_price"],
                "bracket": formatted_deck["bracket"]["number"]
            }
            
            # Insert into database
            response = self.db.client.table("saved_decks").insert(db_deck).execute()
            
            if response.data:
                deck_id = response.data[0]['id']
                
                # Save full deck to file
                file_path = os.path.join('data/generated_decks', f"{deck_id}.json")
                with open(file_path, 'w') as f:
                    json.dump(formatted_deck, f, indent=2)
                
                logger.info(f"Saved deck {deck_id} to database and file")
                return deck_id
            else:
                logger.warning("Failed to save deck to database")
                
                # Create a temporary ID
                import time
                temp_id = f"temp-{int(time.time())}"
                
                # Save full deck to file
                file_path = os.path.join('data/generated_decks', f"{temp_id}.json")
                with open(file_path, 'w') as f:
                    json.dump(formatted_deck, f, indent=2)
                
                logger.info(f"Saved deck {temp_id} to file")
                return temp_id
                
        except Exception as e:
            logger.error(f"Error saving deck: {e}")
            
            # Create a temporary ID on error
            import time
            temp_id = f"temp-{int(time.time())}"
            
            # Save full deck to file
            file_path = os.path.join('data/generated_decks', f"{temp_id}.json")
            with open(file_path, 'w') as f:
                json.dump(formatted_deck, f, indent=2)
            
            logger.info(f"Saved deck {temp_id} to file")
            return temp_id