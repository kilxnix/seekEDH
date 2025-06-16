# rag_wrapper.py - Wrapper functions to make deck_generator use RAG system
import logging
import numpy as np
import json
import random
from typing import List, Dict, Any, Optional
from src.rag_instance import rag_system

logger = logging.getLogger("RAGWrapper")

# Global card dictionary populated from RAG system (for compatibility)
card_dict = {}
flip_card_mapping = {}

# Define basic lands mapping
BASIC_LANDS = {
    'W': 'Plains',
    'U': 'Island', 
    'B': 'Swamp',
    'R': 'Mountain',
    'G': 'Forest',
    'C': 'Wastes'
}

# Commander banned cards (essential for validation)
COMMANDER_BANNED_CARDS = [
    "Ancestral Recall", "Balance", "Biorhythm", "Black Lotus", "Braids, Cabal Minion", 
    "Channel", "Chaos Orb", "Coalition Victory", "Dockside Extortionist", 
    "Emrakul, the Aeons Torn", "Erayo, Soratami Ascendant", "Falling Star",
    "Fastbond", "Flash", "Gifts Ungiven", "Golos, Tireless Pilgrim", 
    "Griselbrand", "Hullbreacher", "Iona, Shield of Emeria", "Jeweled Lotus", 
    "Karakas", "Leovold, Emissary of Trest", "Library of Alexandria", 
    "Limited Resources", "Lutri, the Spellchaser", "Mana Crypt", "Mox Emerald", 
    "Mox Jet", "Mox Pearl", "Mox Ruby", "Mox Sapphire", "Nadu, Winged Wisdom", 
    "Panoptic Mirror", "Paradox Engine", "Primeval Titan", "Prophet of Kruphix", 
    "Recurring Nightmare", "Rofellos, Llanowar Emissary", "Shahrazad", 
    "Sundering Titan", "Sway of the Stars", "Sylvan Primordial", "Time Vault", 
    "Time Walk", "Tinker", "Tolarian Academy", "Trade Secrets", "Upheaval", 
    "Yawgmoth's Bargain"
]

def initialize_rag_deck_system():
    """Initialize the deck generation system using RAG backend"""
    global card_dict, flip_card_mapping
    
    try:
        if not rag_system.db.is_connected:
            logger.error("RAG system not connected to database")
            return False
        
        if rag_system.text_index is None:
            logger.warning("Text index not initialized in RAG system")
            # Try to initialize it
            rag_system._initialize_text_index()
        
        logger.info("RAG-based deck generation system ready")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing RAG deck system: {e}")
        return False

def get_commander_identity(commander_name: str) -> List[str]:
    """Get commander color identity using RAG system"""
    try:
        response = rag_system.db.client.table("mtg_cards").select(
            "color_identity, type_line"
        ).eq("name", commander_name).limit(1).execute()
        
        if response.data:
            card = response.data[0]
            # Verify it's a legendary creature
            type_line = card.get('type_line', '').lower()
            if 'legendary' in type_line and 'creature' in type_line:
                return card.get('color_identity', [])
            else:
                logger.warning(f"{commander_name} is not a legendary creature")
                return []
        else:
            logger.warning(f"Commander '{commander_name}' not found in database")
            return []
            
    except Exception as e:
        logger.error(f"Error getting commander identity: {e}")
        return []

def get_card_info(card_name: str) -> Optional[Dict[str, Any]]:
    """Get card information from RAG database"""
    try:
        response = rag_system.db.client.table("mtg_cards").select(
            "id, scryfall_id, name, oracle_text, type_line, mana_cost, cmc, colors, color_identity, "
            "power, toughness, loyalty, rarity, prices_usd, prices_usd_foil, legalities"
        ).eq("name", card_name).limit(1).execute()
        
        if response.data:
            card = response.data[0]
            
            # Parse legalities if it's a string
            legalities = card.get('legalities', {})
            if isinstance(legalities, str):
                try:
                    legalities = json.loads(legalities)
                except:
                    legalities = {}
            
            # Convert to deck_generator expected format
            formatted_card = {
                'id': card.get('scryfall_id') or card.get('id'),
                'name': card.get('name'),
                'oracle_text': card.get('oracle_text', ''),
                'type_line': card.get('type_line', ''),
                'mana_cost': card.get('mana_cost', ''),
                'mana_value': card.get('cmc', 0),
                'cmc': card.get('cmc', 0),
                'colors': card.get('colors', []),
                'color_identity': card.get('color_identity', []),
                'power': card.get('power'),
                'toughness': card.get('toughness'),
                'loyalty': card.get('loyalty'),
                'rarity': card.get('rarity'),
                'price_usd': card.get('prices_usd'),
                'price_usd_foil': card.get('prices_usd_foil'),
                'legalities': legalities,
                'types': extract_types_from_type_line(card.get('type_line', ''))
            }
            
            # Add to global card_dict for deck_generator compatibility
            card_dict[card_name] = formatted_card
            return formatted_card
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting card info for {card_name}: {e}")
        return None

def search_cards_by_criteria(query: str, commander_identity: List[str], exclude_cards: Optional[List[str]] = None, 
                           n: int = 30, price_data: Optional[Dict] = None, max_price: Optional[float] = None) -> List[str]:
    """Search for cards using RAG system with deck_generator compatibility"""
    exclude_cards = exclude_cards or []
    
    try:
        # Use RAG system's text search
        similar_cards = rag_system.retrieve_cards_by_text(query, top_k=n * 3)  # Get extra for filtering
        
        if not similar_cards:
            # Fallback to keyword search
            similar_cards = rag_system.search_cards_by_keyword(query, top_k=n * 2)
        
        filtered_results = []
        for card in similar_cards:
            card_name = card.get('name')
            if not card_name or card_name in exclude_cards or card_name in COMMANDER_BANNED_CARDS:
                continue
            
            # Get full card info to check constraints
            card_info = get_card_info(card_name)
            if not card_info:
                continue
            
            # Check color identity
            card_colors = card_info.get('color_identity', [])
            if not all(c in commander_identity for c in card_colors):
                continue
            
            # Check price constraint
            if max_price is not None:
                card_price = card_info.get('price_usd')
                if card_price and float(card_price) > max_price:
                    continue
            
            # Check legality (Commander format)
            legalities = card_info.get('legalities', {})
            if isinstance(legalities, dict):
                commander_legal = legalities.get('commander', 'not_legal')
                if commander_legal not in ['legal', 'restricted']:
                    continue
            
            filtered_results.append(card_name)
            
            if len(filtered_results) >= n:
                break
        
        return filtered_results
        
    except Exception as e:
        logger.error(f"Error searching cards by criteria: {e}")
        return []

def validate_deck_identity(deck: Dict[str, int], commander_identity: List[str]) -> tuple:
    """Validate deck color identity using RAG system"""
    valid_deck = {}
    invalid_cards = []
    
    for card_name, quantity in deck.items():
        # Skip basic lands - they're always valid
        if card_name in BASIC_LANDS.values():
            valid_deck[card_name] = quantity
            continue
            
        card_info = get_card_info(card_name)
        
        if not card_info:
            invalid_cards.append(f"{card_name} (not found)")
            continue
        
        # Check banned list
        if card_name in COMMANDER_BANNED_CARDS:
            invalid_cards.append(f"{card_name} (banned)")
            continue
        
        # Check color identity
        card_colors = card_info.get('color_identity', [])
        if not all(c in commander_identity for c in card_colors):
            invalid_cards.append(f"{card_name} (color identity)")
            continue
        
        # Check Commander legality
        legalities = card_info.get('legalities', {})
        if isinstance(legalities, dict):
            commander_legal = legalities.get('commander', 'not_legal')
            if commander_legal == 'banned':
                invalid_cards.append(f"{card_name} (banned)")
                continue
        
        valid_deck[card_name] = quantity
    
    return valid_deck, invalid_cards

def search_lands_by_quality(commander_identity: List[str], land_quality: str, 
                          exclude_cards: Optional[List[str]] = None, count: int = 10,
                          price_data: Optional[Dict] = None, max_price: Optional[float] = None) -> List[str]:
    """Search for lands using RAG system"""
    exclude_cards = exclude_cards or []
    
    try:
        # Build land search query based on quality
        if land_quality == 'competitive':
            queries = ["land enters untapped", "shockland", "fetchland", "dual land"]
        elif land_quality == 'budget':
            queries = ["basic land", "enters tapped", "guild gate", "temple"]
        else:  # balanced
            queries = ["land mana", "dual land", "utility land"]
        
        # Search for lands
        land_results = []
        
        for query in queries:
            if len(land_results) >= count:
                break
                
            lands = rag_system.retrieve_cards_by_text(query, top_k=count * 2)
            
            for land in lands:
                land_name = land.get('name')
                if not land_name or land_name in exclude_cards or land_name in land_results:
                    continue
                
                # Verify it's actually a land
                land_info = get_card_info(land_name)
                if not land_info:
                    continue
                
                types = land_info.get('types', [])
                if 'Land' not in types:
                    continue
                
                # Check color identity
                land_colors = land_info.get('color_identity', [])
                if land_colors and not all(c in commander_identity for c in land_colors):
                    continue
                
                # Check price
                if max_price is not None:
                    land_price = land_info.get('price_usd')
                    if land_price and float(land_price) > max_price:
                        continue
                
                land_results.append(land_name)
                
                if len(land_results) >= count:
                    break
        
        # Add basic lands if needed
        basic_mapping = {'W': 'Plains', 'U': 'Island', 'B': 'Swamp', 'R': 'Mountain', 'G': 'Forest'}
        
        for color in commander_identity:
            if color in basic_mapping:
                basic_land = basic_mapping[color]
                if basic_land not in land_results and len(land_results) < count:
                    land_results.append(basic_land)
        
        return land_results
        
    except Exception as e:
        logger.error(f"Error searching lands: {e}")
        return []

def analyze_commander_for_strategies(commander_name: str, card_dict_unused: Dict, 
                                   price_data: Optional[Dict] = None, max_price: Optional[float] = None) -> List[Dict]:
    """Analyze commander using RAG system"""
    try:
        commander_info = get_card_info(commander_name)
        if not commander_info:
            return []
        
        oracle_text = commander_info.get('oracle_text', '').lower()
        color_identity = commander_info.get('color_identity', [])
        
        # Use RAG to find synergistic cards
        synergy_query = f"synergy with {commander_name} {oracle_text[:100]}"
        synergistic_cards = search_cards_by_criteria(
            synergy_query, color_identity, n=50, price_data=price_data, max_price=max_price
        )
        
        # Create strategies based on commander text analysis
        strategies = []
        
        # Analyze commander text for themes
        themes = {}
        theme_keywords = {
            'token': ['token', 'create', 'creature token'],
            'counter': ['counter', '+1/+1', '-1/-1'],
            'sacrifice': ['sacrifice', 'dies', 'death'],
            'graveyard': ['graveyard', 'return', 'from your graveyard'],
            'draw': ['draw', 'card', 'cards'],
            'damage': ['damage', 'deals damage', 'ping'],
            'artifact': ['artifact', 'equipment'],
            'enchantment': ['enchantment', 'aura'],
            'tribal': ['creature type', 'share a creature type']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in oracle_text for keyword in keywords):
                theme_cards = search_cards_by_criteria(
                    f"{theme} synergy", color_identity, n=20, price_data=price_data, max_price=max_price
                )
                if theme_cards:
                    themes[theme] = theme_cards
        
        # Convert themes to strategies
        theme_names = {
            'token': 'Token Swarm',
            'counter': 'Counter Synergy', 
            'sacrifice': 'Aristocrats',
            'graveyard': 'Graveyard Value',
            'draw': 'Card Advantage',
            'damage': 'Direct Damage',
            'artifact': 'Artifact Synergy',
            'enchantment': 'Enchantress',
            'tribal': 'Tribal Synergy'
        }
        
        for theme, cards in themes.items():
            if len(cards) >= 5:
                strategies.append({
                    'name': theme_names.get(theme, f"{theme.capitalize()} Strategy"),
                    'description': f"Build around {commander_name}'s {theme} synergies",
                    'key_cards': cards[:5],
                    'power_level': 6 + min(len(cards) // 5, 3)
                })
        
        # Add default strategy if none found
        if not strategies:
            strategies.append({
                'name': 'Goodstuff',
                'description': f"Balanced deck using best cards in {commander_name}'s colors",
                'key_cards': synergistic_cards[:5],
                'power_level': 6
            })
        
        return strategies
        
    except Exception as e:
        logger.error(f"Error analyzing commander strategies: {e}")
        return []

def get_price_data() -> Dict[str, Dict]:
    """Get price data from RAG database"""
    try:
        response = rag_system.db.client.table("mtg_cards").select(
            "name, prices_usd, prices_usd_foil"
        ).not_.is_("prices_usd", "null").limit(10000).execute()
        
        price_data = {}
        for card in response.data:
            price_data[card['name']] = {
                'usd': card.get('prices_usd'),
                'usd_foil': card.get('prices_usd_foil')
            }
        
        return price_data
        
    except Exception as e:
        logger.error(f"Error getting price data: {e}")
        return {}

def extract_types_from_type_line(type_line: str) -> List[str]:
    """Extract card types from type line"""
    types = []
    type_line = type_line or ""
    
    for card_type in ['Land', 'Creature', 'Artifact', 'Enchantment', 'Planeswalker', 'Instant', 'Sorcery']:
        if card_type in type_line:
            types.append(card_type)
    
    return types

def save_deck_to_database(deck_data: Dict) -> bool:
    """Save deck to database using RAG system"""
    try:
        response = rag_system.db.client.table("saved_decks").insert(deck_data).execute()
        return True
    except Exception as e:
        logger.error(f"Error saving deck: {e}")
        return False

def get_commander_themes(commander_name: str) -> List[str]:
    """Get themes for a commander using RAG system"""
    try:
        commander_info = get_card_info(commander_name)
        if not commander_info:
            return []
        
        oracle_text = commander_info.get('oracle_text', '').lower()
        themes = []
        
        if 'draw' in oracle_text or 'card' in oracle_text:
            themes.append("card draw")
        if 'damage' in oracle_text:
            themes.append("damage")
        if 'counter' in oracle_text:
            themes.append("counters")
        if 'token' in oracle_text:
            themes.append("tokens")
        if 'graveyard' in oracle_text:
            themes.append("graveyard")
        if 'sacrifice' in oracle_text:
            themes.append("sacrifice")
        if 'discard' in oracle_text:
            themes.append("discard")
        
        return themes
        
    except Exception as e:
        logger.error(f"Error getting commander themes: {e}")
        return []

def generate_synergy_queries(commander_name: str) -> List[str]:
    """Generate synergy queries based on commander"""
    try:
        themes = get_commander_themes(commander_name)
        synergy_queries = []
        
        # Add commander name for direct synergy
        synergy_queries.append(f"synergy with {commander_name}")
        
        # Add theme-based queries
        for theme in themes:
            synergy_queries.append(f"cards that work with {theme}")
        
        return synergy_queries
        
    except Exception as e:
        logger.error(f"Error generating synergy queries: {e}")
        return [f"synergy with {commander_name}"]

# Compatibility functions for deck_generator
def load_card_data():
    """Compatibility function - returns empty list since we use RAG"""
    logger.info("Using RAG system instead of loading card data")
    return []

def setup_retriever(cards):
    """Compatibility function - RAG system handles this"""
    logger.info("Using RAG system retriever")
    pass

def search_similar_cards(query: str, n: int = 10) -> List[str]:
    """Search for similar cards using RAG system"""
    try:
        results = rag_system.retrieve_cards_by_text(query, top_k=n)
        return [card.get('name') for card in results if card.get('name')]
    except Exception as e:
        logger.error(f"Error searching similar cards: {e}")
        return []

def count_unique_physical_cards(deck: Dict[str, int]) -> int:
    """Count unique physical cards (accounting for flip cards)"""
    # Simple implementation since RAG doesn't use flip card mapping yet
    return sum(deck.values())

def count_card_types(deck: Dict[str, int]) -> Dict[str, int]:
    """Count cards by type using RAG system"""
    type_counts = {'Land': 0, 'Creature': 0, 'Artifact': 0,
                  'Enchantment': 0, 'Planeswalker': 0,
                  'Instant': 0, 'Sorcery': 0, 'Other': 0}
    
    for card_name, qty in deck.items():
        if card_name in BASIC_LANDS.values():
            type_counts['Land'] += qty
            continue
            
        card_info = get_card_info(card_name)
        if not card_info:
            type_counts['Other'] += qty
            continue
        
        types = card_info.get('types', [])
        
        # Count by primary type
        if 'Land' in types:
            type_counts['Land'] += qty
        elif 'Creature' in types:
            type_counts['Creature'] += qty
        elif 'Artifact' in types:
            type_counts['Artifact'] += qty
        elif 'Enchantment' in types:
            type_counts['Enchantment'] += qty
        elif 'Planeswalker' in types:
            type_counts['Planeswalker'] += qty
        elif 'Instant' in types:
            type_counts['Instant'] += qty
        elif 'Sorcery' in types:
            type_counts['Sorcery'] += qty
        else:
            type_counts['Other'] += qty
    
    return type_counts

# Global variables for compatibility with deck_generator
faiss_index = None  # Not needed with RAG
embed_model = None  # RAG system handles this
card_texts = []     # Not needed with RAG
card_objects = []   # Not needed with RAG