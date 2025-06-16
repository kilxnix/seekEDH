# MTG Deck Generator with Comprehensive Rules, Dual-Faced Card Support, and Validation
# Enhanced for Commander Deck Building Guidelines with Price Integration
# Run this in Google Colab

import os
import json
import logging
import dotenv
from getpass import getpass
from supabase import create_client, Client
import requests
import torch
import faiss
import numpy as np
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
card_dict = {}
flip_card_mapping = {}
faiss_index, embed_model, card_texts, card_objects = None, None, [], []

# Official Commander banned cards list
COMMANDER_BANNED_CARDS = ["Ancestral Recall", "Balance", "Biorhythm", "Black Lotus", "Braids, Cabal Minion", "Channel", "Chaos Orb",
"Coalition Victory", "Dockside Extortionist", "Emrakul, the Aeons Torn", "Erayo, Soratami Ascendant", "Falling Star",
"Fastbond", "Flash", "Gifts Ungiven", "Golos, Tireless Pilgrim", "Griselbrand", "Hullbreacher", "Iona, Shield of Emeria",
"Jeweled Lotus", "Karakas", "Leovold, Emissary of Trest", "Library of Alexandria", "Limited Resources", "Lutri, the Spellchaser",
"Mana Crypt", "Mox Emerald", "Mox Jet", "Mox Pearl", "Mox Ruby", "Mox Sapphire", "Nadu, Winged Wisdom", "Panoptic Mirror",
"Paradox Engine", "Primeval Titan", "Prophet of Kruphix", "Recurring Nightmare", "Rofellos, Llanowar Emissary", "Shahrazad",
"Sundering Titan", "Sway of the Stars", "Sylvan Primordial", "Time Vault", "Time Walk", "Tinker", "Tolarian Academy",
"Trade Secrets", "Upheaval", "Yawgmoth's Bargain"]

# Basic lands mapping
BASIC_LANDS = {
    'W': 'Plains',
    'U': 'Island',
    'B': 'Swamp',
    'R': 'Mountain',
    'G': 'Forest',
    'C': 'Wastes'
}

# Card categories for search queries
CARD_CATEGORIES = {
    'ramp': ['type:artifact text:add mana', 'type:land text:add mana', 'type:creature text:"add mana"'],
    'draw': ['text:"draw a card"', 'text:"draw cards"'],
    'removal': ['text:destroy', 'text:exile', 'text:"damage to"'],
    'wrath': ['text:"destroy all"', 'text:"exile all"', 'text:"damage to all"'],
    'synergy': []  # Will be populated based on commander
}

# Define Commander brackets for power level targeting
COMMANDER_BRACKETS = {
    1: {
        'name': 'Exhibition',
        'description': 'Heavily themed decks where winning is not the primary goal. Focus on showing off a concept or project.',
        'restrictions': [
            'No Mass Land Denial',
            'No Game Changer Cards',
            'No 2-card Combos',
            'No Extra Turns spells',
            'Tutors should be sparse and specific'
        ],
        'power_level': (1, 3)  # Power range 1-3
    },
    2: {
        'name': 'Core',
        'description': 'The bulk of casual decks and modern precons. Draw power from synergy rather than card quality.',
        'restrictions': [
            'No Mass Land Denial',
            'No Game Changer Cards',
            'No 2-card Combos',
            'Extra Turns spells should be minimal and not intended to be chained or looped',
            'Tutors should be sparse and specific'
        ],
        'power_level': (3, 5)  # Power range 3-5
    },
    3: {
        'name': 'Upgraded',
        'description': 'Intentionally tuned and upgraded decks with improved power level. Theme and flavor take a back seat.',
        'restrictions': [
            'No Mass Land Denial',
            'Up to 3 Game Changer Cards',
            'No 2-card Combos before turn 6',
            'Some extra turns and tutors expected (but not looped)'
        ],
        'power_level': (5, 7)  # Power range 5-7
    },
    4: {
        'name': 'Optimized',
        'description': 'High-powered decks using any cards and strategies, though not meta-focused or tournament driven.',
        'restrictions': [
            'None (other than the banned list)'
        ],
        'power_level': (7, 9)  # Power range 7-9
    },
    5: {
        'name': 'cEDH',
        'description': 'Competitive EDH decks designed for tournament play, making choices dependent on the competitive meta.',
        'restrictions': [
            'None (other than the banned list)'
        ],
        'power_level': (9, 10)  # Power range 9-10
    }
}

# Official Game Changer cards list per the Commander Brackets system
GAME_CHANGER_CARDS = [
    # White
    "Drannith Magistrate", "Enlightened Tutor", "Serra's Sanctum", "Smothering Tithe",
    "Trouble in Pairs",
    # Blue
    "Cyclonic Rift", "Expropriate", "Force of Will", "Fierce Guardianship",
    "Rhystic Study", "Thassa's Oracle", "Urza, Lord High Artificer", "Mystical Tutor",
    "Jin-Gitaxias, Core Augur",
    # Black
    "Bolas's Citadel", "Demonic Tutor", "Imperial Seal", "Opposition Agent",
    "Tergrid, God of Fright", "Vampiric Tutor", "Ad Nauseam",
    # Red
    "Jeska's Will", "Underworld Breach",
    # Green
    "Survival of the Fittest", "Vorinclex, Voice of Hunger", "Gaea's Cradle",
    # Multicolor
    "Kinnan, Bonder Prodigy", "Yuriko, the Tiger's Shadow", "Winota, Joiner of Forces",
    "Grand Arbiter Augustin IV",
    # Colorless
    "Ancient Tomb", "Chrome Mox", "The One Ring", "The Tabernacle at Pendrell Vale",
    "Trinisphere", "Grim Monolith", "Lion's Eye Diamond", "Mox Diamond", "Mana Vault",
    "Glacial Chasm"
]

# Cards that qualify as mass land denial
MASS_LAND_DENIAL = [
    "Armageddon", "Ravages of War", "Catastrophe", "Wildfire", "Burning of Xinye",
    "Ruination", "Death Cloud", "Jokulhaups", "Obliterate", "Decree of Annihilation",
    "Fall of the Thran", "Impending Disaster", "Natural Balance", "Winter Orb",
    "Static Orb", "Rising Waters", "Stasis", "Blood Moon", "Magus of the Moon"
]

# Cards that provide extra turns
EXTRA_TURN_SPELLS = [
    "Time Warp", "Temporal Manipulation", "Capture of Jingzhou", "Time Stretch",
    "Nexus of Fate", "Temporal Mastery", "Walk the Aeons", "Karn's Temporal Sundering",
    "Part the Waterveil", "Temporal Trespass", "Alrund's Epiphany", "Expropriate",
    "Plea for Power"
]

# Cards that tutor for other cards
TUTOR_CARDS = [
    "Demonic Tutor", "Vampiric Tutor", "Worldly Tutor", "Mystical Tutor", "Enlightened Tutor",
    "Imperial Seal", "Diabolic Tutor", "Grim Tutor", "Personal Tutor", "Sylvan Tutor",
    "Demonic Consultation", "Tainted Pact", "Scheming Symmetry", "Cruel Tutor",
    "Diabolic Intent", "Gamble", "Chord of Calling", "Green Sun's Zenith", "Tooth and Nail",
    "Eladamri's Call", "Idyllic Tutor", "Beseech the Queen", "Final Parting"
]

# Known combo cards and their partners
COMBO_CARDS = {
    # Format: card_name: [combo_partner1, combo_partner2, ...]
    "Thassa's Oracle": ["Demonic Consultation", "Tainted Pact"],
    "Laboratory Maniac": ["Demonic Consultation", "Tainted Pact"],
    "Jace, Wielder of Mysteries": ["Demonic Consultation", "Tainted Pact"],
    "Isochron Scepter": ["Dramatic Reversal"],
    "Dramatic Reversal": ["Isochron Scepter"],
    "Splinter Twin": ["Deceiver Exarch", "Pestermite", "Zealous Conscripts"],
    "Kiki-Jiki, Mirror Breaker": ["Deceiver Exarch", "Pestermite", "Zealous Conscripts"],
    "Mikaeus, the Unhallowed": ["Triskelion", "Walking Ballista"],
    "Triskelion": ["Mikaeus, the Unhallowed"],
    "Walking Ballista": ["Mikaeus, the Unhallowed", "Heliod, Sun-Crowned"],
    "Heliod, Sun-Crowned": ["Walking Ballista"],
    "Sanguine Bond": ["Exquisite Blood"],
    "Exquisite Blood": ["Sanguine Bond"],
    "Painters Servant": ["Grindstone"],
    "Grindstone": ["Painters Servant"],
    "Doomsday": ["Thassa's Oracle", "Laboratory Maniac"],
    "Food Chain": ["Eternal Scourge", "Misthollow Griffin", "Squee, the Immortal"],
    "Demonic Consultation": ["Thassa's Oracle", "Laboratory Maniac", "Jace, Wielder of Mysteries"],
    "Tainted Pact": ["Thassa's Oracle", "Laboratory Maniac", "Jace, Wielder of Mysteries"],
    "Earthcraft": ["Squirrel Nest"],
    "Squirrel Nest": ["Earthcraft"]
}

# Land quality settings
LAND_QUALITY_SETTINGS = {
    'competitive': {
        'name': 'Competitive Lands',
        'description': 'Prioritize lands that enter untapped, even at higher financial cost (shocklands, fetchlands, etc.)',
        'allowed_tapped': False,
        'budget': 'high'
    },
    'balanced': {
        'name': 'Balanced Lands',
        'description': 'Mix of tapped and untapped lands with good utility',
        'allowed_tapped': True,
        'budget': 'medium'
    },
    'budget': {
        'name': 'Budget Lands',
        'description': 'Primarily basic lands with some budget tapped lands for utility',
        'allowed_tapped': True,
        'budget': 'low'
    }
}

# Lists to identify enter-tapped lands
ENTERS_TAPPED_LANDS = [
    # Common tap lands
    "Akoum Refuge", "Bloodfell Caves", "Blossoming Sands", "Bojuka Bog", "Cinder Glade",
    "Coastal Tower", "Dismal Backwater", "Evolving Wilds", "Exotic Orchard", "Frontier Bivouac",
    "Frostboil Snarl", "Furycalm Snarl", "Glacial Fortress", "Highland Lake", "Irrigated Farmland",
    "Jungle Hollow", "Meandering River", "Memorial to Genius", "Mystic Monastery", "Nomad Outpost",
    "Path of Ancestry", "Prairie Stream", "Rugged Highlands", "Sandsteppe Citadel", "Scoured Barrens",
    "Sejiri Refuge", "Stone Quarry", "Sunpetal Grove", "Swiftwater Cliffs", "Temple of Abandon",
    "Temple of Deceit", "Temple of Enlightenment", "Temple of Epiphany", "Temple of Malady",
    "Temple of Malice", "Temple of Mystery", "Temple of Plenty", "Temple of Silence", "Temple of Triumph",
    "Terramorphic Expanse", "Thornwood Falls", "Tranquil Cove", "Wind-Scarred Crag", "Woodland Stream",

    # Cycling lands
    "Barren Moor", "Desert of the Fervent", "Desert of the Glorified", "Desert of the Indomitable",
    "Desert of the Mindful", "Desert of the True", "Drifting Meadow", "Forgotten Cave", "Lonely Sandbar",
    "Remote Isle", "Secluded Steppe", "Slippery Karst", "Smoldering Crater", "Tranquil Thicket",

    # Common tap tri-lands
    "Arcane Sanctum", "Crumbling Necropolis", "Frontier Bivouac", "Jungle Shrine", "Mystic Monastery",
    "Nomad Outpost", "Opulent Palace", "Sandsteppe Citadel", "Savage Lands", "Seaside Citadel",

    # Bounce lands
    "Azorius Chancery", "Boros Garrison", "Dimir Aqueduct", "Golgari Rot Farm", "Gruul Turf",
    "Izzet Boilerworks", "Orzhov Basilica", "Rakdos Carnarium", "Selesnya Sanctuary", "Simic Growth Chamber",

    # Gain lands
    "Akoum Refuge", "Bloodfell Caves", "Blossoming Sands", "Dismal Backwater", "Graypelt Refuge",
    "Jungle Hollow", "Jwar Isle Refuge", "Kazandu Refuge", "Rugged Highlands", "Scoured Barrens",
    "Sejiri Refuge", "Swiftwater Cliffs", "Thornwood Falls", "Tranquil Cove", "Wind-Scarred Crag"
]

# List of premium lands that typically enter untapped
PREMIUM_UNTAPPED_LANDS = [
    # Shock lands
    "Breeding Pool", "Blood Crypt", "Godless Shrine", "Hallowed Fountain", "Overgrown Tomb",
    "Sacred Foundry", "Steam Vents", "Stomping Ground", "Temple Garden", "Watery Grave",

    # Fetch lands
    "Arid Mesa", "Bloodstained Mire", "Flooded Strand", "Marsh Flats", "Misty Rainforest",
    "Polluted Delta", "Scalding Tarn", "Verdant Catacombs", "Windswept Heath", "Wooded Foothills",

    # Check lands
    "Clifftop Retreat", "Dragonskull Summit", "Drowned Catacomb", "Glacial Fortress", "Hinterland Harbor",
    "Isolated Chapel", "Rootbound Crag", "Sulfur Falls", "Sunpetal Grove", "Woodland Cemetery",

    # Fast lands
    "Blackcleave Cliffs", "Botanical Sanctum", "Concealed Courtyard", "Copperline Gorge", "Darkslick Shores",
    "Inspiring Vantage", "Razorverge Thicket", "Seachrome Coast", "Spirebluff Canal", "Blooming Marsh",

    # Filter lands
    "Cascade Bluffs", "Fetid Heath", "Fire-Lit Thicket", "Flooded Grove", "Graven Cairns",
    "Mystic Gate", "Rugged Prairie", "Sunken Ruins", "Twilight Mire", "Wooded Bastion",

    # Pain lands
    "Adarkar Wastes", "Battlefield Forge", "Brushland", "Caves of Koilos", "Karplusan Forest",
    "Llanowar Wastes", "Shivan Reef", "Sulfurous Springs", "Underground River", "Yavimaya Coast",

    # Original dual lands
    "Badlands", "Bayou", "Plateau", "Savannah", "Scrubland",
    "Taiga", "Tropical Island", "Tundra", "Underground Sea", "Volcanic Island",

    # Other premium lands
    "City of Brass", "Mana Confluence", "Reflecting Pool", "Cavern of Souls", "Ancient Tomb"
]

def load_price_data():
    """
    Load price data from MTGCardEmbedder system

    Returns:
        dict: A dictionary mapping card names to price information
    """
    try:
        with open('/content/drive/MyDrive/MTGData/price_data.json', 'r') as f:
            price_data = json.load(f)
            logging.info(f"Loaded price data for {len(price_data)} cards")
            return price_data
    except FileNotFoundError:
        logging.warning("Price data not found. Cards will not have price information.")
        return {}

def analyze_commander_for_strategies(commander_name, card_dict, price_data=None, max_price=None):
    """
    Generate custom win conditions based on commander attributes with price filtering

    Args:
        commander_name (str): Name of the commander
        card_dict (dict): Dictionary of card information
        price_data (dict, optional): Dictionary of card pricing data
        max_price (float, optional): Maximum price for cards

    Returns:
        list: List of dictionaries containing strategy information
    """
    commander_info = card_dict.get(commander_name, {})
    oracle_text = commander_info.get('oracle_text', '').lower()
    color_identity = commander_info.get('color_identity', [])

    # Use RAG to find cards that synergize with this commander
    synergistic_cards = search_cards_by_criteria(commander_name, color_identity, n=50, price_data=price_data, max_price=max_price)

    # Analyze synergistic cards to identify themes
    themes = {}
    for card in synergistic_cards:
        if card not in card_dict:
            continue

        card_text = card_dict[card].get('oracle_text', '').lower()
        card_types = card_dict[card].get('types', [])

        # Extract potential themes from card properties
        for keyword in ['token', 'counter', 'sacrifice', 'graveyard', 'draw', 'damage', 'mana']:
            if keyword in card_text:
                if keyword not in themes:
                    themes[keyword] = {'cards': [], 'count': 0}
                themes[keyword]['cards'].append(card)
                themes[keyword]['count'] += 1

        # Add type-based themes
        for type_name in card_types:
            type_key = type_name.lower()
            if type_key not in themes:
                themes[type_key] = {'cards': [], 'count': 0}
            themes[type_key]['cards'].append(card)
            themes[type_key]['count'] += 1

    # Convert themes to strategies, keeping only the most relevant ones
    strategies = []
    theme_names = {
        'token': 'Token Swarm',
        'counter': 'Counter Synergy',
        'sacrifice': 'Aristocrats',
        'graveyard': 'Graveyard Value',
        'draw': 'Card Advantage',
        'damage': 'Direct Damage',
        'mana': 'Ramp Strategy',
        'creature': 'Creature Focus',
        'artifact': 'Artifact Synergy',
        'enchantment': 'Enchantress',
        'instant': 'Spellslinger',
        'sorcery': 'Spellslinger'
    }

    # Sort themes by card count and take top 3
    sorted_themes = sorted(themes.items(), key=lambda x: x[1]['count'], reverse=True)
    for key, data in sorted_themes[:3]:
        if data['count'] >= 5:  # Only consider themes with at least 5 cards
            name = theme_names.get(key, f"{key.capitalize()} Strategy")
            description = f"Build around {commander_name}'s abilities with a focus on {key}-based effects."
            strategies.append({
                'name': name,
                'description': description,
                'key_cards': data['cards'][:5],  # Top 5 cards for this theme
                'power_level': 6 + min(data['count'] // 5, 3)  # Scale power level with card count
            })

    # If no strategies, add a default one
    if not strategies:
        strategies = [{
            'name': 'Goodstuff',
            'description': f'A balanced approach using the best cards in {commander_name}\'s color identity.',
            'key_cards': synergistic_cards[:5],
            'power_level': 6
        }]

    # Add price estimation to strategies
    if price_data:
        for strategy in strategies:
            # Calculate deck price
            key_card_prices = []
            for card in strategy.get('key_cards', []):
                if card in price_data and price_data[card].get('usd'):
                    try:
                        key_card_prices.append(float(price_data[card]['usd']))
                    except (ValueError, TypeError):
                        pass

            if key_card_prices:
                avg_price = sum(key_card_prices) / len(key_card_prices)
                total_price = avg_price * 65 + 0.5 * 35  # Estimate for a full deck

                strategy['price_estimate'] = total_price

                # Categorize by price
                if total_price < 50:
                    strategy['price_category'] = "Budget"
                elif total_price < 200:
                    strategy['price_category'] = "Affordable"
                elif total_price < 500:
                    strategy['price_category'] = "Moderate"
                else:
                    strategy['price_category'] = "Premium"

    return strategies

def find_synergy_cards(commander_name, themes, color_identity, n=8, price_data=None, max_price=None):
    """
    Find cards that synergize with the given themes in the commander's color identity

    Args:
        commander_name (str): Name of the commander
        themes (list): List of themes to find synergy with
        color_identity (list): Color identity of the commander
        n (int, optional): Number of cards to return
        price_data (dict, optional): Dictionary of card pricing data
        max_price (float, optional): Maximum price for cards

    Returns:
        list: List of synergistic cards
    """
    results = []

    # Combine all search results from different themes
    for theme in themes:
        cards = search_cards_by_criteria(theme, color_identity, n=n, price_data=price_data, max_price=max_price)
        results.extend(cards)

    # Deduplicate results and limit to n cards
    unique_results = list(dict.fromkeys(results))
    return unique_results[:n]

def discover_synergies(commander_name, commander_identity, embed_model):
    """Discovers synergies organically rather than using predefined lists"""
    # Get commander text and extract key terms
    commander_info = card_dict.get(commander_name, {})
    oracle_text = commander_info.get('oracle_text', '')

    # Extract meaningful terms using NLP techniques
    key_terms = extract_key_terms(oracle_text)

    # Search for cards that synergize with these key terms
    synergy_cards = {}
    for term in key_terms:
        results = search_cards_by_criteria(
            f"synergize with {term}",
            commander_identity,
            n=30
        )

        for card in results:
            if card not in synergy_cards:
                synergy_cards[card] = 0
            synergy_cards[card] += 1

    # Return cards sorted by synergy score
    return sorted(synergy_cards.items(), key=lambda x: x[1], reverse=True)

def evaluate_card_power(card_name, card_dict, faiss_index, embed_model):
    """Evaluates card power level organically"""
    card_info = card_dict.get(card_name, {})
    oracle_text = card_info.get('oracle_text', '')
    mana_value = card_info.get('mana_value', 0)

    # Create embeddings for evaluation queries
    power_queries = [
        "powerful card in commander",
        "game-winning card",
        "unfair advantage card",
        "competitive commander staple"
    ]

    # Get card embedding
    card_embedding = embed_model.encode([oracle_text])

    # Compare to power query embeddings
    power_embeddings = embed_model.encode(power_queries)
    similarities = cosine_similarities(card_embedding, power_embeddings)

    # Calculate power score based on similarities and mana efficiency
    power_score = (sum(similarities) / len(similarities)) * (7 / (mana_value + 1))

    return power_score
def analyze_deck_with_rag(deck, commander_identity):
    """
    Use RAG to analyze deck for power level and categorization

    Args:
        deck (dict): Dictionary of cards and quantities
        commander_identity (list): Color identity of the commander

    Returns:
        dict: Analysis results including power cards and strategies
    """
    analysis = {
        'game_changers': [],
        'mass_land_denial': [],
        'extra_turns': [],
        'tutors': [],
        'combo_pieces': [],
        'combo_pairs': []
    }

    # Check for game changers
    for card in deck:
        if card in GAME_CHANGER_CARDS:
            analysis['game_changers'].append(card)

    # Check for mass land denial
    for card in deck:
        if card in MASS_LAND_DENIAL:
            analysis['mass_land_denial'].append(card)

    # Check for extra turn cards
    for card in deck:
        if card in EXTRA_TURN_SPELLS:
            analysis['extra_turns'].append(card)

    # Check for tutors
    for card in deck:
        if card in TUTOR_CARDS:
            analysis['tutors'].append(card)

    # Check for combo pieces
    for card in deck:
        if card in COMBO_CARDS:
            analysis['combo_pieces'].append(card)

            # Look for combo partners
            for partner in COMBO_CARDS[card]:
                if partner in deck:
                    combo = tuple(sorted([card, partner]))
                    if combo not in analysis['combo_pairs']:
                        analysis['combo_pairs'].append(combo)

    return analysis

def analyze_deck_for_bracket(deck, commander=None):
    """
    Analyze a deck using RAG to recommend appropriate bracket

    Args:
        deck (dict): Dictionary of cards and quantities
        commander (str, optional): Name of the commander

    Returns:
        tuple: (recommended_bracket, reasons)
    """
    if commander:
        commander_identity = get_commander_identity(commander)
    else:
        # Try to find commander in the deck
        commander_identity = []
        for card in deck:
            card_info = card_dict.get(card, {})
            if card_info and 'Commander' in card_info.get('type_line', ''):
                commander_identity = card_info.get('color_identity', [])
                break

        # Default to all colors if no commander found
        if not commander_identity:
            commander_identity = ['W', 'U', 'B', 'R', 'G']

    # Run RAG analysis
    analysis = analyze_deck_with_rag(deck, commander_identity)

    # Determine bracket
    bracket = determine_bracket_from_analysis(analysis)

    # Build reasons list
    reasons = []

    # Game changers
    if not analysis['game_changers']:
        reasons.append("You have no game changer cards")
    elif len(analysis['game_changers']) <= 3:
        reasons.append(f"You have {len(analysis['game_changers'])} game changer card{'s' if len(analysis['game_changers']) > 1 else ''}: {', '.join(analysis['game_changers'])}")
    else:
        reasons.append(f"You have more than 3 game changer cards: {', '.join(analysis['game_changers'])}")

    # Mass land denial
    if analysis['mass_land_denial']:
        reasons.append(f"You have mass land denial cards: {', '.join(analysis['mass_land_denial'])}")
    else:
        reasons.append("You have no mass land denial cards")

    # Extra turns
    if analysis['extra_turns']:
        reasons.append(f"You have extra turn cards: {', '.join(analysis['extra_turns'])}")
    else:
        reasons.append("You have no extra turn cards")

    # Tutors
    if analysis['tutors']:
        reasons.append(f"You have {len(analysis['tutors'])} tutor card{'s' if len(analysis['tutors']) > 1 else ''}: {', '.join(analysis['tutors'])}")
    else:
        reasons.append("You have no tutor cards")

    # Combos
    if analysis['combo_pairs']:
        reasons.append(f"You have {len(analysis['combo_pairs'])} potential combo{'s' if len(analysis['combo_pairs']) > 1 else ''}:")
        for combo in analysis['combo_pairs']:
            reasons.append(f"  - {' + '.join(combo)}")
    else:
        reasons.append("You have no two-card combos")

    return bracket, reasons

def determine_bracket_from_analysis(analysis):
    """
    Determine appropriate bracket based on deck analysis

    Args:
        analysis (dict): Results of deck analysis

    Returns:
        int: Recommended bracket (1-5)
    """
    # Extract analyzed categories
    game_changers = analysis['game_changers']
    mass_land_denial = analysis['mass_land_denial']
    extra_turns = analysis['extra_turns']
    tutors = analysis['tutors']
    combo_pairs = analysis['combo_pairs']

    # cEDH (5)
    if (len(game_changers) >= 5 or
        len(mass_land_denial) >= 1 or
        len(combo_pairs) >= 2 or
        (len(game_changers) >= 2 and len(tutors) >= 3)):
        return 5

    # Optimized (4)
    if (len(game_changers) >= 2 or
        len(combo_pairs) == 1 or
        len(extra_turns) >= 2 or
        (len(tutors) >= 3 and len(game_changers) >= 1)):
        return 4

    # Upgraded (3)
    if (len(game_changers) > 0 or
        len(tutors) >= 2 or
        len(extra_turns) == 1):
        return 3

    # Core (2)
    if len(tutors) > 0:
        return 2

    # Exhibition (1)
    return 1

def check_card_bracket_compatibility(card_name, bracket):
    """
    Check if a card is compatible with the bracket's restrictions

    Args:
        card_name (str): Name of the card
        bracket (int): Bracket to check against (1-5)

    Returns:
        bool: True if compatible, False otherwise
    """
    if card_name not in card_dict:
        return True  # If we don't have card info, default to allowing it

    # Always allow the card in brackets 4 and 5 (no restrictions)
    if bracket >= 4:
        return True

    # Check for mass land denial
    if bracket <= 3 and card_name in MASS_LAND_DENIAL:
        return False

    # Check for game changers in lower brackets
    if bracket <= 2 and card_name in GAME_CHANGER_CARDS:
        return False

    # For bracket 3, limit to 3 game changers (handled elsewhere)

    # Extra turns limitations
    if bracket <= 2 and card_name in EXTRA_TURN_SPELLS:
        return False

    # Tutor limitations (stricter for lower brackets)
    if bracket <= 2 and card_name in TUTOR_CARDS:
        # For bracket 2, allow some specific tutors but limit quantity
        return False

    # Check for combo pieces
    if bracket <= 3 and card_name in COMBO_CARDS:
        return False

    return True

def filter_deck_for_bracket(deck, bracket):
    """
    Filter a deck to comply with the chosen bracket's restrictions

    Args:
        deck (dict): Dictionary of cards and quantities
        bracket (int): Bracket to filter for (1-5)

    Returns:
        tuple: (filtered_deck, removed_cards)
    """
    filtered_deck = {}
    removed_cards = {}

    # If bracket is 4 or 5, no restrictions apply
    if bracket >= 4:
        return deck, {}

    # Count restricted card types
    game_changers_count = 0
    tutor_count = 0
    extra_turns_count = 0
    combo_pieces = {}

    # First pass: identify all restricted cards
    for card, qty in deck.items():
        # Check for game changers
        if card in GAME_CHANGER_CARDS:
            game_changers_count += qty

        # Check for tutors
        if card in TUTOR_CARDS:
            tutor_count += qty

        # Check for extra turns
        if card in EXTRA_TURN_SPELLS:
            extra_turns_count += qty

        # Check for combo pieces
        if card in COMBO_CARDS:
            combo_partners = COMBO_CARDS[card]
            for partner in combo_partners:
                if partner in deck:
                    if card not in combo_pieces:
                        combo_pieces[card] = []
                    combo_pieces[card].append(partner)

    # Second pass: apply bracket-specific filters
    for card, qty in deck.items():
        # Exhibition and Core brackets (1 & 2)
        if bracket <= 2:
            # No game changers
            if card in GAME_CHANGER_CARDS:
                removed_cards[card] = qty
                continue

            # No mass land denial
            if card in MASS_LAND_DENIAL:
                removed_cards[card] = qty
                continue

            # No extra turns
            if card in EXTRA_TURN_SPELLS:
                removed_cards[card] = qty
                continue

            # Limited tutors (allow just 1-2 specific ones for bracket 2)
            if card in TUTOR_CARDS:
                if bracket == 1 or (bracket == 2 and tutor_count > 2):
                    removed_cards[card] = qty
                    continue

            # No 2-card combos
            if card in combo_pieces:
                removed_cards[card] = qty
                continue

        # Upgraded bracket (3)
        elif bracket == 3:
            # No mass land denial
            if card in MASS_LAND_DENIAL:
                removed_cards[card] = qty
                continue

            # Limit game changers to 3
            if card in GAME_CHANGER_CARDS and game_changers_count > 3:
                removed_cards[card] = qty
                game_changers_count -= qty
                continue

        # If we made it here, the card is allowed
        filtered_deck[card] = qty

    return filtered_deck, removed_cards

def replace_removed_cards(deck, removed_cards, commander_identity, bracket, price_data=None, max_price=None):
    """
    Replace cards that were removed due to bracket restrictions

    Args:
        deck (dict): Current deck
        removed_cards (dict): Cards that were removed
        commander_identity (list): Color identity of the commander
        bracket (int): Bracket to build for (1-5)
        price_data (dict, optional): Dictionary of card pricing data
        max_price (float, optional): Maximum price for cards

    Returns:
        dict: Updated deck with replacement cards
    """
    if not removed_cards:
        return deck

    # Calculate how many cards we need to replace
    cards_to_add = sum(removed_cards.values())
    logging.info(f"Replacing {cards_to_add} cards removed due to bracket {bracket} restrictions")

    # Get list of cards already in the deck
    exclude_list = list(deck.keys()) + list(removed_cards.keys())

    # Generate replacement queries based on removed card types
    replacement_queries = []

    # Check what types of cards were removed and generate appropriate replacement queries
    has_removed_tutors = any(card in TUTOR_CARDS for card in removed_cards)
    has_removed_extra_turns = any(card in EXTRA_TURN_SPELLS for card in removed_cards)
    has_removed_land_denial = any(card in MASS_LAND_DENIAL for card in removed_cards)
    has_removed_game_changers = any(card in GAME_CHANGER_CARDS for card in removed_cards)
    has_removed_combos = any(card in COMBO_CARDS for card in removed_cards)

    # Add replacement queries based on what was removed
    if has_removed_tutors:
        replacement_queries.extend(["card draw", "card advantage"])

    if has_removed_extra_turns:
        replacement_queries.extend(["additional combat", "untap effects"])

    if has_removed_land_denial:
        replacement_queries.extend(["targeted removal", "single target land destruction"])

    if has_removed_game_changers:
        replacement_queries.extend(["value engine", "gradual advantage"])

    if has_removed_combos:
        replacement_queries.extend(["synergy pieces", "value creatures"])

    # If we don't have specific replacements, use general good stuff
    if not replacement_queries:
        replacement_queries = ["card draw", "removal", "ramp", "board wipe", "value creature"]

    # Add cards by cycling through queries
    added_count = 0
    query_index = 0

    while added_count < cards_to_add:
        query = replacement_queries[query_index % len(replacement_queries)]
        query_index += 1

        # Search for cards matching the query
        results = search_cards_by_criteria(query, commander_identity, exclude_cards=exclude_list, n=20, price_data=price_data, max_price=max_price)

        # Filter results for bracket compatibility
        bracket_compatible = [card for card in results if check_card_bracket_compatibility(card, bracket)]

        if bracket_compatible:
            # Pick one randomly
            card_to_add = random.choice(bracket_compatible)

            # Add to deck
            if card_to_add in deck:
                deck[card_to_add] += 1
            else:
                deck[card_to_add] = 1

            exclude_list.append(card_to_add)
            added_count += 1

            # If we're struggling to find cards, relax the search a bit
            if query_index > 10 * len(replacement_queries):
                # Try to add basic lands as a last resort
                for color in commander_identity:
                    if color in BASIC_LANDS:
                        land = BASIC_LANDS[color]
                        if land in deck:
                            deck[land] += 1
                        else:
                            deck[land] = 1
                        added_count += 1
                        if added_count >= cards_to_add:
                            break
                break

    return deck

def get_appropriate_bracket_for_win_condition(win_condition):
    """
    Determine the appropriate bracket for a win condition based on its power level

    Args:
        win_condition (dict): Win condition information

    Returns:
        int: Recommended bracket (1-5)
    """
    power_level = win_condition.get('power_level', 5)  # Default to mid-power if not specified

    if power_level <= 3:
        return 1  # Exhibition
    elif power_level <= 5:
        return 2  # Core
    elif power_level <= 7:
        return 3  # Upgraded
    elif power_level <= 9:
        return 4  # Optimized
    else:
        return 5  # cEDH

def filter_lands_by_quality(lands, land_quality):
    """Enhanced filter with COLORLESS LAND SUPPORT"""
    setting = LAND_QUALITY_SETTINGS.get(land_quality, LAND_QUALITY_SETTINGS['balanced'])

    filtered_lands = []
    for land in lands:
        # Always include basic lands (including Wastes for colorless)
        if any(land == basic for basic in BASIC_LANDS.values()):
            filtered_lands.append(land)
            continue

        # Define utility lands that get an exception to the enters-tapped rule
        utility_exceptions = ["Command Tower", "Path of Ancestry"]

        # For competitive, strictly enforce the no-tapped-lands rule
        if land_quality == 'competitive':
            # Add more exceptions for competitive
            utility_exceptions.extend(["Reflecting Pool", "Exotic Orchard",
                                      "City of Brass", "Mana Confluence"])

            card_info = card_dict.get(land, {})
            oracle_text = card_info.get('oracle_text', '').lower()

            # Skip land if it explicitly mentions entering tapped AND it's not in our exceptions
            if ('enters the battlefield tapped' in oracle_text or land in ENTERS_TAPPED_LANDS) and land not in utility_exceptions:
                continue

            # Cycling lands, tap lands, and bounce lands are all excluded for competitive
            if any(tap_land_type in land.lower() for tap_land_type in ['refuge', 'guildgate', 'karoo', 'vivid']):
                continue

        # For balanced, allow some tapped lands
        elif not setting['allowed_tapped'] and land in ENTERS_TAPPED_LANDS and land not in utility_exceptions:
            continue

        # For budget setting, exclude expensive premium lands
        if setting['budget'] == 'low' and land in PREMIUM_UNTAPPED_LANDS:
            continue

        # Land passed filters
        filtered_lands.append(land)

    return filtered_lands

def search_lands_by_quality(commander_identity, land_quality, exclude_cards=None, count=10, price_data=None, max_price=None):
    """
    Enhanced search for lands that strictly enforces the land quality setting

    Args:
        commander_identity (list): Color identity of the commander
        land_quality (str): Quality setting ('competitive', 'balanced', or 'budget')
        exclude_cards (list, optional): Cards to exclude from search
        count (int, optional): Number of lands to return
        price_data (dict, optional): Dictionary of card pricing data
        max_price (float, optional): Maximum price for cards

    Returns:
        list: List of lands matching the criteria
    """
    setting = LAND_QUALITY_SETTINGS.get(land_quality, LAND_QUALITY_SETTINGS['balanced'])
    exclude_cards = exclude_cards or []

    # For competitive lands, completely exclude tapped lands from consideration
    if land_quality == 'competitive':
        # Start with premium untapped lands within budget
        premium_lands = [land for land in PREMIUM_UNTAPPED_LANDS
                        if land in card_dict and
                        all(c in commander_identity for c in card_dict[land].get('color_identity', [])) and
                        (max_price is None or price_data is None or land not in price_data or
                         not price_data[land].get('usd') or float(price_data[land].get('usd', 0) or 0) <= max_price)]

        # Filter premium lands by strictly confirming they're not in ENTERS_TAPPED_LANDS
        premium_lands = [land for land in premium_lands if land not in ENTERS_TAPPED_LANDS]

        land_results = premium_lands.copy()

        # If we need more lands, search for additional untapped options using strict query
        if len(land_results) < count:
            # Use more specific query language and exclude all known tapped lands
            untapped_query = "type:land enters untapped"
            additional_lands = search_cards_by_criteria(
                untapped_query,
                commander_identity,
                exclude_cards=exclude_cards + land_results + ENTERS_TAPPED_LANDS,
                n=count-len(land_results),
                price_data=price_data,
                max_price=max_price
            )

            # Double-check these lands aren't in our tapped lands list
            additional_lands = [land for land in additional_lands if land not in ENTERS_TAPPED_LANDS]
            land_results.extend(additional_lands)

        # Basic lands are always OK
        basic_lands = [BASIC_LANDS[color] for color in commander_identity if color in BASIC_LANDS]
        for basic in basic_lands:
            if basic not in land_results and len(land_results) < count:
                land_results.append(basic)

    elif setting['budget'] == 'low':
        # Start with some tap lands for budget setting
        tap_lands = [land for land in ENTERS_TAPPED_LANDS
                    if land in card_dict and
                    all(c in commander_identity for c in card_dict[land].get('color_identity', [])) and
                    (max_price is None or price_data is None or land not in price_data or
                     not price_data[land].get('usd') or float(price_data[land].get('usd', 0) or 0) <= max_price)]

        # Get a subset of tap lands
        land_results = tap_lands[:min(count // 2, len(tap_lands))]

        # Add budget untapped lands if needed
        if len(land_results) < count:
            additional_lands = search_cards_by_criteria(
                "type:land",
                commander_identity,
                exclude_cards=exclude_cards + PREMIUM_UNTAPPED_LANDS + land_results,
                n=count-len(land_results),
                price_data=price_data,
                max_price=max_price
            )
            land_results.extend(additional_lands)

    else:  # balanced approach
        # Mix of tapped and untapped lands
        tap_lands = [land for land in ENTERS_TAPPED_LANDS
                    if land in card_dict and
                    all(c in commander_identity for c in card_dict[land].get('color_identity', [])) and
                    (max_price is None or price_data is None or land not in price_data or
                     not price_data[land].get('usd') or float(price_data[land].get('usd', 0) or 0) <= max_price)]

        # Take some tap lands
        land_results = tap_lands[:min(count // 3, len(tap_lands))]

        # Add some premium lands if within budget
        premium_lands = [land for land in PREMIUM_UNTAPPED_LANDS
                        if land in card_dict and
                        all(c in commander_identity for c in card_dict[land].get('color_identity', [])) and
                        (max_price is None or price_data is None or land not in price_data or
                         not price_data[land].get('usd') or float(price_data[land].get('usd', 0) or 0) <= max_price)]

        land_results.extend(premium_lands[:min(count // 3, len(premium_lands))])

        # Fill remaining slots with other lands
        if len(land_results) < count:
            additional_lands = search_cards_by_criteria(
                "type:land",
                commander_identity,
                exclude_cards=exclude_cards + land_results,
                n=count-len(land_results),
                price_data=price_data,
                max_price=max_price
            )
            land_results.extend(additional_lands)

    # Apply a final strict filter for competitive lands
    if land_quality == 'competitive':
        land_results = filter_lands_by_quality(land_results, land_quality)

    # Filter results to ensure they match commander identity
    filtered_results = [land for land in land_results
                       if land in card_dict and
                       all(c in commander_identity for c in card_dict[land].get('color_identity', []))]

    return filtered_results[:count]

def load_card_data():
    """
    Load card data from Scryfall API

    Returns:
        list: List of legal Commander cards
    """
    global card_dict, card_objects, flip_card_mapping
    bulk_data = requests.get("https://api.scryfall.com/bulk-data").json()
    oracle_url = next(item for item in bulk_data["data"] if item["name"] == "Oracle Cards")["download_uri"]
    cards = requests.get(oracle_url).json()
    card_objects = []

    for card in cards:
        if 'name' not in card or 'legalities' not in card:
            continue
        if card.get('legalities', {}).get('commander') in ['legal', 'restricted']:
            card_objects.append(card)
            card_dict[card['name']] = {
                'color_identity': card.get('color_identity', []),
                'type_line': card.get('type_line', ''),
                'oracle_text': card.get('oracle_text', ''),
                'mana_value': card.get('cmc', 0),
                'types': extract_types(card.get('type_line', ''))
            }
            if 'card_faces' in card:
              for face in card['card_faces']:
                  if 'name' in face and face['name'] != card['name']:
                      # Add to flip_card_mapping to track face->main card relationships
                      flip_card_mapping[face['name']] = card['name']
                      # Still add face info to card_dict for search purposes
                      card_dict[face['name']] = {
                          'color_identity': card.get('color_identity', []),
                          'type_line': face.get('type_line', ''),
                          'oracle_text': face.get('oracle_text', ''),
                          'mana_value': card.get('cmc', 0),
                          'types': extract_types(face.get('type_line', '')),
                          'is_face': True,
                          'main_card': card['name']
                      }
    logging.info(f"Loaded {len(card_objects)} legal Commander cards")
    return card_objects

def extract_types(type_line):
    """
    Extract card types from type line

    Args:
        type_line (str): Card type line

    Returns:
        list: List of card types
    """
    types = []
    if "Land" in type_line:
        types.append("Land")
    if "Creature" in type_line:
        types.append("Creature")
    if "Artifact" in type_line:
        types.append("Artifact")
    if "Enchantment" in type_line:
        types.append("Enchantment")
    if "Planeswalker" in type_line:
        types.append("Planeswalker")
    if "Instant" in type_line:
        types.append("Instant")
    if "Sorcery" in type_line:
        types.append("Sorcery")
    return types

def setup_retriever(cards):
    """
    Set up semantic search for cards

    Args:
        cards (list): List of card objects
    """
    global faiss_index, embed_model, card_texts
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create more detailed card texts for better semantic search
    card_texts = []
    for c in cards:
        if 'name' not in c or 'type_line' not in c:
            continue

        # Extract key card properties for embedding context
        card_name = c['name']
        type_line = c['type_line']
        oracle_text = c.get('oracle_text', '')
        keywords = c.get('keywords', [])
        mana_cost = c.get('mana_cost', '')

        # Create a rich text representation
        text = f"{card_name}. Cost: {mana_cost}. Types: {type_line}. "

        if keywords:
            text += f"Keywords: {', '.join(keywords)}. "

        if oracle_text:
            text += f"Text: {oracle_text}"

        # Add card faces for dual-faced cards
        if 'card_faces' in c:
            face_texts = []
            for face in c['card_faces']:
                if 'name' in face and 'type_line' in face:
                    face_name = face['name']
                    face_type = face['type_line']
                    face_text = face.get('oracle_text', '')
                    face_cost = face.get('mana_cost', '')

                    face_desc = f"{face_name}. Cost: {face_cost}. Types: {face_type}. "
                    if face_text:
                        face_desc += f"Text: {face_text}"
                    face_texts.append(face_desc)

            if face_texts:
                text += f" Faces: {' | '.join(face_texts)}"

        card_texts.append(text)

    logging.info(f"Generating embeddings for {len(card_texts)} cards...")

    # Create embeddings with batched processing for memory efficiency
    batch_size = 256
    all_embeddings = []

    for i in range(0, len(card_texts), batch_size):
        batch = card_texts[i:i + batch_size]
        batch_embeddings = embed_model.encode(batch, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings).astype('float32')

    # Create and populate the FAISS index
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    logging.info("Embeddings completed and indexed")

def load_language_model(model_dir):
    """
    Load language model optimized for GPU

    Args:
        model_dir (str): Directory containing the model

    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Force GPU usage
    device = torch.device("cuda")

    # Load model with GPU optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16  # Use half precision for GPU
    )

    # Explicitly move model to GPU
    model = model.to(device)

    logging.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    return tokenizer, model

def get_commander_identity(commander_name):
    """
    Get the color identity of a commander

    Args:
        commander_name (str): Name of the commander

    Returns:
        list: Color identity of the commander
    """
    commander_info = card_dict.get(commander_name)
    if not commander_info:
        logging.warning(f"Commander '{commander_name}' not found in database")
        # Try to find similar commander names
        similar_names = search_similar_cards(commander_name, n=5)
        logging.info(f"Did you mean one of these? {', '.join(similar_names)}")
        return []
    return commander_info.get('color_identity', [])

def validate_deck_identity(deck, commander_identity):
    """
    Validate that all cards in the deck match the commander's color identity

    Args:
        deck (dict): Dictionary of cards and quantities
        commander_identity (list): Color identity of the commander

    Returns:
        tuple: (valid_deck, invalid_cards)
    """
    valid_deck, invalid_cards = {}, []
    for card, qty in deck.items():
        # Skip validation for basic lands that match commander color identity
        if card in BASIC_LANDS.values():
            if not commander_identity or any(color in commander_identity for color, land in BASIC_LANDS.items() if land == card):
                valid_deck[card] = qty
                continue

        card_info = card_dict.get(card)
        if not card_info:
            invalid_cards.append(f"{card} (not found)")
            continue

        card_identity = card_info.get('color_identity', [])

        # Check if card's color identity is a subset of commander's identity
        if all(c in commander_identity for c in card_identity) and card not in COMMANDER_BANNED_CARDS:
            valid_deck[card] = qty
        else:
            reason = "banned" if card in COMMANDER_BANNED_CARDS else "color identity mismatch"
            invalid_cards.append(f"{card} ({reason})")

    logging.info(f"Removed {len(invalid_cards)} invalid cards: {', '.join(invalid_cards)}")
    return valid_deck, invalid_cards

def search_similar_cards(query, n=10):
    """
    Search for cards similar to the query

    Args:
        query (str): Search query
        n (int, optional): Number of results to return

    Returns:
        list: List of similar card names
    """
    if not faiss_index or not card_objects:
        return []

    query_embedding = embed_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, n)

    results = []
    for i in indices[0]:
        if i < len(card_objects):
            card_name = card_objects[i]['name']
            # Check if this is a flip card face
            if card_name in flip_card_mapping:
                card_name = flip_card_mapping[card_name]
            if card_name not in results:
                results.append(card_name)

    return results[:n]

def search_cards_by_criteria(query, commander_identity, exclude_cards=None, n=30, price_data=None, max_price=None):
    """Enhanced search with COLORLESS CARD SUPPORT"""
    if not exclude_cards:
        exclude_cards = set()
    else:
        exclude_cards = set(exclude_cards)

    # Create a more specific query to get better results
    enhanced_query = f"best {query} for commander deck building"

    query_embedding = embed_model.encode([enhanced_query])
    distances, indices = faiss_index.search(query_embedding, n*5)  # Get even more results to filter

    results = []
    for i in indices[0]:
        if i < len(card_objects):
            card = card_objects[i]
            if 'name' not in card:
                continue

            card_name = card['name']

            # If this is a flip card face, get the main card name
            if card_name in flip_card_mapping:
                card_name = flip_card_mapping[card_name]

            # Skip if already in exclude list or results
            if card_name in exclude_cards or card_name in results:
                continue

            # FIXED COLOR IDENTITY CHECK FOR COLORLESS CARDS
            card_identity = card.get('color_identity', [])
            # Colorless cards (empty identity) can be played in any deck
            if card_identity:  # Only check if card has colors
                if not all(c in commander_identity for c in card_identity):
                    continue

            # Check legality
            if card.get('legalities', {}).get('commander') not in ['legal', 'restricted'] or card_name in COMMANDER_BANNED_CARDS:
                continue

            # Price check (unchanged)
            if max_price is not None and price_data:
                if card_name not in price_data:
                    continue
                card_price = price_data[card_name].get('usd')
                if not card_price or card_price == 'null':
                    continue
                try:
                    price_float = float(card_price)
                    if price_float > max_price:
                        continue
                except (ValueError, TypeError):
                    continue

            results.append(card_name)
            if len(results) >= n:
                break

    return results

def get_commander_themes(commander_name):
    """
    Identify potential themes for the commander

    Args:
        commander_name (str): Name of the commander

    Returns:
        list: List of themes
    """
    if not commander_name in card_dict:
        return []

    commander_info = card_dict[commander_name]
    oracle_text = commander_info.get('oracle_text', '')

    themes = []
    if 'draw' in oracle_text.lower() or 'card' in oracle_text.lower():
        themes.append("card draw")
    if 'damage' in oracle_text.lower():
        themes.append("damage")
    if 'counter' in oracle_text.lower():
        themes.append("counters")
    if 'token' in oracle_text.lower():
        themes.append("tokens")
    if 'graveyard' in oracle_text.lower() or 'cemetery' in oracle_text.lower():
        themes.append("graveyard")
    if 'sacrifice' in oracle_text.lower():
        themes.append("sacrifice")
    if 'discard' in oracle_text.lower():
        themes.append("discard")

    return themes

def generate_synergy_queries(commander_name):
    """
    Generate search queries based on commander themes

    Args:
        commander_name (str): Name of the commander

    Returns:
        list: List of search queries
    """
    themes = get_commander_themes(commander_name)
    synergy_queries = []

    # Add commander name for direct synergy
    synergy_queries.append(f"synergy with {commander_name}")

    # Add theme-based queries
    for theme in themes:
        synergy_queries.append(f"cards that work with {theme}")

    # Add general synergy queries based on commander text
    if commander_name in card_dict:
        commander_text = card_dict[commander_name].get('oracle_text', '')
        key_terms = re.findall(r'\b\w+\b', commander_text.lower())

        # Filter out common words
        stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'of', 'with', 'by'}
        key_terms = [term for term in key_terms if term not in stop_words and len(term) > 3]

        # Add key terms as synergy queries
        for term in key_terms[:3]:  # Use top 3 terms
            synergy_queries.append(f"cards with {term}")

    return synergy_queries

def generate_deck(commander_name, tokenizer, model):
    """
    Generate initial deck using language model

    Args:
        commander_name (str): Name of the commander
        tokenizer: Tokenizer for the language model
        model: Language model

    Returns:
        str: Generated deck text
    """
    prompt = f"Generate a synergistic Commander decklist for {commander_name}. Include card names and quantities, with 99 cards plus the commander."
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        **inputs,
        max_length=2000,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_deck(deck_text):
    """
    Extract card names and quantities from generated text

    Args:
        deck_text (str): Generated deck text

    Returns:
        dict: Dictionary of cards and quantities
    """
    deck = {}

    # Pattern for "NxCardName" format
    pattern1 = r"(\d+)x ([^\n,]+)"
    # Pattern for "N CardName" format
    pattern2 = r"(\d+) ([^\n,]+)"
    # Pattern for lines with card names only
    pattern3 = r"^([^0-9\n][^\n]+)$"

    # Find all matches for each pattern
    matches1 = re.findall(pattern1, deck_text)
    matches2 = re.findall(pattern2, deck_text)

    # Process matches from patterns with quantities
    for qty_str, card in matches1 + matches2:
        try:
            qty = int(qty_str)
            card_name = card.strip()
            if card_name and 0 < qty <= 99:  # Sanity check
                # Check if this is a flip card face
                if card_name in flip_card_mapping:
                    main_card = flip_card_mapping[card_name]
                    if main_card in deck:
                        deck[main_card] += qty
                    else:
                        deck[main_card] = qty
                else:
                    deck[card_name] = qty
        except ValueError:
            continue

    # Only use pattern3 if we didn't get enough cards
    if len(deck) < 20:
        lines = deck_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not any(c.isdigit() for c in line[:2]):  # Avoid lines starting with numbers
                # Check if this is a flip card face
                if line in flip_card_mapping:
                    main_card = flip_card_mapping[line]
                    if main_card in deck:
                        deck[main_card] += 1
                    else:
                        deck[main_card] = 1
                else:
                    deck[line] = 1

    return deck

def count_unique_physical_cards(deck):
    """
    Count total number of physical cards in the deck, accounting for flip cards

    Args:
        deck (dict): Dictionary of cards and quantities

    Returns:
        int: Number of physical cards
    """
    total_count = 0
    processed_cards = set()

    for card, qty in deck.items():
        # If this is a flip card face, get the main card
        if card in flip_card_mapping:
            main_card = flip_card_mapping[card]
            if main_card not in processed_cards:
                processed_cards.add(main_card)
                total_count += qty
        else:
            if card not in processed_cards:
                processed_cards.add(card)
                total_count += qty

    return total_count

def calculate_mana_curve(deck):
    """
    Calculate the mana curve of the deck

    Args:
        deck (dict): Dictionary of cards and quantities

    Returns:
        dict: Mana curve distribution
    """
    curve = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, '7+': 0}

    for card, qty in deck.items():
        card_info = card_dict.get(card)
        if not card_info:
            continue

        # Skip lands
        if 'Land' in card_info.get('types', []):
            continue

        mana_value = card_info.get('mana_value', 0)

        # Categorize by mana value
        if mana_value >= 7:
            curve['7+'] += qty
        else:
            curve[int(mana_value)] += qty

    return curve

def analyze_mana_curve(curve):
    """
    Analyze the mana curve for potential issues

    Args:
        curve (dict): Mana curve distribution

    Returns:
        list: List of issues
    """
    issues = []

    # Calculate total spells
    total_spells = sum(curve.values())

    if total_spells < 10:
        return []  # Not enough spells to analyze

    # Check for imbalances in the curve
    if curve[1] + curve[2] < total_spells * 0.2:
        issues.append("low_early_drops")

    if curve['7+'] > total_spells * 0.15:
        issues.append("top_heavy")

    # Check for gaps in the curve
    for i in range(2, 5):
        if curve[i] == 0:
            issues.append(f"gap_at_{i}")

    return issues

def fix_mana_curve(deck, curve, commander_identity, exclude_list, price_data=None, max_price=None):
    """
    Attempt to fix issues with the mana curve

    Args:
        deck (dict): Dictionary of cards and quantities
        curve (dict): Mana curve distribution
        commander_identity (list): Color identity of the commander
        exclude_list (list): Cards to exclude
        price_data (dict, optional): Dictionary of card pricing data
        max_price (float, optional): Maximum price for cards

    Returns:
        dict: Updated deck
    """
    issues = analyze_mana_curve(curve)

    if not issues:
        return deck

    # Keep track of the deck's physical card count
    physical_card_count = count_unique_physical_cards(deck)

    # Fix common issues
    if "low_early_drops" in issues:
        # Add more low-cost cards
        logging.info("Fixing mana curve: Adding more low-drop cards")

        # Replace some high-cost cards with low-cost ones
        high_cost_cards = []
        for card, qty in deck.items():
            card_info = card_dict.get(card)
            if not card_info:
                continue

            if 'Land' in card_info.get('types', []):
                continue

            mana_value = card_info.get('mana_value', 0)
            if mana_value >= 5:
                high_cost_cards.extend([card] * qty)

        # Shuffle to randomize selections
        random.shuffle(high_cost_cards)

        # Replace up to 5 high-cost cards
        replacements = min(5, len(high_cost_cards))

        for i in range(replacements):
            if i < len(high_cost_cards):
                card_to_replace = high_cost_cards[i]

                # Find low-cost replacements with price filtering
                low_cost_cards = search_cards_by_criteria(
                    "mana value 1 or mana value 2",
                    commander_identity,
                    exclude_cards=exclude_list,
                    price_data=price_data,
                    max_price=max_price
                )

                if low_cost_cards:
                    # Remove high-cost card
                    if deck[card_to_replace] > 1:
                        deck[card_to_replace] -= 1
                    else:
                        del deck[card_to_replace]

                    # Add low-cost card
                    replacement = low_cost_cards[0]

                    # Check if replacement is a flip card face
                    if replacement in flip_card_mapping:
                        replacement = flip_card_mapping[replacement]

                    if replacement in deck:
                        deck[replacement] += 1
                    else:
                        deck[replacement] = 1
                    exclude_list.append(replacement)

    if "top_heavy" in issues:
        # Replace some high-cost cards with mid-range ones
        logging.info("Fixing mana curve: Reducing top-heavy cards")

        high_cost_cards = []
        for card, qty in deck.items():
            card_info = card_dict.get(card)
            if not card_info:
                continue

            if 'Land' in card_info.get('types', []):
                continue

            mana_value = card_info.get('mana_value', 0)
            if mana_value >= 7:
                high_cost_cards.extend([card] * qty)

        # Replace up to half of the high-cost cards
        replacements = min(len(high_cost_cards) // 2 + 1, len(high_cost_cards))

        for i in range(replacements):
            if i < len(high_cost_cards):
                card_to_replace = high_cost_cards[i]

                # Find mid-range replacements with price filtering
                mid_cost_cards = search_cards_by_criteria(
                    "mana value 3 or mana value 4 or mana value 5",
                    commander_identity,
                    exclude_cards=exclude_list,
                    price_data=price_data,
                    max_price=max_price
                )

                if mid_cost_cards:
                    # Remove high-cost card
                    if deck[card_to_replace] > 1:
                        deck[card_to_replace] -= 1
                    else:
                        del deck[card_to_replace]

                    # Add mid-cost card
                    replacement = mid_cost_cards[0]

                    # Check if replacement is a flip card face
                    if replacement in flip_card_mapping:
                        replacement = flip_card_mapping[replacement]

                    if replacement in deck:
                        deck[replacement] += 1
                    else:
                        deck[replacement] = 1
                    exclude_list.append(replacement)

    # Fix gaps in the curve
    for issue in issues:
        if issue.startswith("gap_at_"):
            mana_value = int(issue.split("_")[-1])
            logging.info(f"Fixing mana curve: Adding cards at mana value {mana_value}")

            # Find cards with the missing mana value
            gap_cards = search_cards_by_criteria(
                f"mana value {mana_value}",
                commander_identity,
                exclude_cards=exclude_list,
                price_data=price_data,
                max_price=max_price
            )

            if gap_cards:
                # Find a card to replace
                replaced = False
                for card, qty in list(deck.items()):
                    card_info = card_dict.get(card)
                    if not card_info:
                        continue

                    if 'Land' in card_info.get('types', []):
                        continue

                    existing_mv = card_info.get('mana_value', 0)
                    if (existing_mv >= 6 or existing_mv <= 1) and card not in exclude_list:
                        # Replace this card
                        if qty > 1:
                            deck[card] -= 1
                        else:
                            del deck[card]

                        # Add gap-filling card
                        replacement = gap_cards[0]

                        # Check if replacement is a flip card face
                        if replacement in flip_card_mapping:
                            replacement = flip_card_mapping[replacement]

                        if replacement in deck:
                            deck[replacement] += 1
                        else:
                            deck[replacement] = 1
                        exclude_list.append(replacement)
                        replaced = True
                        break

    return deck

def count_card_types(deck):
    """
    Count cards by type in the deck

    Args:
        deck (dict): Dictionary of cards and quantities

    Returns:
        dict: Count of each card type
    """
    type_counts = {'Land': 0, 'Creature': 0, 'Artifact': 0,
                  'Enchantment': 0, 'Planeswalker': 0,
                  'Instant': 0, 'Sorcery': 0, 'Other': 0}

    # Track cards we've already counted to avoid double-counting flip faces
    counted_cards = set()

    for card, qty in deck.items():
        # Skip if this is a flip card face that we've already counted
        if card in flip_card_mapping:
            main_card = flip_card_mapping[card]
            if main_card in counted_cards:
                continue
            counted_cards.add(main_card)
            # Use the main card for type determination
            card = main_card
        else:
            # Add regular cards to counted set
            counted_cards.add(card)

        if card in BASIC_LANDS.values():
            type_counts['Land'] += qty
            continue

        card_info = card_dict.get(card)
        if not card_info:
            type_counts['Other'] += qty
            continue

        types = card_info.get('types', [])

        # Count by primary type (use first match in hierarchy)
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

def determine_win_condition(commander, tokenizer, model, commander_identity, price_data=None, max_price=None):
    """
    Generate win conditions using GPU acceleration with price filtering

    Args:
        commander (str): Name of the commander
        tokenizer: Tokenizer for the language model
        model: Language model
        commander_identity (list): Color identity of the commander
        price_data (dict, optional): Dictionary of card pricing data
        max_price (float, optional): Maximum price for cards

    Returns:
        tuple: (win_conditions, win_condition_text)
    """
    # First, generate custom strategies based on commander attributes with price filtering
    custom_strategies = analyze_commander_for_strategies(commander, card_dict, price_data, max_price)

    # Create a very concise prompt
    prompt = f"""Analyze {commander}. Two win conditions:
- Strategy name
- Brief description
- 3 key cards
- Power level (1-10)"""

    try:
        # Create inputs and move to GPU
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate with reduced tokens
        outputs = model.generate(
            **inputs,
            max_length=256,  # Very short to avoid token issues
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        win_condition_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.warning(f"Error generating win conditions: {str(e)}")
        win_condition_text = f"Analysis of {commander} (generated strategies used instead)"

    # Define pre-set win conditions as fallback
    preset_win_conditions = [
        {
            'name': 'Storm Combo',
            'description': 'Cast multiple cheap spells in a turn to generate mana with Birgi, then finish with a big payoff spell.',
            'key_cards': ['Grapeshot', 'Mana Geyser', 'Seething Song', 'Jeska\'s Will', 'Runaway Steam-Kin', 'Grinning Ignus'],
            'power_level': 8
        },
        {
            'name': 'Wheel Effects',
            'description': 'Use Harnfel\'s "discard to draw" ability with wheel effects to cycle through your deck quickly.',
            'key_cards': ['Wheel of Fortune', 'Reforge the Soul', 'Magus of the Wheel', 'Past in Flames', 'Underworld Breach'],
            'power_level': 7
        },
        {
            'name': 'Artifact Combo',
            'description': 'Use cost reducers and Birgi\'s mana generation to chain artifact casts together.',
            'key_cards': ['Grinning Ignus', 'Sensei\'s Divining Top', 'Aetherflux Reservoir', 'Helm of Awakening', 'Skullclamp'],
            'power_level': 7
        },
        {
            'name': 'Red Aggro',
            'description': 'Cast numerous cheap red creatures and use Birgi\'s mana to overwhelm opponents.',
            'key_cards': ['Runaway Steam-Kin', 'Monastery Swiftspear', 'Lightning Bolt', 'Light Up the Stage', 'Reckless Impulse'],
            'power_level': 6
        }
    ]

    # Process and filter the strategies based on price
    model_strategies = []
    try:
        # Parse the win condition text to extract strategies
        sections = re.split(r'(?:\n\n|\n#+\s*|\n\d+\.?\s*)(Win Condition|Strategy|Approach)[:\s-]+', win_condition_text)

        # If we have sections after splitting
        if len(sections) > 1:
            for i in range(1, len(sections), 2):
                if i+1 < len(sections):
                    section_header = sections[i].strip()
                    section_content = sections[i+1].strip()

                    # Extract strategy details
                    strategy_name = section_header
                    if not strategy_name:
                        first_line = section_content.split('\n')[0].strip()
                        strategy_name = first_line

                    # Extract key cards - look for bullet points, numbers, or capital letters at line start
                    key_cards = []
                    description = ""
                    power_level = 0

                    for line in section_content.split('\n'):
                        line = line.strip()

                        # Look for power level rating
                        if re.search(r'(?:power|rating|score).*?(\d+)\s*(/|out of)\s*10', line.lower()):
                            match = re.search(r'(?:power|rating|score).*?(\d+)\s*(/|out of)\s*10', line.lower())
                            if match:
                                power_level = int(match.group(1))

                        # If line starts with a bullet, number, or capital letter, it might be a card
                        elif line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line) or re.match(r'^[A-Z]', line):
                            # Clean up formatting
                            card = re.sub(r'^[-*•\d.)\]]+\s*', '', line)

                            # Remove explanations after the card name
                            if ' - ' in card:
                                card = card.split(' - ')[0].strip()
                            elif ' – ' in card:
                                card = card.split(' – ')[0].strip()
                            elif ' — ' in card:
                                card = card.split(' — ')[0].strip()
                            elif ': ' in card:
                                card = card.split(': ')[0].strip()

                            # Remove brackets if present
                            card = re.sub(r'[\[\]]', '', card)

                            # Validate card exists in database and is within price range
                            if card in card_dict:
                                if max_price is None or price_data is None or card not in price_data or not price_data[card].get('usd') or float(price_data[card].get('usd', 0) or 0) <= max_price:
                                    key_cards.append(card)
                        else:
                            # If not a card or power level, it's part of the description
                            description += line + " "

                    # Only add if we found some key cards within price range
                    if key_cards:
                        model_strategies.append({
                            'name': strategy_name,
                            'description': description.strip(),
                            'key_cards': key_cards,
                            'power_level': power_level
                        })

        # If we successfully parsed strategies from the model, add them
        if model_strategies:
            # Combine custom strategies with model-generated ones
            win_conditions = custom_strategies + model_strategies
        else:
            win_conditions = custom_strategies

    except Exception as e:
        logging.warning(f"Error parsing win conditions: {str(e)}")
        win_conditions = custom_strategies

    # Add price data to strategies
    if price_data:
        for strategy in win_conditions:
            prices = []
            for card in strategy.get('key_cards', []):
                if card in price_data and price_data[card].get('usd'):
                    try:
                        prices.append(float(price_data[card]['usd']))
                    except (ValueError, TypeError):
                        pass

            if prices:
                avg_price = sum(prices) / len(prices)
                total_price = avg_price * 65 + 0.5 * 35  # Estimate for a full deck

                strategy['price_estimate'] = total_price

                # Categorize by price
                if total_price < 50:
                    strategy['price_category'] = "Budget"
                elif total_price < 200:
                    strategy['price_category'] = "Affordable"
                elif total_price < 500:
                    strategy['price_category'] = "Moderate"
                else:
                    strategy['price_category'] = "Premium"

    # If we somehow still don't have any strategies, fall back to presets
    if not win_conditions:
        # Filter preset conditions to match commander's color identity and price constraints
        valid_presets = []
        for condition in preset_win_conditions:
            valid_cards = [card for card in condition['key_cards']
                          if card in card_dict and
                          all(c in commander_identity for c in card_dict[card].get('color_identity', [])) and
                          (max_price is None or price_data is None or card not in price_data or
                           not price_data[card].get('usd') or float(price_data[card].get('usd', 0) or 0) <= max_price)]

            if valid_cards:
                valid_condition = condition.copy()
                valid_condition['key_cards'] = valid_cards
                valid_presets.append(valid_condition)

        if valid_presets:
            win_conditions = valid_presets
        else:
            win_conditions = [{
                'name': 'Generic Synergy',
                'description': f'A balanced approach utilizing {commander}\'s abilities.',
                'key_cards': [],
                'power_level': 5
            }]

    return win_conditions, win_condition_text

def optimize_for_cedh(deck, commander_identity):
    """
    Add powerful cEDH-level cards to the deck when bracket 5 is selected

    Args:
        deck (dict): Dictionary of cards and quantities
        commander_identity (list): Color identity of the commander

    Returns:
        dict: Updated deck with cEDH cards
    """
    # Define must-have cEDH staples based on color identity
    cedh_staples = {
        'W': ["Swords to Plowshares", "Path to Exile", "Enlightened Tutor", "Smothering Tithe", "Teferi's Protection"],
        'U': ["Force of Will", "Fierce Guardianship", "Mana Drain", "Rhystic Study", "Mystic Remora", "Cyclonic Rift"],
        'B': ["Demonic Tutor", "Vampiric Tutor", "Imperial Seal", "Ad Nauseam", "Necropotence", "Bolas's Citadel"],
        'R': ["Jeska's Will", "Deflecting Swat", "Underworld Breach", "Dockside Extortionist", "Gamble"],
        'G': ["Sylvan Library", "Birds of Paradise", "Carpet of Flowers", "Survival of the Fittest", "Elvish Mystic"]
    }

    # Colorless staples for any deck
    colorless_staples = [
        "Sol Ring", "Mana Crypt", "Chrome Mox", "Mox Diamond", "Mana Vault",
        "Ancient Tomb", "Command Tower", "Talisman of Dominance", "Sensei's Divining Top"
    ]

    # Build the list of staples based on color identity
    applicable_staples = colorless_staples.copy()
    for color in commander_identity:
        if color in cedh_staples:
            applicable_staples.extend(cedh_staples[color])

    # Cards to replace (lower power cards)
    to_remove = []
    for card in deck:
        card_info = card_dict.get(card, {})
        # Skip lands
        if 'Land' in card_info.get('types', []):
            continue
        # Skip if it's already a powerful card
        if card in applicable_staples or card in GAME_CHANGER_CARDS:
            continue
        to_remove.append(card)

    # Shuffle to randomize which cards get replaced
    random.shuffle(to_remove)

    # Add cEDH staples by replacing existing cards
    added_cedh_cards = []
    for staple in applicable_staples:
        # Check if card is in color identity and not already in deck
        card_info = card_dict.get(staple, {})
        card_identity = card_info.get('color_identity', [])
        if not all(c in commander_identity for c in card_identity):
            continue

        if staple not in deck and to_remove:
            card_to_remove = to_remove.pop()
            if card_to_remove in deck:
                qty = deck[card_to_remove]
                del deck[card_to_remove]
                deck[staple] = qty
                added_cedh_cards.append(staple)

        if len(added_cedh_cards) >= 15:  # Limit to 15 replacements
            break

    logging.info(f"Added {len(added_cedh_cards)} cEDH staples: {', '.join(added_cedh_cards)}")
    return deck

def generate_conditioned_deck(commander, tokenizer, model, win_condition, commander_identity, temp_boost=0, price_data=None, max_price=None):
    """
    Generate deck with GPU acceleration

    Args:
        commander (str): Name of the commander
        tokenizer: Tokenizer for the language model
        model: Language model
        win_condition (dict): Win condition information
        commander_identity (list): Color identity of the commander
        temp_boost (float, optional): Temperature boost for variation
        price_data (dict, optional): Dictionary of card pricing data
        max_price (float, optional): Maximum price for cards

    Returns:
        str: Generated deck text
    """
    # Create a more comprehensive prompt
    key_cards_str = ", ".join(win_condition.get('key_cards', [])[:5])

    prompt = f"""Generate a complete Commander deck for {commander} using a {win_condition['name']} strategy.
Include 99 cards plus the commander.
Key synergy cards: {key_cards_str}.
Format: 1x Card Name per line.
Include a balanced mana base with appropriate lands."""

    # Move to GPU
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate with sufficient tokens for a complete deck
    outputs = model.generate(
        **inputs,
        max_length=1024,  # Increased from 256 to allow full deck generation
        temperature=0.7 + temp_boost,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    conditioned_deck_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return conditioned_deck_text

def generate_random_deck(commander, commander_identity, land_quality='balanced', price_data=None, max_price=None):
    """
    Generate a random but coherent deck for the given commander

    Args:
        commander (str): Name of the commander
        commander_identity (list): Color identity of the commander
        land_quality (str, optional): Quality setting for lands
        price_data (dict, optional): Dictionary of card pricing data
        max_price (float, optional): Maximum price for cards

    Returns:
        dict: Dictionary of cards and quantities
    """
    logging.info(f"Generating random deck for {commander}")

    # Start with an empty deck
    deck = {}

    # Add commander if not already in deck
    if commander not in deck:
        deck[commander] = 1

    # Define card categories and their target percentages
    categories = {
        'Ramp': 0.10,     # 10% ramp effects
        'Draw': 0.10,     # 10% card draw
        'Removal': 0.10,  # 10% removal
        'Wrath': 0.05,    # 5% board wipes
        'Creatures': 0.25, # 25% creatures
        'Synergy': 0.15,  # 15% synergy pieces
        'Lands': 0.37     # 37% lands (including basic lands)
    }

    # Calculate target counts
    total_cards = 99  # 99 cards plus commander = 100
    card_counts = {category: int(percentage * total_cards) for category, percentage in categories.items()}

    # Adjust to ensure we hit exactly 99 cards
    remaining = total_cards - sum(card_counts.values())
    if remaining > 0:
        card_counts['Creatures'] += remaining

    # Add cards by category
    for category, count in card_counts.items():
        if category == 'Lands':
            # Handle lands separately with land quality
            continue

        logging.info(f"Adding {count} {category} cards")

        # Generate query based on category
        if category == 'Ramp':
            queries = ['type:artifact text:add mana', 'text:"add mana"', 'text:ritual']
        elif category == 'Draw':
            queries = ['text:"draw a card"', 'text:"draw cards"']
        elif category == 'Removal':
            queries = ['text:destroy target', 'text:exile target', 'text:damage to target']
        elif category == 'Wrath':
            queries = ['text:"destroy all"', 'text:"exile all"', 'text:"damage to all"']
        elif category == 'Creatures':
            queries = ['type:creature power>2', 'type:creature toughness>2', 'type:creature text:when']
        elif category == 'Synergy':
            # Generate synergy queries based on commander text
            queries = generate_synergy_queries(commander)
            if not queries:
                queries = ['type:instant', 'type:sorcery']

        # Add cards for each query
        added = 0
        exclude_list = list(deck.keys())

        for query in queries:
            if added >= count:
                break

            results = search_cards_by_criteria(query, commander_identity, exclude_cards=exclude_list, price_data=price_data, max_price=max_price)
            random.shuffle(results)  # Randomize the results

            # Add a portion of results
            to_add = min(len(results), count - added)
            for i in range(to_add):
                if i < len(results):
                    card_name = results[i]
                    deck[card_name] = 1
                    exclude_list.append(card_name)
                    added += 1

    # Add lands to reach target with appropriate quality
    lands_to_add = card_counts['Lands']

    # First add some non-basic lands based on land quality
    nonbasic_count = min(lands_to_add // 3, 12)  # Up to 12 non-basic lands
    nonbasic_lands = search_lands_by_quality(commander_identity, land_quality, exclude_cards=list(deck.keys()), count=nonbasic_count, price_data=price_data, max_price=max_price)

    for land in nonbasic_lands:
        if lands_to_add <= 0:
            break
        deck[land] = 1
        lands_to_add -= 1

    # Fill the rest with basic lands
    if lands_to_add > 0:
        # Determine which basic lands to add based on color identity
        land_types = []
        for color in commander_identity:
            if color in BASIC_LANDS:
                land_types.append(BASIC_LANDS[color])

        # If no colors or colorless, default to wastes or mountains
        if not land_types and 'C' in commander_identity:
            land_types = ['Wastes']
        elif not land_types:
            land_types = ['Mountain']  # Default to mountain if no color identity found

        # Add equal numbers of each basic land type
        per_land = lands_to_add // len(land_types)
        remainder = lands_to_add % len(land_types)

        for i, land in enumerate(land_types):
            qty = per_land + (1 if i < remainder else 0)
            if land in deck:
                deck[land] += qty
            else:
                deck[land] = qty

    return deck

def complete_deck(partial_deck, commander_name, commander_identity, win_condition=None, land_quality='balanced', price_data=None, max_price=None):
    """
    Complete deck with appropriate cards to reach 100 cards total

    Args:
        partial_deck (dict): Partial deck to complete
        commander_name (str): Name of the commander
        commander_identity (list): Color identity of the commander
        win_condition (dict, optional): Win condition information
        land_quality (str, optional): Quality setting for lands
        price_data (dict, optional): Dictionary of card pricing data
        max_price (float, optional): Maximum price for cards

    Returns:
        dict: Completed deck
    """
    logging.info(f"Completing deck with {len(partial_deck)} initial cards")

    # Create a copy of the partial deck
    completed_deck = dict(partial_deck)

    # Ensure commander is in deck
    if commander_name not in completed_deck:
        completed_deck[commander_name] = 1

    # Count total cards (not unique physical cards)
    total_cards = sum(completed_deck.values())
    logging.info(f"Current total: {total_cards} cards")

    # Calculate how many cards to add
    cards_needed = 100 - total_cards
    if cards_needed <= 0:
        return completed_deck

    logging.info(f"Need to add {cards_needed} more cards to reach 100")

    # Calculate current card type distribution
    type_counts = count_card_types(completed_deck)
    logging.info(f"Current type distribution: {type_counts}")

    # Target distribution for a typical EDH deck
    # These are percentage ranges, actual cards will be calculated
    target_distribution = {
        'Land': (0.36, 0.40),  # 36-40% lands
        'Creature': (0.25, 0.30),  # 25-30% creatures
        'Artifact': (0.05, 0.15),  # 5-15% artifacts
        'Enchantment': (0.05, 0.15),  # 5-15% enchantments
        'Planeswalker': (0.00, 0.05),  # 0-5% planeswalkers
        'Instant': (0.05, 0.15),  # 5-15% instants
        'Sorcery': (0.05, 0.15),  # 5-15% sorceries
    }

    # Calculate how many cards to add by type
    cards_to_add_by_type = {}
    remaining_needed = cards_needed

    # First, determine land count - this is the most critical
    current_lands = type_counts.get('Land', 0)

    # Calculate desired land target (37 is typical in EDH)
    land_target = 37
    # Adjust based on strategy - some strategies want more or less lands
    if win_condition:
        strategy_name = win_condition.get('name', '').lower()
        # Strategies that prefer more lands
        if any(x in strategy_name for x in ['landfall', 'land', 'ramp']):
            land_target = 40
        # Strategies that can work with fewer lands
        elif any(x in strategy_name for x in ['aggro', 'low curve', 'fast']):
            land_target = 34

    lands_needed = max(0, land_target - current_lands)
    lands_needed = min(lands_needed, cards_needed)  # Don't add more than we need total
    cards_to_add_by_type['Land'] = lands_needed
    remaining_needed -= lands_needed

    # Then calculate other types based on remaining needs
    if remaining_needed > 0:
        # Calculate creature ratio
        current_creatures = type_counts.get('Creature', 0)
        target_creatures = int(100 * target_distribution['Creature'][0])
        creatures_needed = max(0, target_creatures - current_creatures)
        creatures_needed = min(creatures_needed, remaining_needed // 2)  # No more than half remaining
        cards_to_add_by_type['Creature'] = creatures_needed
        remaining_needed -= creatures_needed

        # Distribute the rest proportionally
        other_categories = ['Artifact', 'Enchantment', 'Instant', 'Sorcery', 'Planeswalker']
        for category in other_categories:
            if remaining_needed <= 0:
                cards_to_add_by_type[category] = 0
                continue

            current_count = type_counts.get(category, 0)
            min_target = int(100 * target_distribution[category][0])

            # Calculate how many we need to add
            category_needed = max(0, min_target - current_count)
            # Don't add more than proportionally fair share of what's left
            category_share = remaining_needed // (len(other_categories) + 1)
            category_to_add = min(category_needed, category_share)

            cards_to_add_by_type[category] = category_to_add
            remaining_needed -= category_to_add

        # If we still have cards to add, put them in creatures or spells
        if remaining_needed > 0:
            # Prefer creatures for the remainder
            additional_creatures = min(remaining_needed, remaining_needed // 2 + 1)
            cards_to_add_by_type['Creature'] += additional_creatures
            remaining_needed -= additional_creatures

            # Distribute any remaining cards across spells
            spell_categories = ['Instant', 'Sorcery']
            for i in range(remaining_needed):
                category = spell_categories[i % len(spell_categories)]
                cards_to_add_by_type[category] += 1

    logging.info(f"Cards to add by type: {cards_to_add_by_type}")

    # Track cards to exclude from searches
    exclude_list = list(completed_deck.keys())

    # Add lands first (most important)
    if cards_to_add_by_type.get('Land', 0) > 0:
        lands_to_add = cards_to_add_by_type['Land']

        # Add some nonbasic lands
        nonbasic_count = min(lands_to_add // 2, 10)  # Up to 10 nonbasic lands
        if nonbasic_count > 0:
            nonbasic_lands = search_lands_by_quality(
                commander_identity,
                land_quality,
                exclude_cards=exclude_list,
                count=nonbasic_count,
                price_data=price_data,
                max_price=max_price
            )

            # Add nonbasic lands
            for land in nonbasic_lands:
                if land in completed_deck:
                    completed_deck[land] += 1
                else:
                    completed_deck[land] = 1
                exclude_list.append(land)
                lands_to_add -= 1

                if lands_to_add <= 0:
                    break

        # Add basic lands for the remainder
        if lands_to_add > 0:
            # Determine which basic lands to add
            basic_lands = []
            for color in commander_identity:
                if color in BASIC_LANDS:
                    basic_lands.append(BASIC_LANDS[color])

            # Default to Plains if no color identity
            if not basic_lands:
                basic_lands = ["Plains"]

            # Distribute basic lands
            per_type = lands_to_add // len(basic_lands)
            remainder = lands_to_add % len(basic_lands)

            for i, land in enumerate(basic_lands):
                to_add = per_type + (1 if i < remainder else 0)
                if to_add > 0:
                    if land in completed_deck:
                        completed_deck[land] += to_add
                    else:
                        completed_deck[land] = to_add

    # Add creatures next
    if cards_to_add_by_type.get('Creature', 0) > 0:
        creature_count = cards_to_add_by_type['Creature']
        logging.info(f"Adding {creature_count} creatures")

        # Determine creature query based on win condition
        creature_query = "type:creature"
        if win_condition:
            strategy = win_condition.get('name', '').lower()
            creature_query = f"type:creature for {strategy} strategy"

        # Get creature search results
        creatures = search_cards_by_criteria(
            creature_query,
            commander_identity,
            exclude_cards=exclude_list,
            n=creature_count * 3,  # Get more than we need for variety
            price_data=price_data,
            max_price=max_price
        )

        # Shuffle to randomize selection
        random.shuffle(creatures)

        # Add creatures up to needed count
        added = 0
        for creature in creatures:
            if added >= creature_count:
                break

            if creature in completed_deck:
                completed_deck[creature] += 1
            else:
                completed_deck[creature] = 1

            exclude_list.append(creature)
            added += 1

    # Add other card types
    other_types = ['Artifact', 'Enchantment', 'Instant', 'Sorcery', 'Planeswalker']
    for card_type in other_types:
        if cards_to_add_by_type.get(card_type, 0) <= 0:
            continue

        type_count = cards_to_add_by_type[card_type]
        logging.info(f"Adding {type_count} {card_type.lower()}s")

        # Build appropriate query
        query = f"type:{card_type.lower()}"
        if win_condition:
            strategy = win_condition.get('name', '').lower()
            query = f"{query} for {strategy} strategy"

        # Special case for planeswalkers - be more selective
        if card_type == 'Planeswalker' and type_count > 0:
            query = f"best planeswalker for {commander_name}"

        # Search for cards
        cards = search_cards_by_criteria(
            query,
            commander_identity,
            exclude_cards=exclude_list,
            n=type_count * 3,
            price_data=price_data,
            max_price=max_price
        )

        # Shuffle results
        random.shuffle(cards)

        # Add cards
        added = 0
        for card in cards:
            if added >= type_count:
                break

            if card in completed_deck:
                completed_deck[card] += 1
            else:
                completed_deck[card] = 1

            exclude_list.append(card)
            added += 1

    # Final check: ensure we have exactly 100 cards
    total_cards = sum(completed_deck.values())

    # If we have too many cards, remove the least important ones
    while total_cards > 100:
        # Find candidates to remove (non-essential cards)
        candidates = []
        for card, qty in completed_deck.items():
            if (card != commander_name and
                qty > 0 and
                card not in BASIC_LANDS.values() and
                (not win_condition or card not in win_condition.get('key_cards', []))):
                candidates.append(card)

        if candidates:
            # Remove a random non-essential card
            to_remove = random.choice(candidates)
            if completed_deck[to_remove] > 1:
                completed_deck[to_remove] -= 1
            else:
                del completed_deck[to_remove]
            logging.info(f"Removed excess card: {to_remove}")
        else:
            # No good candidates, reduce a basic land
            for land in BASIC_LANDS.values():
                if land in completed_deck and completed_deck[land] > 1:
                    completed_deck[land] -= 1
                    logging.info(f"Reduced basic land: {land}")
                    break

        # Recalculate total
        total_cards = sum(completed_deck.values())

    # If we need more cards, add basic lands
    if total_cards < 100:
        to_add = 100 - total_cards
        basic_land = BASIC_LANDS.get(commander_identity[0] if commander_identity else 'W', "Plains")

        if basic_land in completed_deck:
            completed_deck[basic_land] += to_add
        else:
            completed_deck[basic_land] = to_add

        logging.info(f"Added {to_add}x {basic_land} to reach 100 cards")

    # Verify correct card count
    final_count = sum(completed_deck.values())
    if final_count != 100:
        logging.warning(f"WARNING: Final deck has {final_count} cards instead of 100!")

    return completed_deck

def print_formatted_deck(deck, commander):
    """
    Print the deck in a nicely formatted way

    Args:
        deck (dict): Dictionary of cards and quantities
        commander (str): Name of the commander
    """
    print("\n--- Commander ---")
    print(f"1x {commander}")

    # Keep track of which cards we've displayed
    displayed_cards = set()

    print("\n--- Lands ---")
    for card, qty in sorted(deck.items()):
        if card == commander or card in displayed_cards:
            continue

        # Skip flip card faces
        if card in flip_card_mapping:
            continue

        card_info = card_dict.get(card, {})
        types = card_info.get('types', [])
        if 'Land' in types or card in BASIC_LANDS.values():
            print(f"{qty}x {card}")
            displayed_cards.add(card)

    print("\n--- Creatures ---")
    for card, qty in sorted(deck.items()):
        if card == commander or card in displayed_cards:
            continue
        card_info = card_dict.get(card, {})
        types = card_info.get('types', [])
        if 'Creature' in types and 'Land' not in types:
            print(f"{qty}x {card}")
            displayed_cards.add(card)

    print("\n--- Artifacts ---")
    for card, qty in sorted(deck.items()):
        if card == commander or card in displayed_cards:
            continue
        card_info = card_dict.get(card, {})
        types = card_info.get('types', [])
        if 'Artifact' in types and 'Creature' not in types and 'Land' not in types:
            print(f"{qty}x {card}")
            displayed_cards.add(card)

    print("\n--- Enchantments ---")
    for card, qty in sorted(deck.items()):
        if card == commander or card in displayed_cards:
            continue
        card_info = card_dict.get(card, {})
        types = card_info.get('types', [])
        if 'Enchantment' in types and 'Creature' not in types and 'Land' not in types:
            print(f"{qty}x {card}")
            displayed_cards.add(card)

    print("\n--- Planeswalkers ---")
    for card, qty in sorted(deck.items()):
        if card == commander or card in displayed_cards:
            continue
        card_info = card_dict.get(card, {})
        types = card_info.get('types', [])
        if 'Planeswalker' in types:
            print(f"{qty}x {card}")
            displayed_cards.add(card)

    print("\n--- Instants and Sorceries ---")
    for card, qty in sorted(deck.items()):
        if card == commander or card in displayed_cards:
            continue
        card_info = card_dict.get(card, {})
        types = card_info.get('types', [])
        if ('Instant' in types or 'Sorcery' in types) and 'Creature' not in types and 'Land' not in types:
            print(f"{qty}x {card}")
            displayed_cards.add(card)

    # Check for any remaining cards
    for card, qty in sorted(deck.items()):
        if card == commander or card in displayed_cards:
            continue
        print(f"{qty}x {card} (Other)")
        displayed_cards.add(card)

def print_formatted_deck_with_images(deck, commander):
    """
    Print the deck in a nicely formatted way with card images

    Args:
        deck (dict): Dictionary of cards and quantities
        commander (str): Name of the commander
    """
    from IPython.display import display, HTML
    import os

    # CSS for the card display
    css = """
    <style>
    .deck-section {
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        background-color: #f0f0f0;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 10px;
    }
    .card-item {
        display: flex;
        align-items: center;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        transition: all 0.3s ease;
    }
    .card-item:hover {
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
    }
    .card-image {
        width: 100px;
        height: 140px;
        object-fit: contain;
        margin-right: 10px;
        border-radius: 4px;
    }
    .card-info {
        flex: 1;
        font-size: 14px;
    }
    </style>
    """

    html_content = css + "<div class='deck-container'>"

    # Display commander
    html_content += "<div class='deck-section'>"
    html_content += "<div class='section-title'>Commander</div>"
    html_content += "<div class='card-grid'>"

    # Get commander image
    commander_image_url = get_card_image_url(commander)
    html_content += f"""
        <div class='card-item'>
            <img class='card-image' src='{commander_image_url}' alt='{commander}'/>
            <div class='card-info'>1x {commander}</div>
        </div>
    """
    html_content += "</div></div>"

    # Display each section
    sections = [
        ("Lands", lambda card: 'Land' in card_dict.get(card, {}).get('types', []) or card in BASIC_LANDS.values()),
        ("Creatures", lambda card: 'Creature' in card_dict.get(card, {}).get('types', []) and 'Land' not in card_dict.get(card, {}).get('types', [])),
        ("Artifacts", lambda card: 'Artifact' in card_dict.get(card, {}).get('types', []) and 'Creature' not in card_dict.get(card, {}).get('types', []) and 'Land' not in card_dict.get(card, {}).get('types', [])),
        ("Enchantments", lambda card: 'Enchantment' in card_dict.get(card, {}).get('types', []) and 'Creature' not in card_dict.get(card, {}).get('types', []) and 'Land' not in card_dict.get(card, {}).get('types', [])),
        ("Planeswalkers", lambda card: 'Planeswalker' in card_dict.get(card, {}).get('types', [])),
        ("Instants and Sorceries", lambda card: ('Instant' in card_dict.get(card, {}).get('types', []) or 'Sorcery' in card_dict.get(card, {}).get('types', [])) and 'Creature' not in card_dict.get(card, {}).get('types', []) and 'Land' not in card_dict.get(card, {}).get('types', []))
    ]

    # Process unique physical cards to avoid duplicates
    processed_cards = set([commander])

    for section_name, section_filter in sections:
        # Filter cards for this section
        section_cards = []
        for card, qty in sorted(deck.items()):
            if card in processed_cards:
                continue

            # Handle flip cards
            if card in flip_card_mapping:
                main_card = flip_card_mapping[card]
                if main_card in processed_cards:
                    continue
                if section_filter(main_card):
                    section_cards.append((main_card, qty))
                    processed_cards.add(main_card)
            else:
                if section_filter(card):
                    section_cards.append((card, qty))
                    processed_cards.add(card)

        if not section_cards:
            continue

        html_content += f"<div class='deck-section'>"
        html_content += f"<div class='section-title'>{section_name}</div>"
        html_content += "<div class='card-grid'>"

        for card, qty in section_cards:
            card_image_url = get_card_image_url(card)
            html_content += f"""
                <div class='card-item'>
                    <img class='card-image' src='{card_image_url}' alt='{card}'/>
                    <div class='card-info'>{qty}x {card}</div>
                </div>
            """

        html_content += "</div></div>"

    # Check for any remaining cards
    remaining_cards = []
    for card, qty in sorted(deck.items()):
        if card not in processed_cards and card != commander:
            remaining_cards.append((card, qty))
            processed_cards.add(card)

    if remaining_cards:
        html_content += "<div class='deck-section'>"
        html_content += "<div class='section-title'>Other</div>"
        html_content += "<div class='card-grid'>"

        for card, qty in remaining_cards:
            card_image_url = get_card_image_url(card)
            html_content += f"""
                <div class='card-item'>
                    <img class='card-image' src='{card_image_url}' alt='{card}'/>
                    <div class='card-info'>{qty}x {card}</div>
                </div>
            """

        html_content += "</div></div>"

    html_content += "</div>"
    display(HTML(html_content))

def get_card_image_url(card_name):
    """
    Get image URL for a card from local data or Scryfall API

    Args:
        card_name (str): Name of the card

    Returns:
        str: URL of the card image
    """
    import os
    import requests
    import time

    # First check if we already have the image URL in our cache
    if hasattr(get_card_image_url, "cache") and card_name in get_card_image_url.cache:
        return get_card_image_url.cache[card_name]

    # Check if cache directory exists, create if not
    cache_dir = "/content/drive/MyDrive/MTGData/card_images_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Check if we've saved the URL to a json file
    cache_file = f"{cache_dir}/image_urls.json"
    if os.path.exists(cache_file) and not hasattr(get_card_image_url, "cache"):
        try:
            import json
            with open(cache_file, 'r') as f:
                get_card_image_url.cache = json.load(f)
            if card_name in get_card_image_url.cache:
                return get_card_image_url.cache[card_name]
        except Exception as e:
            logging.warning(f"Error loading image cache: {e}")
            get_card_image_url.cache = {}
    elif not hasattr(get_card_image_url, "cache"):
        get_card_image_url.cache = {}

    # Otherwise, get the image URL from Scryfall
    try:
        # URL encode the card name for the API request
        encoded_name = requests.utils.quote(card_name)
        response = requests.get(f"https://api.scryfall.com/cards/named?exact={encoded_name}")

        # Add delay to respect Scryfall rate limits
        time.sleep(0.1)

        if response.status_code == 200:
            card_data = response.json()

            # Handle dual-faced cards
            if 'card_faces' in card_data and 'image_uris' not in card_data:
                # Use front face image
                image_url = card_data.get('card_faces', [{}])[0].get('image_uris', {}).get('normal')
            else:
                # Get the normal image URL
                image_url = card_data.get('image_uris', {}).get('normal')

            if image_url:
                # Save to cache
                get_card_image_url.cache[card_name] = image_url

                # Save cache to disk periodically
                if len(get_card_image_url.cache) % 10 == 0:
                    try:
                        import json
                        with open(cache_file, 'w') as f:
                            json.dump(get_card_image_url.cache, f)
                    except Exception as e:
                        logging.warning(f"Error saving image cache: {e}")

                return image_url

        # Return a placeholder if card not found
        placeholder = "https://cards.scryfall.io/large/front/0/c/0c082aa8-bf7f-47f2-baf8-43ad253fd7d7.jpg"
        get_card_image_url.cache[card_name] = placeholder
        return placeholder

    except Exception as e:
        logging.warning(f"Error fetching image for {card_name}: {e}")
        # Return a generic back of card image
        return "https://cards.scryfall.io/large/back/0/c/0c082aa8-bf7f-47f2-baf8-43ad253fd7d7.jpg"

def in_notebook_environment():
    """
    Check if code is running in Jupyter/Colab notebook

    Returns:
        bool: True if in notebook, False otherwise
    """
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
        return False
    except:
        return False

# ─── SAVE HELPER ────────────────────────────────────────────────────────────
def save_deck_to_db(supabase: Client,
                    commander: str,
                    win_condition: dict = None,
                    bracket: int = 0,
                    deck_list: dict = None,
                    user_email: str = None):
    """
    Inserts a generated deck into the Supabase `saved_decks` table.
    """
    try:
        # Handle None win_condition by creating a default
        if win_condition is None:
            win_condition = {
                'name': 'Random Deck',
                'description': f'A randomly generated deck for {commander}.',
                'key_cards': list(deck_list.keys())[:5] if deck_list else [],
                'power_level': bracket * 2
            }

        payload = {
            "commander": commander,
            "win_condition": win_condition,
            "bracket": bracket,
            "deck_list": deck_list,
            "user_email": user_email
        }

        res = supabase.table("saved_decks").insert(payload).execute()

        if hasattr(res, 'error') and res.error:
            logging.error(f"[Supabase] failed to save deck: {res.error.message}")
            return False
        else:
            if hasattr(res, 'data') and res.data and len(res.data) > 0:
                deck_id = res.data[0].get("id", "unknown")
                logging.info(f"[Supabase] deck saved (id={deck_id})")
            else:
                logging.info("[Supabase] deck saved successfully")
            return True
    except Exception as e:
        logging.error(f"[Supabase] exception when saving deck: {str(e)}")
        return False

# ─── MAIN FUNCTION ─────────────────────────────────────────────────────────
# ─── MAIN FUNCTION ─────────────────────────────────────────────────────────
def main():
    """Main function for the deck generator"""
    # Load card data
    cards = load_card_data()
    setup_retriever(cards)

    # Load price data
    price_data = load_price_data()

    # Load language model
    model_dir = '/media/sheltron/Backups&Games/Lambacloudbasedcomputing/all_folders/mtg-ai-framework/models/versions/First-Version'
    tokenizer, model = load_language_model(model_dir)

    # Initialize Supabase (always)
    try:
        dotenv.load_dotenv()
        supabase_url = os.getenv('SUPABASE_URL', 'https://hvsbfeyufrzmohzxsbcq.supabase.co')
        supabase_key = os.getenv('SUPABASE_KEY')
        if not supabase_key:
            print("Supabase key not found in environment variables.")
            supabase_key = getpass("Enter your Supabase API key: ")
        supabase = create_client(supabase_url, supabase_key)
        logging.info("Supabase client initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing Supabase: {e}")
        raise RuntimeError("Supabase initialization failed; please check your API key and network.")

    commander = input("Enter commander name: ")
    commander_identity = get_commander_identity(commander)

    if not commander_identity:
        logging.error(f"Could not determine color identity for {commander}")
        similar_commanders = search_similar_cards(commander, n=5)
        print(f"Did you mean one of these? {', '.join(similar_commanders)}")
        commander = input("Try entering commander name again: ")
        commander_identity = get_commander_identity(commander)
        if not commander_identity:
            print("Still couldn't find commander. Using default color identity (colorless).")
            commander_identity = []

    print(f"\nAnalyzing {commander} (Color Identity: {', '.join(commander_identity)})")

    # Price filter option
    print("\nSelect a price range for the deck:")
    price_ranges = [
        "No limit",
        "Budget (under $50)",
        "Affordable (under $200)",
        "Moderate (under $500)",
        "Premium (over $500)"
    ]
    for i, price_range in enumerate(price_ranges, 1):
        print(f"{i}. {price_range}")

    price_selection = input("Enter choice (1-5) [1]: ").strip()
    max_price = None
    if price_selection == "2":
        max_price = 50
    elif price_selection == "3":
        max_price = 200
    elif price_selection == "4":
        max_price = 500

    # Determine possible win conditions
    win_conditions, win_condition_text = determine_win_condition(
        commander, tokenizer, model, commander_identity, price_data, max_price
    )

    # Display win conditions
    print("\n=== Possible Win Conditions ===")
    print(win_condition_text)
    print("\n===========================")

    # Let user select a win condition or choose random
    if len(win_conditions) > 1:
        print("\nSelect a win condition to build around:")
        for i, wc in enumerate(win_conditions):
            key_cards_str = ", ".join(wc.get('key_cards', [])[:3])
            power_level = wc.get('power_level', 0)
            power_str = f" (Power: {power_level}/10)" if power_level > 0 else ""
            suggested_bracket = get_appropriate_bracket_for_win_condition(wc)
            bracket_str = f" - Bracket: {suggested_bracket} ({COMMANDER_BRACKETS[suggested_bracket]['name']})" if suggested_bracket > 0 else ""
            price_str = ''
            if 'price_estimate' in wc:
                price_str = f" - Est. Cost: ${wc['price_estimate']:.2f}"
            elif 'price_category' in wc:
                price_str = f" - {wc['price_category']}"
            print(f"{i+1}. {wc['name']}{power_str}{bracket_str}{price_str} - Key cards: {key_cards_str}...")

        print(f"{len(win_conditions)+1}. Random - Let me pick a random win condition")
        print(f"{len(win_conditions)+2}. Fully Random Deck - Generate a completely random deck")

        selection = input(f"Enter 1-{len(win_conditions)+2} [1]: ").strip()
        if not selection:
            selected_idx = 0
        else:
            try:
                selected_idx = int(selection) - 1
                if selected_idx < 0 or selected_idx > len(win_conditions) + 1:
                    selected_idx = 0
            except ValueError:
                selected_idx = 0

        # Handle random selection
        if selected_idx == len(win_conditions):
            selected_idx = random.randint(0, len(win_conditions) - 1)
            print(f"Randomly selected: {win_conditions[selected_idx]['name']}")

        # Handle fully random deck
        elif selected_idx == len(win_conditions) + 1:
            print(f"\nGenerating random deck for {commander}")

            print("\nSelect a land quality for the mana base:")
            for i, (key, land_setting) in enumerate(LAND_QUALITY_SETTINGS.items()):
                print(f"{i+1}. {land_setting['name']} - {land_setting['description']}")

            land_quality_selection = input("Enter choice (1-3) [2]: ").strip()
            try:
                land_choice = int(land_quality_selection) - 1
                if land_choice < 0 or land_choice >= len(LAND_QUALITY_SETTINGS):
                    land_choice = 1
            except ValueError:
                land_choice = 1

            land_quality = list(LAND_QUALITY_SETTINGS.keys())[land_choice]
            print(f"Selected land quality: {LAND_QUALITY_SETTINGS[land_quality]['name']}")

            random_deck = generate_random_deck(commander, commander_identity, land_quality, price_data, max_price)

            print("\nSelect a power level bracket for the deck:")
            for i, (bracket_num, bracket_info) in enumerate(COMMANDER_BRACKETS.items()):
                print(f"{bracket_num}. {bracket_info['name']} - {bracket_info['description']}")

            bracket_selection = input("Enter bracket (1-5) [2]: ").strip()
            try:
                selected_bracket = int(bracket_selection)
                if selected_bracket < 1 or selected_bracket > 5:
                    selected_bracket = 2
            except ValueError:
                selected_bracket = 2

            if selected_bracket == 5:
                random_deck = optimize_for_cedh(random_deck, commander_identity)

            filtered_deck, removed_cards = filter_deck_for_bracket(random_deck, selected_bracket)

            if removed_cards:
                print(f"Removed {len(removed_cards)} cards that don't comply with bracket {selected_bracket} restrictions:")
                for card, qty in removed_cards.items():
                    print(f"  {qty}x {card}")
                filtered_deck = replace_removed_cards(filtered_deck, removed_cards, commander_identity, selected_bracket, price_data, max_price)

            suggested_bracket, bracket_reasons = analyze_deck_for_bracket(filtered_deck)
            # Create a default win condition for random decks
            random_win_condition = {
                'name': 'Random Deck',
                'description': f'A randomly generated deck for {commander}.',
                'key_cards': list(filtered_deck.keys())[:5],  # First 5 cards as key cards
                'power_level': selected_bracket * 2  # Simple power level estimate based on bracket
            }
            type_counts = count_card_types(filtered_deck)
            print(f"\nRandom Deck for {commander} (Color Identity: {', '.join(commander_identity)}):")
            print(f"Bracket: {selected_bracket} - {COMMANDER_BRACKETS[selected_bracket]['name']}")
            print(f"Land Quality: {LAND_QUALITY_SETTINGS[land_quality]['name']}")
            if max_price:
                print(f"Price Limit: ${max_price}")
            print(f"Total cards: {sum(filtered_deck.values())}")
            print(f"Card type distribution: {type_counts}")

            print("\nBracket Analysis:")
            print(f"Recommended bracket based on deck contents: {suggested_bracket} - {COMMANDER_BRACKETS[suggested_bracket]['name']}")
            print("Reasons:")
            for reason in bracket_reasons:
                print(f"* {reason}")

            if in_notebook_environment():
                print_formatted_deck_with_images(filtered_deck, commander)
            else:
                print_formatted_deck(filtered_deck, commander)

            # Always save to Supabase
            try:
                save_deck_to_db(
                    supabase,
                    commander=commander,
                    win_condition=random_win_condition,  # Use our default win condition
                    bracket=selected_bracket,
                    deck_list=filtered_deck,
                    user_email=None
                )
                print("Deck saved to Supabase ✅")
            except Exception as e:
                logging.error(f"Error saving deck to Supabase: {e}")
                print("Warning: Error saving deck to Supabase")

            # Return early to avoid the normal deck generation flow
            return
    else:
        selected_idx = 0

    # Safely handle win condition selection - critical fix for IndexError
    if 0 <= selected_idx < len(win_conditions):
        selected_win_condition = win_conditions[selected_idx]
    else:
        # Create a fallback win condition if selection is out of range
        selected_win_condition = {
            'name': 'Default Strategy',
            'description': f'A balanced approach for {commander}.',
            'key_cards': [],
            'power_level': 5
        }

    # Determine appropriate bracket based on win condition
    suggested_bracket = get_appropriate_bracket_for_win_condition(selected_win_condition)

    print("\nSelect a power level bracket for the deck:")
    for i, (bracket_num, bracket_info) in enumerate(COMMANDER_BRACKETS.items()):
        if bracket_num == suggested_bracket:
            print(f"{bracket_num}. {bracket_info['name']} (RECOMMENDED) - {bracket_info['description']}")
        else:
            print(f"{bracket_num}. {bracket_info['name']} - {bracket_info['description']}")

    bracket_selection = input(f"Enter bracket (1-5) [{suggested_bracket}]: ").strip()
    try:
        selected_bracket = int(bracket_selection)
        if selected_bracket < 1 or selected_bracket > 5:
            selected_bracket = suggested_bracket
    except ValueError:
        selected_bracket = suggested_bracket

    print("\nSelect a land quality for the mana base:")
    for i, (key, land_setting) in enumerate(LAND_QUALITY_SETTINGS.items()):
        print(f"{i+1}. {land_setting['name']} - {land_setting['description']}")

    land_quality_selection = input("Enter choice (1-3) [2]: ").strip()
    try:
        land_choice = int(land_quality_selection) - 1
        if land_choice < 0 or land_choice >= len(LAND_QUALITY_SETTINGS):
            land_choice = 1
    except ValueError:
        land_choice = 1

    land_quality = list(LAND_QUALITY_SETTINGS.keys())[land_choice]
    print(f"Selected land quality: {LAND_QUALITY_SETTINGS[land_quality]['name']}")

    if 'price_estimate' in selected_win_condition:
        print(f"Estimated Deck Cost: ${selected_win_condition['price_estimate']:.2f}")
    elif 'price_category' in selected_win_condition:
        print(f"Price Category: {selected_win_condition['price_category']}")

    print(f"\nGenerating deck for {commander} with '{selected_win_condition['name']}' win condition")
    print(f"Using bracket {selected_bracket} - {COMMANDER_BRACKETS[selected_bracket]['name']}")
    if max_price:
        print(f"Price Limit: ${max_price}")

    use_variation = input("Would you like more variety in the generated deck? (y/n) [n]: ").strip().lower()
    temp_boost = 0.2 if use_variation == 'y' else 0.0

    deck_text = generate_conditioned_deck(commander, tokenizer, model,
                                          selected_win_condition,
                                          commander_identity, temp_boost,
                                          price_data, max_price)
    initial_deck = extract_deck(deck_text)

    for card in selected_win_condition.get('key_cards', []):
        if card not in initial_deck and card != commander:
            card_info = card_dict.get(card)
            if card_info and all(c in commander_identity for c in card_info.get('color_identity', [])):
                initial_deck[card] = 1

    valid_deck, invalid_cards = validate_deck_identity(initial_deck, commander_identity)
    final_deck = complete_deck(valid_deck, commander, commander_identity,
                               selected_win_condition, land_quality,
                               price_data, max_price)

    if selected_bracket == 5:
        final_deck = optimize_for_cedh(final_deck, commander_identity)

    filtered_deck, removed_cards = filter_deck_for_bracket(final_deck.copy(), selected_bracket)

    if removed_cards:
        print(f"Removed {len(removed_cards)} cards that don't comply with bracket {selected_bracket} restrictions:")
        for card, qty in removed_cards.items():
            print(f"  {qty}x {card}")
        filtered_deck = replace_removed_cards(filtered_deck, removed_cards, commander_identity, selected_bracket, price_data, max_price)

    if commander not in filtered_deck:
        filtered_deck[commander] = 1

    suggested_bracket, bracket_reasons = analyze_deck_for_bracket(filtered_deck, commander)

    if price_data:
        total_price = 0
        card_with_price = 0
        for card, qty in filtered_deck.items():
            if card in price_data and price_data[card].get('usd'):
                try:
                    card_price = float(price_data[card]['usd'])
                    total_price += card_price * qty
                    card_with_price += qty
                except (ValueError, TypeError):
                    pass
        if card_with_price:
            print(f"\nEstimated Deck Value: ${total_price:.2f}")
            print(f"Average price per card: ${total_price/card_with_price:.2f}")

    type_counts = count_card_types(filtered_deck)
    print(f"\nFinal Deck for {commander} (Color Identity: {', '.join(commander_identity)})")
    print(f"Strategy: {selected_win_condition['name']}")
    print(f"Bracket: {selected_bracket} - {COMMANDER_BRACKETS[selected_bracket]['name']}")
    print(f"Land Quality: {LAND_QUALITY_SETTINGS[land_quality]['name']}")
    if max_price:
        print(f"Price Limit: ${max_price}")
    print(f"Total physical cards: {count_unique_physical_cards(filtered_deck)}")
    print(f"Card type distribution: {type_counts}")

    print("\nBracket Analysis:")
    print(f"Recommended bracket: {suggested_bracket} - {COMMANDER_BRACKETS[suggested_bracket]['name']}")
    for reason in bracket_reasons:
        print(f"* {reason}")

    if in_notebook_environment():
        print_formatted_deck(filtered_deck, commander)
        print_formatted_deck_with_images(filtered_deck, commander)
    else:
        print_formatted_deck(filtered_deck, commander)

    # Always save to Supabase with proper error handling
    try:
        save_deck_to_db(
            supabase,
            commander=commander,
            win_condition=selected_win_condition,
            bracket=selected_bracket,
            deck_list=filtered_deck,
            user_email=None
        )
        print("Deck saved to Supabase ✅")
    except Exception as e:
        logging.error(f"Error saving deck to Supabase: {e}")
        print(f"Warning: Error saving deck to Supabase: {str(e)}")

if __name__ == '__main__':
    main()