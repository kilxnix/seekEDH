import os
import re
import json
import logging
import numpy as np
from flask import request, jsonify
from src.api_server import app
from src.rag_system import MTGRetrievalSystem
from src.rag_instance import rag_system

# Configure logging
logger = logging.getLogger("UniversalSearch")

# Import the RAG system instance from update_api_server
from src.update_api_server import rag_system

class UniversalSearchHandler:
    """Handler for universal natural language search queries"""
    
    def __init__(self, rag_system):
        self.rag = rag_system
        # Define query patterns for classification
        self.query_patterns = [
            {
                'type': 'similar_cards',
                'patterns': [
                    r'(?:find|show|get|search)\s+(?:cards|card)?\s*(?:like|similar\s+to)\s+(.+)',
                    r'cards?\s+(?:like|similar\s+to)\s+(.+)',
                    r'what\s+(?:cards?|else)\s+(?:is|are)\s+(?:like|similar\s+to)\s+(.+)'
                ],
                'extract_params': self._extract_similar_cards_params,
                'handler': self._handle_similar_cards
            },
            {
                'type': 'card_synergy',
                'patterns': [
                    r'(?:synergy|interaction|combo|work together)\s+(?:between|with|for)\s+(.+)',
                    r'what\s+cards?\s+(?:synergize|work)\s+(?:well\s+)?with\s+(.+)',
                    r'(?:cards?|synergies)\s+for\s+(.+)',
                ],
                'extract_params': self._extract_synergy_params,
                'handler': self._handle_card_synergy
            },
            {
                'type': 'deck_improvement',
                'patterns': [
                    r'(?:improve|upgrade|enhance)\s+(?:my)?\s*(.*?)\s*deck',
                    r'make\s+(?:my)?\s*(.*?)\s*deck\s+better',
                    r'what\s+(?:to|should\s+I)\s+add\s+(?:to)?\s+(?:my)?\s*(.*?)\s*deck'
                ],
                'extract_params': self._extract_deck_improvement_params,
                'handler': self._handle_deck_improvement
            },
            {
                'type': 'mana_curve',
                'patterns': [
                    r'(?:mana|curve).*?(?:if|with|when).*?add(?:ing)?\s+(.+)',
                    r'mana\s+curve\s+for\s+(.+)'
                ],
                'extract_params': self._extract_mana_curve_params,
                'handler': self._handle_mana_curve
            },
            {
                'type': 'rules_interaction',
                'patterns': [
                    r'(?:rules|rule|interaction).*?(?:between|with)\s+(.+)',
                    r'how\s+(?:do|does)\s+(.+?)\s+(?:work|interact)',
                    r'explain\s+(?:how|why|the\s+interaction)\s+(.+?)'
                ],
                'extract_params': self._extract_rules_interaction_params,
                'handler': self._handle_rules_interaction
            },
            {
                'type': 'budget_alternatives',
                'patterns': [
                    r'(?:budget|cheap|affordable).*?(?:alternative|option|replacement).*?(?:to|for)\s+(.+)',
                    r'(?:alternative|option|replacement).*?(?:cheaper|less\s+expensive).*?(?:than|to|for)\s+(.+)'
                ],
                'extract_params': self._extract_budget_alternatives_params,
                'handler': self._handle_budget_alternatives
            },
            {
                'type': 'generate_deck',
                'patterns': [
                    r'(?:generate|create|build)\s+(?:a|an)?\s*(.+?)\s*deck',
                    r'(?:make|give\s+me)\s+(?:a|an)?\s*(.+?)\s*deck',
                ],
                'extract_params': self._extract_generate_deck_params,
                'handler': self._handle_generate_deck
            },
            {
                'type': 'card_info',
                'patterns': [
                    r'(?:info|information|details|text|oracle)\s+(?:about|for|on)\s+(.+)',
                    r'what\s+(?:does|is|are)\s+(.+?)\s+(?:do|say|display|show)',
                    r'show\s+me\s+(.+)'
                ],
                'extract_params': self._extract_card_info_params,
                'handler': self._handle_card_info
            }
        ]
        
        # Add a general fallback handler
        self.fallback_handler = self._handle_general_search
    
    def classify_query(self, query_text):
        """Classify the query to determine which handler to use"""
        for pattern_group in self.query_patterns:
            for pattern in pattern_group['patterns']:
                match = re.search(pattern, query_text, re.IGNORECASE)
                if match:
                    # Extract parameters using the pattern group's extractor
                    params = pattern_group['extract_params'](match, query_text)
                    return {
                        'type': pattern_group['type'],
                        'params': params,
                        'handler': pattern_group['handler']
                    }
        
        # If no pattern matches, use a general search approach
        return {
            'type': 'general_search',
            'params': {'query': query_text},
            'handler': self.fallback_handler
        }
    
    def process_query(self, query_text, filters=None):
        """Process a natural language query"""
        try:
            # Apply any provided filters
            filters = filters or {}
            
            # Classify the query
            classification = self.classify_query(query_text)
            
            # Log the query classification
            logger.info(f"Query '{query_text}' classified as {classification['type']}")
            
            # Add filters to params
            classification['params'].update(filters)
            
            # Call the appropriate handler
            return classification['handler'](classification['params'], query_text)
            
        except Exception as e:
            logger.error(f"Error processing query '{query_text}': {e}")
            return {
                'success': False,
                'error': f"Error processing query: {str(e)}"
            }
    
    def _extract_similar_cards_params(self, match, full_query):
        """Extract parameters for similar cards queries"""
        card_name = match.group(1).strip()
        return {'card_name': card_name}
    
    def _handle_similar_cards(self, params, original_query):
        """Handle queries for similar cards"""
        card_name = params.get('card_name')
        if not card_name:
            return {
                'success': False,
                'error': "Couldn't determine which card to find similar cards for"
            }
        
        # Get card colors filter if present
        colors = params.get('colors', [])
        card_type = params.get('card_type')
        max_price = params.get('max_price')
        
        try:
            # First try to find by text similarity
            similar_by_text = self.rag.retrieve_cards_by_text(f"similar to {card_name}", top_k=10)
            
            # Then try to find by price similarity
            similar_by_price = self.rag.retrieve_cards_by_price(card_name, top_k=5)
            
            # Combine results, prioritizing text similarity
            result_cards = []
            
            # Add text similarity results first
            seen_cards = set()
            for card in similar_by_text:
                if self._passes_filters(card, colors, card_type, max_price):
                    result_cards.append(card)
                    seen_cards.add(card['name'])
            
            # Add price similarity results that weren't already included
            for card in similar_by_price:
                if card['name'] not in seen_cards and self._passes_filters(card, colors, card_type, max_price):
                    # Mark these as price-based matches
                    card['match_type'] = 'price'
                    result_cards.append(card)
                    seen_cards.add(card['name'])
            
            # Calculate enhanced relevance scores that combine semantic and price similarity
            for card in result_cards:
                # Start with semantic similarity score (0-1 range)
                base_score = card.get('similarity', 0.5)
                
                # Adjust score based on factors:
                
                # 1. Card type match bonus
                if card_type and card_type.lower() in card.get('type_line', '').lower():
                    base_score *= 1.2  # 20% bonus for matching requested type
                
                # 2. Color match bonus
                if colors and all(color in card.get('color_identity', []) for color in colors):
                    base_score *= 1.15  # 15% bonus for perfect color match
                
                # 3. Price match consideration
                if card.get('match_type') == 'price':
                    # Slightly lower base score for price matches
                    base_score *= 0.8
                
                # Cap at 1.0
                card['relevance_score'] = min(1.0, base_score)
            
            # Sort by enhanced relevance score
            result_cards.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return {
                'success': True,
                'query_type': f"Find cards similar to {card_name}",
                'cards': result_cards,
                'explanation': f"Showing cards that are similar to {card_name} based on card text, functionality, and/or price."
            }
            
        except Exception as e:
            logger.error(f"Error handling similar cards query: {e}")
            return {
                'success': False,
                'error': f"Error finding similar cards: {str(e)}"
            }
    
    def _extract_synergy_params(self, match, full_query):
        """Extract parameters for card synergy queries"""
        cards_text = match.group(1).strip()
        
        # Try to extract multiple card names
        if ',' in cards_text or ' and ' in cards_text:
            cards = re.split(r',|\s+and\s+', cards_text)
            cards = [card.strip() for card in cards if card.strip()]
        else:
            cards = [cards_text]
            
        return {'cards': cards}
    
    def _handle_card_synergy(self, params, original_query):
        """Handle queries for card synergies"""
        cards = params.get('cards', [])
        if not cards:
            return {
                'success': False,
                'error': "Couldn't determine which cards to find synergies for"
            }
        
        try:
            # Call RAG synergy search endpoint
            synergy_data = {
                'cards': cards,
                'count': 15
            }
            
            # Use the RAG system's synergy search function directly
            from src.update_api_server import calculate_card_synergies
            synergistic_cards = calculate_card_synergies(cards, 15)
            
            # Extract and format results
            formatted_results = []
            for item in synergistic_cards:
                card = item['card']
                formatted_results.append({
                    "name": card["name"],
                    "similarity_score": item["similarity_score"],
                    "rules_score": item["rules_score"],
                    "total_score": item["total_score"],
                    "relevance_score": item["total_score"],  # Use total score as relevance score
                    "oracle_text": card.get("oracle_text", ""),
                    "type_line": card.get("type_line", ""),
                    "mana_cost": card.get("mana_cost", ""),
                    "color_identity": card.get("color_identity", []),
                    "price_usd": card.get("prices_usd", 0),
                    "explanations": item.get("explanations", [])
                })
            
            return {
                'success': True,
                'query_type': f"Find cards that synergize with {', '.join(cards)}",
                'cards': formatted_results,
                'explanation': f"Showing cards that synergize well with {', '.join(cards)} based on game mechanics and card interactions."
            }
            
        except Exception as e:
            logger.error(f"Error handling card synergy query: {e}")
            return {
                'success': False,
                'error': f"Error finding synergies: {str(e)}"
            }
    
    def _extract_deck_improvement_params(self, match, full_query):
        """Extract parameters for deck improvement queries"""
        commander_or_theme = match.group(1).strip() if match.group(1) else ""
        
        # Try to extract budget constraint
        budget_match = re.search(r'(?:under|less\s+than|below|budget\s+of|max|maximum)\s+\$?(\d+)', full_query, re.IGNORECASE)
        budget = int(budget_match.group(1)) if budget_match else None
        
        return {
            'commander_or_theme': commander_or_theme,
            'budget': budget
        }
    
    def _handle_deck_improvement(self, params, original_query):
        """Handle queries for deck improvement suggestions"""
        commander_or_theme = params.get('commander_or_theme', '')
        budget = params.get('budget')
        colors = params.get('colors', [])
        
        try:
            # Determine whether it's a commander or a theme
            is_commander = False
            if commander_or_theme:
                # Check if it's a known commander card
                if self.rag.db.is_connected:
                    response = self.rag.db.client.table("mtg_cards").select(
                        "id"
                    ).eq("name", commander_or_theme).limit(1).execute()
                    
                    if response.data:
                        is_commander = True
            
            # Build appropriate search query
            if is_commander:
                strategy = f"Improve {commander_or_theme} commander deck"
                commander = commander_or_theme
            else:
                strategy = f"Improve {commander_or_theme} deck" if commander_or_theme else "Improve commander deck"
                commander = None
            
            # Generate deck recommendations
            deck_recommendation = self.rag.generate_deck_recommendation(
                strategy=strategy,
                commander=commander,
                budget=budget,
                colors=colors
            )
            
            # Format the results
            if 'cards' in deck_recommendation:
                for card in deck_recommendation['cards']:
                    # Calculate a relevance score based on how well it fits the strategy
                    card['relevance_score'] = 0.85  # Base score for recommended cards
                
                result = {
                    'success': True,
                    'query_type': f"Improve {'commander ' if is_commander else ''}{commander_or_theme} deck",
                    'cards': deck_recommendation['cards'],
                    'explanation': f"Here are recommended cards to improve your {commander_or_theme} deck."
                }
                
                # Add budget information if provided
                if budget:
                    result['explanation'] += f" Budget constraint: ${budget}."
                
                return result
            else:
                return {
                    'success': False,
                    'error': "Couldn't generate deck improvement recommendations"
                }
                
        except Exception as e:
            logger.error(f"Error handling deck improvement query: {e}")
            return {
                'success': False,
                'error': f"Error generating improvement suggestions: {str(e)}"
            }
    
    def _extract_mana_curve_params(self, match, full_query):
        """Extract parameters for mana curve queries"""
        card_name = match.group(1).strip()
        
        # Try to extract context about the existing deck
        deck_context_match = re.search(r'(?:my|in\s+(?:a|my|the))?\s*(.+?)\s*deck', full_query, re.IGNORECASE)
        deck_context = deck_context_match.group(1).strip() if deck_context_match else ""
        
        return {
            'card_name': card_name,
            'deck_context': deck_context
        }
    
    def _handle_mana_curve(self, params, original_query):
        """Handle queries about mana curve impact"""
        card_name = params.get('card_name', '')
        deck_context = params.get('deck_context', '')
        
        try:
            # Get information about the card
            card_data = None
            if self.rag.db.is_connected:
                response = self.rag.db.client.table("mtg_cards").select(
                    "*"
                ).eq("name", card_name).limit(1).execute()
                
                if response.data:
                    card_data = response.data[0]
            
            if not card_data:
                return {
                    'success': False,
                    'error': f"Couldn't find card: {card_name}"
                }
            
            # Get the card's mana value
            mana_value = card_data.get('cmc', 0)
            
            # Generate an explanation based on the mana value
            explanation = f"{card_name} has a mana value of {mana_value}. "
            
            if mana_value <= 2:
                explanation += "This is a low-cost card that will help lower your mana curve, making your deck faster and more consistent in the early game."
            elif mana_value <= 4:
                explanation += "This is a mid-range card that fits well in most mana curves, providing value in the mid-game."
            else:
                explanation += "This is a high-cost card that will shift your mana curve upward. Make sure your deck has enough ramp and early plays to support casting this."
            
            # Suggest complementary cards based on the mana value
            complementary_cards = []
            
            if mana_value >= 5:
                # For high-cost cards, suggest ramp
                ramp_cards = self.rag.retrieve_cards_by_text("ramp mana acceleration", top_k=5)
                complementary_cards.extend(ramp_cards)
                explanation += " Consider including more mana acceleration to help cast this card consistently."
            
            # Get the color identity to recommend complementary cards
            color_identity = card_data.get('color_identity', [])
            
            # Add the card itself to the response
            result_cards = [card_data]
            
            # Add complementary cards
            for card in complementary_cards:
                # Skip if it's the same as the queried card
                if card['name'] == card_name:
                    continue
                    
                # Set relevance score based on how well it complements
                card['relevance_score'] = 0.75  # Base score for complementary cards
                
                # Add it to results
                result_cards.append(card)
            
            return {
                'success': True,
                'query_type': f"Mana curve impact of adding {card_name}" + (f" to {deck_context} deck" if deck_context else ""),
                'cards': result_cards,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"Error handling mana curve query: {e}")
            return {
                'success': False,
                'error': f"Error analyzing mana curve: {str(e)}"
            }
    
    def _extract_rules_interaction_params(self, match, full_query):
        """Extract parameters for rules interaction queries"""
        interaction_text = match.group(1).strip()
        
        # Try to extract card names
        if ',' in interaction_text or ' and ' in interaction_text or ' with ' in interaction_text:
            # Split by common separators
            cards = re.split(r',|\s+and\s+|\s+with\s+', interaction_text)
            cards = [card.strip() for card in cards if card.strip()]
        else:
            cards = [interaction_text]
            
        return {'cards': cards}
    
    def _handle_rules_interaction(self, params, original_query):
        """Handle queries about rules interactions"""
        cards = params.get('cards', [])
        if not cards:
            return {
                'success': False,
                'error': "Couldn't determine which cards to explain rules for"
            }
        
        try:
            # Use the rules interaction endpoint
            from src.update_api_server import rules_interaction
            
            # Direct integration with the rules_interaction function
            interaction_data = {
                'cards': cards
            }
            
            # Call RAG rules interaction function directly
            from src.update_api_server import initialize_mechanics_mappings, load_rules_data, get_card_data, get_card_mechanics, find_related_rules
            
            # Ensure rules data is loaded
            if not hasattr(self, 'rules_cache') or not self.rules_cache.get('loaded', False):
                load_rules_data()
            
            # Ensure mechanics mappings are initialized
            initialize_mechanics_mappings()
            
            # Get card data for each card
            card_data_map = {}
            for card_name in cards:
                card_data = get_card_data(card_name)
                card_data_map[card_name] = card_data
            
            # Get mechanics for each card
            card_mechanics_map = {}
            for card_name, card_data in card_data_map.items():
                mechanics = get_card_mechanics(card_data)
                card_mechanics_map[card_name] = mechanics
            
            # Find rules related to these mechanics
            all_mechanics = []
            for mechanics in card_mechanics_map.values():
                all_mechanics.extend(mechanics)
            
            related_rules = find_related_rules(all_mechanics)
            
            # Format the response
            explanation = f"Here's how {', '.join(cards)} interact according to the MTG rules:"
            
            # Include specific interactions if found
            if related_rules:
                explanation += "\n\n"
                for rule in related_rules[:3]:  # Include top 3 rules
                    explanation += f"Rule {rule['rule_number']}: {rule['text']}\n\n"
            
            # Include card data in response
            result_cards = []
            for card_name, card_data in card_data_map.items():
                # Add explanation based on mechanics
                card_data['explanations'] = [f"Has mechanics: {', '.join(card_mechanics_map[card_name])}"]
                
                # Add relevance score
                card_data['relevance_score'] = 1.0  # Max score for directly queried cards
                
                result_cards.append(card_data)
            
            return {
                'success': True,
                'query_type': f"Rules interaction between {', '.join(cards)}",
                'cards': result_cards,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"Error handling rules interaction query: {e}")
            return {
                'success': False,
                'error': f"Error explaining rules interaction: {str(e)}"
            }
    
    def _extract_budget_alternatives_params(self, match, full_query):
        """Extract parameters for budget alternatives queries"""
        card_name = match.group(1).strip()
        
        # Try to extract budget constraint
        budget_match = re.search(r'(?:under|less\s+than|below|budget\s+of|max|maximum)\s+\$?(\d+)', full_query, re.IGNORECASE)
        max_price = float(budget_match.group(1)) if budget_match else 10.0  # Default to $10 if not specified
        
        return {
            'card_name': card_name,
            'max_price': max_price
        }
    
    def _handle_budget_alternatives(self, params, original_query):
        """Handle queries for budget alternatives to cards"""
        card_name = params.get('card_name', '')
        max_price = params.get('max_price', 10.0)
        colors = params.get('colors', [])
        
        try:
            # First get the original card to know what to find alternatives for
            original_card = None
            if self.rag.db.is_connected:
                response = self.rag.db.client.table("mtg_cards").select(
                    "*"
                ).eq("name", card_name).limit(1).execute()
                
                if response.data:
                    original_card = response.data[0]
            
            if not original_card:
                return {
                    'success': False,
                    'error': f"Couldn't find card: {card_name}"
                }
            
            # Get the original card's price
            original_price = float(original_card.get('prices_usd', 0) or 0)
            
            # If the original card is already within budget, inform the user
            if original_price <= max_price:
                return {
                    'success': True,
                    'query_type': f"Budget alternatives to {card_name}",
                    'cards': [original_card],
                    'explanation': f"{card_name} already costs ${original_price:.2f}, which is within your budget of ${max_price:.2f}."
                }
            
            # Find alternatives based on text similarity
            alternatives_by_text = self.rag.retrieve_cards_by_text(
                f"similar to {card_name} but cheaper", 
                top_k=10
            )
            
            # Find alternatives based on price similarity
            alternatives_by_price = self.rag.retrieve_cards_by_price(
                card_name, 
                top_k=10
            )
            
            # Combine and filter results
            all_alternatives = []
            seen_cards = set()
            
            # Process text-based alternatives first
            for card in alternatives_by_text:
                price = float(card.get('prices_usd', 0) or 0)
                
                # Only include cards cheaper than the original and within budget
                if (price < original_price and price <= max_price and 
                    card['name'] != card_name and card['name'] not in seen_cards):
                    
                    # Apply color filter if specified
                    if not colors or any(color in card.get('color_identity', []) for color in colors):
                        # Calculate relevance score based on price and similarity
                        price_ratio = 1 - (price / original_price)  # Higher for cheaper cards
                        similarity = card.get('similarity', 0.5)
                        
                        # Combined score formula: 60% similarity, 40% price ratio
                        card['relevance_score'] = 0.6 * similarity + 0.4 * price_ratio
                        
                        # Add explanation of price difference
                        card['price_difference'] = original_price - price
                        card['price_percentage'] = (price / original_price) * 100
                        
                        all_alternatives.append(card)
                        seen_cards.add(card['name'])
            
            # Add price-based alternatives that weren't already included
            for card in alternatives_by_price:
                price = float(card.get('prices_usd', 0) or 0)
                
                # Only include cards cheaper than the original and within budget
                if (price < original_price and price <= max_price and 
                    card['name'] != card_name and card['name'] not in seen_cards):
                    
                    # Apply color filter if specified
                    if not colors or any(color in card.get('color_identity', []) for color in colors):
                        # Calculate relevance score based on price discount
                        price_ratio = 1 - (price / original_price)  # Higher for cheaper cards
                        
                        # For price-based matches, base score on price ratio
                        card['relevance_score'] = 0.3 + (0.5 * price_ratio)  # Range: 0.3-0.8
                        
                        # Add explanation of price difference
                        card['price_difference'] = original_price - price
                        card['price_percentage'] = (price / original_price) * 100
                        
                        all_alternatives.append(card)
                        seen_cards.add(card['name'])
            
            # Sort by relevance score
            all_alternatives.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Add the original card first for comparison
            result_cards = [original_card]
            
            # Include the alternatives with explanations
            for card in all_alternatives[:10]:  # Limit to top 10
                # Add explanation text
                card['explanation'] = f"${float(card.get('prices_usd', 0) or 0):.2f} ({card.get('price_percentage', 0):.0f}% of {card_name}'s price)"
                result_cards.append(card)
            
            return {
                'success': True,
                'query_type': f"Budget alternatives to {card_name} under ${max_price:.2f}",
                'cards': result_cards,
                'explanation': f"Found {len(all_alternatives)} budget alternatives to {card_name} (${original_price:.2f}) under ${max_price:.2f}."
            }
            
        except Exception as e:
            logger.error(f"Error handling budget alternatives query: {e}")
            return {
                'success': False,
                'error': f"Error finding budget alternatives: {str(e)}"
            }
    
    def _extract_generate_deck_params(self, match, full_query):
        """Extract parameters for deck generation queries"""
        theme = match.group(1).strip()
        
        # Try to extract budget constraint
        budget_match = re.search(r'(?:under|less\s+than|below|budget\s+of|max|maximum)\s+\$?(\d+)', full_query, re.IGNORECASE)
        budget = int(budget_match.group(1)) if budget_match else None
        
        # Try to extract commander if specified
        commander_match = re.search(r'(?:with|using|for)\s+(?:commander\s+)?([^,]+?)(?:\s+as\s+(?:the\s+)?commander)?(?:\s+and|\s+under|\s+with|\s*$)', full_query, re.IGNORECASE)
        commander = commander_match.group(1).strip() if commander_match else None
        
        return {
            'theme': theme,
            'budget': budget,
            'commander': commander
        }
    
    def _handle_generate_deck(self, params, original_query):
        """Handle queries for generating decks"""
        theme = params.get('theme', '')
        budget = params.get('budget')
        commander = params.get('commander')
        colors = params.get('colors', [])
        
        try:
            # Check if the commander exists
            if commander and self.rag.db.is_connected:
                response = self.rag.db.client.table("mtg_cards").select(
                    "id, name"
                ).eq("name", commander).limit(1).execute()
                
                if not response.data:
                    # If not found, search for similar commanders
                    similar_commanders = self.rag.retrieve_cards_by_text(
                        f"legendary creature {commander}", 
                        top_k=5
                    )
                    
                    # Filter to keep only legendary creatures
                    similar_commanders = [card for card in similar_commanders if 'Legendary' in card.get('type_line', '') and 'Creature' in card.get('type_line', '')]
                    
                    return {
                        'success': True,
                        'query_type': f"Generate {theme} deck",
                        'cards': similar_commanders,
                        'explanation': f"Couldn't find commander '{commander}'. Here are some similar legendary creatures you might be interested in."
                    }
            
            # Build the strategy description
            strategy = f"{theme}"
            if budget:
                strategy += f" under ${budget}"
            
            # Generate deck recommendation
            deck_recommendation = self.rag.generate_deck_recommendation(
                strategy=strategy,
                commander=commander,
                budget=budget,
                colors=colors
            )
            
            # Format the results
            if 'cards' in deck_recommendation:
                # Add explanation based on the deck recommendation
                explanation = f"Generated a {theme} deck"
                if commander:
                    explanation += f" with {commander} as the commander"
                if budget:
                    explanation += f" under a budget of ${budget}"
                explanation += "."
                
                return {
                    'success': True,
                    'query_type': f"Generate {theme} deck",
                    'cards': deck_recommendation['cards'],
                    'explanation': explanation
                }
            else:
                return {
                    'success': False,
                    'error': "Couldn't generate deck"
                }
                
        except Exception as e:
            logger.error(f"Error handling deck generation query: {e}")
            return {
                'success': False,
                'error': f"Error generating deck: {str(e)}"
            }
    
    def _extract_card_info_params(self, match, full_query):
        """Extract parameters for card info queries"""
        card_name = match.group(1).strip()
        return {'card_name': card_name}
    
    def _handle_card_info(self, params, original_query):
        """Handle queries for card information"""
        card_name = params.get('card_name', '')
        
        try:
            # Get card information
            card_data = None
            if self.rag.db.is_connected:
                response = self.rag.db.client.table("mtg_cards").select(
                    "*"
                ).eq("name", card_name).limit(1).execute()
                
                if response.data:
                    card_data = response.data[0]
            
            if not card_data:
                # Try to find similar cards
                similar_cards = self.rag.retrieve_cards_by_text(card_name, top_k=5)
                
                return {
                    'success': True,
                    'query_type': f"Card information search",
                    'cards': similar_cards,
                    'explanation': f"Couldn't find exact card: '{card_name}'. Here are some similar cards."
                }
            
            # Format the result
            result_cards = [card_data]
            
            return {
                'success': True,
                'query_type': f"Card information for {card_name}",
                'cards': result_cards,
                'explanation': f"Here's information about {card_name}."
            }
            
        except Exception as e:
            logger.error(f"Error handling card info query: {e}")
            return {
                'success': False,
                'error': f"Error retrieving card information: {str(e)}"
            }
    
    def _handle_general_search(self, params, original_query):
        """Handle general search queries that don't match specific patterns"""
        query = params.get('query', '')
        colors = params.get('colors', [])
        card_type = params.get('card_type')
        max_price = params.get('max_price')
        
        try:
            # Try to find cards matching the query
            matching_cards = self.rag.retrieve_cards_by_text(query, top_k=15)
            
            # Filter the results based on provided filters
            filtered_cards = []
            for card in matching_cards:
                if self._passes_filters(card, colors, card_type, max_price):
                    filtered_cards.append(card)
            
            return {
                'success': True,
                'query_type': "General card search",
                'cards': filtered_cards,
                'explanation': f"Found {len(filtered_cards)} cards matching your query."
            }
            
        except Exception as e:
            logger.error(f"Error handling general search query: {e}")
            return {
                'success': False,
                'error': f"Error searching cards: {str(e)}"
            }
    
    def _passes_filters(self, card, colors=None, card_type=None, max_price=None):
        """Check if a card passes the specified filters - FIXED FOR COLORLESS CARDS"""
        # Apply color filter - COLORLESS CARDS FIX
        if colors:
            card_colors = card.get('color_identity', [])
            # Colorless cards (empty color_identity) can be played in any deck
            if card_colors:  # Only filter if card actually has colors
                if not any(color in card_colors for color in colors):
                    return False
        
        # Apply card type filter
        if card_type and card_type.lower() not in card.get('type_line', '').lower():
            return False
        
        # Apply price filter
        if max_price is not None:
            try:
                price = float(card.get('prices_usd', 0) or 0)
                if price > max_price:
                    return False
            except (ValueError, TypeError):
                pass
        
        return True