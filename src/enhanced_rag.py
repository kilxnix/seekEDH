# enhanced_rag_search.py
import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import faiss

logger = logging.getLogger("EnhancedRAG")

class EnhancedMTGRAGSystem:
    """Enhanced RAG system for MTG cards with synergy detection and advanced search"""
    
    def __init__(self, base_rag_system):
        """Initialize enhanced system with existing RAG system"""
        self.base_rag = base_rag_system
        self.db = base_rag_system.db
        
        # Synergy detection mappings
        self.mechanics_keywords = {
            'artifact': ['artifact', 'equipment', 'vehicle'],
            'creature': ['creature', 'tribal'],
            'graveyard': ['graveyard', 'exile', 'flashback', 'unearth', 'delve'],
            'counters': ['+1/+1 counter', '-1/-1 counter', 'proliferate', 'counter'],
            'tokens': ['token', 'create', 'populate'],
            'lifegain': ['gain life', 'lifegain', 'life you gain'],
            'sacrifice': ['sacrifice', 'dies', 'death trigger'],
            'draw': ['draw', 'card draw', 'draw cards'],
            'ramp': ['mana', 'ramp', 'land', 'search your library for a land'],
            'control': ['counter', 'removal', 'destroy', 'exile'],
            'aggro': ['haste', 'trample', 'first strike', 'double strike'],
            'flying': ['flying', 'reach'],
            'burn': ['damage', 'burn', 'lightning', 'shock'],
            'mill': ['mill', 'library', 'put cards from library into graveyard'],
            'storm': ['storm', 'instant', 'sorcery'],
            'tribal': ['elf', 'goblin', 'human', 'dragon', 'angel', 'demon', 'zombie']
        }
        
        # Card interaction patterns
        self.interaction_patterns = [
            # ETB (Enter the Battlefield) synergies
            {
                'pattern': r'when .* enters the battlefield',
                'synergy_type': 'etb',
                'keywords': ['flicker', 'blink', 'bounce', 'clone']
            },
            # Activated abilities
            {
                'pattern': r'(\{[^}]+\}): ',
                'synergy_type': 'activated_ability',
                'keywords': ['untap', 'mana', 'tap']
            },
            # Sacrifice synergies
            {
                'pattern': r'sacrifice.*:',
                'synergy_type': 'sacrifice',
                'keywords': ['token', 'death', 'grave']
            },
            # Cost reduction
            {
                'pattern': r'cost.*less|costs? \{[^}]*\} less',
                'synergy_type': 'cost_reduction',
                'keywords': ['artifact', 'instant', 'sorcery', 'creature']
            }
        ]
    
    def enhanced_card_search(self, 
                           query: str, 
                           filters: Dict[str, Any] = None,
                           top_k: int = 20) -> Dict[str, Any]:
        """
        Enhanced card search with multiple search strategies and filtering
        
        Args:
            query: Search query (text, card name, or strategy)
            filters: Dictionary of filters (colors, types, price, etc.)
            top_k: Number of results to return
            
        Returns:
            Dictionary with search results and metadata
        """
        filters = filters or {}
        
        try:
            # Multi-strategy search
            results = []
            
            # 1. Exact name match
            exact_match = self._exact_name_search(query)
            if exact_match:
                results.append({
                    'card': exact_match,
                    'match_type': 'exact_name',
                    'relevance_score': 1.0
                })
            
            # 2. Semantic similarity search
            semantic_results = self._semantic_search(query, top_k)
            for result in semantic_results:
                results.append({
                    'card': result,
                    'match_type': 'semantic',
                    'relevance_score': result.get('similarity', 0.5)
                })
            
            # 3. Keyword-based search
            keyword_results = self._keyword_search(query, top_k)
            for result in keyword_results:
                results.append({
                    'card': result,
                    'match_type': 'keyword',
                    'relevance_score': 0.6  # Base score for keyword matches
                })
            
            # 4. Mechanic-based search
            mechanic_results = self._mechanic_search(query, top_k)
            for result in mechanic_results:
                results.append({
                    'card': result,
                    'match_type': 'mechanic',
                    'relevance_score': 0.7
                })
            
            # Remove duplicates and apply filters
            unique_results = self._deduplicate_and_filter(results, filters)
            
            # Sort by relevance score
            unique_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Limit results
            final_results = unique_results[:top_k]
            
            return {
                'success': True,
                'query': query,
                'total_found': len(unique_results),
                'returned': len(final_results),
                'cards': [r['card'] for r in final_results],
                'search_metadata': {
                    'match_types': list(set(r['match_type'] for r in final_results)),
                    'avg_relevance': np.mean([r['relevance_score'] for r in final_results]) if final_results else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced card search: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
    
    def find_card_synergies(self, 
                          seed_cards: List[str], 
                          top_k: int = 15) -> Dict[str, Any]:
        """
        Find cards that synergize well with the given seed cards
        
        Args:
            seed_cards: List of card names to find synergies for
            top_k: Number of synergistic cards to return
            
        Returns:
            Dictionary with synergistic cards and explanations
        """
        try:
            # Get seed card data
            seed_card_data = []
            for card_name in seed_cards:
                card_data = self._get_card_data(card_name)
                if card_data:
                    seed_card_data.append(card_data)
            
            if not seed_card_data:
                return {
                    'success': False,
                    'error': 'No valid seed cards found',
                    'seed_cards': seed_cards
                }
            
            # Extract mechanics and themes from seed cards
            seed_mechanics = self._extract_card_mechanics(seed_card_data)
            seed_themes = self._extract_card_themes(seed_card_data)
            
            # Find synergistic cards
            synergy_candidates = []
            
            # 1. Mechanic-based synergies
            mechanic_synergies = self._find_mechanic_synergies(seed_mechanics, top_k * 2)
            synergy_candidates.extend(mechanic_synergies)
            
            # 2. Theme-based synergies
            theme_synergies = self._find_theme_synergies(seed_themes, top_k * 2)
            synergy_candidates.extend(theme_synergies)
            
            # 3. Color identity synergies
            color_synergies = self._find_color_synergies(seed_card_data, top_k)
            synergy_candidates.extend(color_synergies)
            
            # Calculate synergy scores
            scored_synergies = self._calculate_synergy_scores(
                seed_card_data, synergy_candidates
            )
            
            # Remove seed cards from results
            seed_names = {card['name'].lower() for card in seed_card_data}
            filtered_synergies = [
                s for s in scored_synergies 
                if s['card']['name'].lower() not in seed_names
            ]
            
            # Sort by synergy score and limit results
            filtered_synergies.sort(key=lambda x: x['synergy_score'], reverse=True)
            final_synergies = filtered_synergies[:top_k]
            
            return {
                'success': True,
                'seed_cards': seed_cards,
                'seed_mechanics': seed_mechanics,
                'seed_themes': seed_themes,
                'synergistic_cards': final_synergies,
                'total_candidates_analyzed': len(synergy_candidates)
            }
            
        except Exception as e:
            logger.error(f"Error finding card synergies: {e}")
            return {
                'success': False,
                'error': str(e),
                'seed_cards': seed_cards
            }
    
    def analyze_card_mechanics(self, card_name: str) -> Dict[str, Any]:
        """
        Analyze a card's mechanics and find related mechanics/cards
        
        Args:
            card_name: Name of the card to analyze
            
        Returns:
            Analysis of the card's mechanics and related cards
        """
        try:
            card_data = self._get_card_data(card_name)
            if not card_data:
                return {
                    'success': False,
                    'error': f'Card "{card_name}" not found'
                }
            
            # Extract mechanics
            mechanics = self._extract_card_mechanics([card_data])
            themes = self._extract_card_themes([card_data])
            interactions = self._find_card_interactions(card_data)
            
            # Find related cards for each mechanic
            related_cards = {}
            for mechanic in mechanics:
                related = self._find_mechanic_synergies([mechanic], 5)
                related_cards[mechanic] = [r['card'] for r in related]
            
            return {
                'success': True,
                'card': card_data,
                'mechanics': mechanics,
                'themes': themes,
                'interactions': interactions,
                'related_cards_by_mechanic': related_cards,
                'synergy_potential': len(mechanics) + len(themes)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing card mechanics: {e}")
            return {
                'success': False,
                'error': str(e),
                'card_name': card_name
            }
    
    def _exact_name_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Search for exact card name match"""
        if not self.db.is_connected:
            return None
        
        try:
            response = self.db.client.table("mtg_cards").select(
                "*"
            ).eq("name", query).limit(1).execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error in exact name search: {e}")
            return None
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Use the base RAG system for semantic search"""
        try:
            return self.base_rag.retrieve_cards_by_text(query, top_k)
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for cards containing keywords"""
        if not self.db.is_connected:
            return []
        
        try:
            # Split query into keywords
            keywords = [word.strip().lower() for word in query.split() if len(word) > 2]
            
            results = []
            for keyword in keywords:
                response = self.db.client.table("mtg_cards").select(
                    "*"
                ).ilike("oracle_text", f"%{keyword}%").limit(top_k // len(keywords) + 5).execute()
                
                if response.data:
                    results.extend(response.data)
            
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _mechanic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for cards based on game mechanics"""
        results = []
        query_lower = query.lower()
        
        # Check if query matches any known mechanics
        for mechanic, keywords in self.mechanics_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Find cards with this mechanic
                mechanic_cards = self._find_cards_with_mechanic(mechanic, top_k // 2)
                results.extend(mechanic_cards)
        
        return results[:top_k]
    
    def _find_cards_with_mechanic(self, mechanic: str, limit: int) -> List[Dict[str, Any]]:
        """Find cards that have a specific mechanic"""
        if not self.db.is_connected:
            return []
        
        try:
            keywords = self.mechanics_keywords.get(mechanic, [mechanic])
            results = []
            
            for keyword in keywords:
                response = self.db.client.table("mtg_cards").select(
                    "*"
                ).ilike("oracle_text", f"%{keyword}%").limit(limit // len(keywords) + 2).execute()
                
                if response.data:
                    results.extend(response.data)
            
            return results[:limit]
        except Exception as e:
            logger.error(f"Error finding cards with mechanic {mechanic}: {e}")
            return []
    
    def _deduplicate_and_filter(self, 
                               results: List[Dict[str, Any]], 
                               filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Remove duplicates and apply filters"""
        seen_cards = set()
        unique_results = []
        
        for result in results:
            card = result['card']
            card_id = card.get('id') or card.get('name')
            
            if card_id in seen_cards:
                continue
            
            # Apply filters
            if self._passes_filters(card, filters):
                seen_cards.add(card_id)
                unique_results.append(result)
        
        return unique_results
    
    def _passes_filters(self, card: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if card passes the given filters"""
        # Color filter - UPDATED TO HANDLE COLORLESS
        if 'colors' in filters:
            card_colors = card.get('color_identity', [])
            if isinstance(card_colors, str):
                card_colors = card_colors.split(',') if card_colors else []
            
            required_colors = filters['colors']
            
            # Special handling for colorless filter
            if 'C' in required_colors:
                # If colorless is requested, only show colorless cards (empty color identity)
                if card_colors:  # If card has any colors, exclude it
                    return False
            elif required_colors:
                # For colored cards, check if any required color matches
                # Colorless cards (empty color_identity) can be played in any deck
                if card_colors and not any(color in card_colors for color in required_colors):
                    return False
        
        # Type filter
        if 'type' in filters:
            card_type = card.get('type_line', '')
            if filters['type'].lower() not in card_type.lower():
                return False
        
        # Price filter
        if 'max_price' in filters:
            try:
                price = float(card.get('prices_usd', 0) or 0)
                if price > filters['max_price']:
                    return False
            except (ValueError, TypeError):
                pass
        
        # Format legality filter
        if 'format' in filters:
            legalities = card.get('legalities', {})
            if isinstance(legalities, str):
                try:
                    legalities = json.loads(legalities)
                except:
                    legalities = {}
            
            format_legal = legalities.get(filters['format'], 'not_legal')
            if format_legal not in ['legal', 'restricted']:
                return False
        
        return True
    
    def _get_card_data(self, card_name: str) -> Optional[Dict[str, Any]]:
        """Get full card data by name"""
        if not self.db.is_connected:
            return None
        
        try:
            response = self.db.client.table("mtg_cards").select(
                "*"
            ).eq("name", card_name).limit(1).execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting card data for {card_name}: {e}")
            return None
    
    def _extract_card_mechanics(self, cards: List[Dict[str, Any]]) -> List[str]:
        """Extract game mechanics from a list of cards"""
        mechanics = set()
        
        for card in cards:
            oracle_text = card.get('oracle_text', '').lower()
            type_line = card.get('type_line', '').lower()
            
            # Check for known mechanics
            for mechanic, keywords in self.mechanics_keywords.items():
                if any(keyword in oracle_text or keyword in type_line for keyword in keywords):
                    mechanics.add(mechanic)
            
            # Check for specific ability words and keywords
            ability_patterns = [
                r'\b(flying|trample|haste|vigilance|deathtouch|lifelink)\b',
                r'\b(flash|hexproof|indestructible|menace|reach)\b',
                r'\b(first strike|double strike|protection)\b'
            ]
            
            for pattern in ability_patterns:
                matches = re.findall(pattern, oracle_text)
                mechanics.update(matches)
        
        return list(mechanics)
    
    def _extract_card_themes(self, cards: List[Dict[str, Any]]) -> List[str]:
        """Extract thematic elements from cards"""
        themes = set()
        
        for card in cards:
            oracle_text = card.get('oracle_text', '').lower()
            type_line = card.get('type_line', '').lower()
            
            # Tribal themes
            creature_types = ['elf', 'goblin', 'human', 'dragon', 'angel', 'demon', 'zombie', 'vampire']
            for creature_type in creature_types:
                if creature_type in oracle_text or creature_type in type_line:
                    themes.add(f'{creature_type}_tribal')
            
            # Strategy themes
            if any(word in oracle_text for word in ['token', 'create']):
                themes.add('tokens')
            if any(word in oracle_text for word in ['counter', '+1/+1']):
                themes.add('counters')
            if any(word in oracle_text for word in ['graveyard', 'grave']):
                themes.add('graveyard')
        
        return list(themes)
    
    def _find_mechanic_synergies(self, mechanics: List[str], limit: int) -> List[Dict[str, Any]]:
        """Find cards that synergize with given mechanics"""
        synergy_cards = []
        
        for mechanic in mechanics:
            cards = self._find_cards_with_mechanic(mechanic, limit // len(mechanics) + 2)
            for card in cards:
                synergy_cards.append({
                    'card': card,
                    'synergy_type': 'mechanic',
                    'synergy_reason': f'Shares {mechanic} mechanic'
                })
        
        return synergy_cards
    
    def _find_theme_synergies(self, themes: List[str], limit: int) -> List[Dict[str, Any]]:
        """Find cards that synergize with given themes"""
        synergy_cards = []
        
        for theme in themes:
            # Convert theme back to searchable terms
            if '_tribal' in theme:
                creature_type = theme.replace('_tribal', '')
                cards = self._find_cards_with_creature_type(creature_type, limit // len(themes) + 2)
                for card in cards:
                    synergy_cards.append({
                        'card': card,
                        'synergy_type': 'tribal',
                        'synergy_reason': f'Supports {creature_type} tribal strategy'
                    })
        
        return synergy_cards
    
    def _find_cards_with_creature_type(self, creature_type: str, limit: int) -> List[Dict[str, Any]]:
        """Find cards that support a specific creature type"""
        if not self.db.is_connected:
            return []
        
        try:
            response = self.db.client.table("mtg_cards").select(
                "*"
            ).or_(
                f'type_line.ilike.%{creature_type}%,oracle_text.ilike.%{creature_type}%'
            ).limit(limit).execute()
            
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Error finding cards with creature type {creature_type}: {e}")
            return []
    
    def _find_color_synergies(self, seed_cards: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Find cards that fit the color identity of seed cards"""
        # Determine color identity from seed cards
        all_colors = set()
        has_colorless = False
        
        for card in seed_cards:
            colors = card.get('color_identity', [])
            if isinstance(colors, str):
                colors = colors.split(',') if colors else []
            
            if not colors:  # Empty array means colorless
                has_colorless = True
            else:
                all_colors.update(colors)
        
        if not self.db.is_connected:
            return []
        
        try:
            synergy_cards = []
            
            # If we have colorless cards, find more colorless cards
            if has_colorless:
                # Find colorless cards (empty color_identity)
                response = self.db.client.table("mtg_cards").select(
                    "*"
                ).or_('color_identity.is.null,color_identity.eq.{}').limit(limit // 2).execute()
                
                for card in response.data if response.data else []:
                    synergy_cards.append({
                        'card': card,
                        'synergy_type': 'colorless',
                        'synergy_reason': 'Colorless card - fits in any deck'
                    })
            
            # Find cards that fit within the color identity
            if all_colors:
                color_filter = list(all_colors)
                response = self.db.client.table("mtg_cards").select(
                    "*"
                ).contains('color_identity', color_filter).limit(limit // 2).execute()
                
                for card in response.data if response.data else []:
                    synergy_cards.append({
                        'card': card,
                        'synergy_type': 'color_identity',
                        'synergy_reason': f'Fits {",".join(sorted(all_colors))} color identity'
                    })
            
            return synergy_cards[:limit]
            
        except Exception as e:
            logger.error(f"Error finding color synergies: {e}")
            return []
    
    def _calculate_synergy_scores(self, 
                                seed_cards: List[Dict[str, Any]], 
                                candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate synergy scores for candidate cards"""
        scored_cards = []
        
        for candidate in candidates:
            card = candidate['card']
            base_score = 0.5
            
            # Adjust score based on synergy type
            synergy_type = candidate.get('synergy_type', 'unknown')
            if synergy_type == 'mechanic':
                base_score += 0.3
            elif synergy_type == 'tribal':
                base_score += 0.25
            elif synergy_type == 'color_identity':
                base_score += 0.1
            
            # Bonus for multiple synergies
            card_text = card.get('oracle_text', '').lower()
            synergy_count = 0
            
            for seed_card in seed_cards:
                seed_text = seed_card.get('oracle_text', '').lower()
                # Simple keyword overlap check
                common_keywords = self._find_common_keywords(card_text, seed_text)
                synergy_count += len(common_keywords)
            
            # Apply synergy bonus
            if synergy_count > 0:
                base_score += min(0.3, synergy_count * 0.1)
            
            scored_cards.append({
                'card': card,
                'synergy_score': min(1.0, base_score),
                'synergy_type': synergy_type,
                'synergy_reason': candidate.get('synergy_reason', 'Unknown synergy'),
                'synergy_count': synergy_count
            })
        
        return scored_cards
    
    def _find_common_keywords(self, text1: str, text2: str) -> List[str]:
        """Find common important keywords between two card texts"""
        important_keywords = [
            'flying', 'trample', 'haste', 'vigilance', 'deathtouch', 'lifelink',
            'token', 'counter', 'graveyard', 'exile', 'sacrifice', 'draw',
            'mana', 'artifact', 'enchantment', 'creature', 'instant', 'sorcery'
        ]
        
        common = []
        for keyword in important_keywords:
            if keyword in text1 and keyword in text2:
                common.append(keyword)
        
        return common
    
    def _find_card_interactions(self, card: Dict[str, Any]) -> List[Dict[str, str]]:
        """Find potential interactions for a card based on its text"""
        interactions = []
        oracle_text = card.get('oracle_text', '')
        
        for pattern_info in self.interaction_patterns:
            pattern = pattern_info['pattern']
            matches = re.findall(pattern, oracle_text, re.IGNORECASE)
            
            if matches:
                interactions.append({
                    'type': pattern_info['synergy_type'],
                    'pattern': pattern,
                    'keywords': pattern_info['keywords'],
                    'matches': matches
                })
        
        return interactions