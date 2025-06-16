# enhanced_universal_search.py - COMPLETE FIXED VERSION
import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    import requests  # used for Scryfall fallback
except Exception:  # pragma: no cover - requests may be missing
    requests = None

logger = logging.getLogger("EnhancedUniversalSearch")

class EnhancedUniversalSearchHandler:
    """FIXED: Enhanced handler for universal natural language search queries"""

    def __init__(self, rag_system):
        self.rag = rag_system

        # Initialize card name set for validation
        self.card_name_set = set()
        self._load_card_names()

        # Initialize model if available
        self.model = None
        if hasattr(self.rag, 'embedding_model') and self.rag.embedding_model:
            self.model = self.rag.embedding_model
            logger.info("Model available for card name extraction")

        # IMPROVED: Better card name detection patterns
        self.card_detection_patterns = [
            # Quoted names (highest priority)
            r'"([^"]+)"',
            r"'([^']+)'",

            # FIXED: Direct mentions with proper word boundaries
            r'(?:with|using|like|including)\s+([A-Z][A-Za-z0-9\s,\'/\-]{2,40})(?=\s*(?:deck|commander|and|,|\.|$))',

            # FIXED: Better synergy patterns
            r'(?:synergies?\s+(?:for|with))\s+([A-Z][A-Za-z0-9\s,\'/\-]{2,40})(?=\s*(?:deck|and|,|\.|$))',

            # FIXED: Deck/commander mentions
            r'(?:my|the)\s+([A-Z][A-Za-z0-9\s,\'/\-]{3,40})\s+(?:deck|commander)',

            # FIXED: Standalone title case phrases (2-4 words) - but more restrictive
            r'\b([A-Z][A-Za-z0-9/\-]+(?:\s+[A-Z][A-Za-z0-9/\-]+){1,3})\b(?=\s*(?:deck|commander|,|\.|$|and))',

            # FIXED: Card names at end of sentence
            r'(?:with|using)\s+([A-Z][A-Za-z0-9\s,\'/\-]{3,40})$'
        ]

        # IMPROVED: Known MTG card name patterns for validation
        self.mtg_name_indicators = [
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*$',  # Title case words
            r'^[A-Z][a-z]+,\s+[A-Z][a-z\s]+$',  # "Jace, the Mind Sculptor" pattern
            r'^[A-Z][a-z]+\s+of\s+[A-Z][a-z]+$',  # "Lord of Atlantis" pattern
        ]

        # NEW: Category detection patterns
        self.category_patterns = {
            'colors': {
                'white': ['W'],
                'blue': ['U'],
                'black': ['B'],
                'red': ['R'],
                'green': ['G'],
                'colorless': ['C'],
                'mono': 'mono',
                'multicolor': 'multi'
            },
            'types': {
                'creature': 'Creature',
                'creatures': 'Creature',
                'artifact': 'Artifact',
                'artifacts': 'Artifact',
                'enchantment': 'Enchantment',
                'enchantments': 'Enchantment',
                'instant': 'Instant',
                'instants': 'Instant',
                'sorcery': 'Sorcery',
                'sorceries': 'Sorcery',
                'planeswalker': 'Planeswalker',
                'planeswalkers': 'Planeswalker',
                'land': 'Land',
                'lands': 'Land'
            },
            'strategies': {
                'token': ['token', 'tokens'],
                'ramp': ['ramp', 'mana acceleration'],
                'removal': ['removal', 'destroy', 'exile'],
                'draw': ['draw', 'card draw'],
                'counter': ['counter', 'counterspell'],
                'tribal': ['tribal'],
                'combo': ['combo', 'infinite'],
                'aggro': ['aggro', 'aggressive'],
                'control': ['control'],
                'midrange': ['midrange']
            },
            'budget': {
                'cheap': 5.0,
                'budget': 10.0,
                'affordable': 15.0,
                'expensive': 50.0
            }
        }

        # NEW: Question words that indicate category searches
        self.category_indicators = [
            'what', 'which', 'show', 'find', 'get', 'need', 'want',
            'list', 'suggest', 'recommend', 'good', 'best'
        ]

    def _load_card_names(self):
        """Load all card names from database into a set for fast lookup"""
        try:
            if self.rag.db.is_connected:
                response = self.rag.db.client.table("mtg_cards").select("name").execute()
                if response.data:
                    self.card_name_set = {card['name'].lower() for card in response.data}
                    logger.info(f"Loaded {len(self.card_name_set)} card names for validation")
                else:
                    logger.warning("No card names loaded from database")
            else:
                logger.warning("Database not connected, card name validation disabled")
        except Exception as e:
            logger.error(f"Error loading card names: {e}")
            self.card_name_set = set()

    def detect_query_type(self, query_text: str) -> Dict[str, Any]:
        """NEW: Detect if this is a category search or card-specific search"""
        query_lower = query_text.lower()

        # Check for category indicators
        has_category_words = any(word in query_lower for word in self.category_indicators)

        # Extract potential constraints
        constraints = self.extract_constraints(query_text)

        # Check for potential card names
        potential_cards = self.extract_card_names(query_text, quick_check=True)

        # Determine query type
        if potential_cards and len(potential_cards) > 0:
            # This looks like a card-specific query
            return {
                'type': 'card_specific',
                'cards': potential_cards,
                'constraints': constraints
            }
        elif has_category_words or any(constraints.values()):
            # This looks like a category search
            return {
                'type': 'category_search',
                'constraints': constraints
            }
        else:
            # Fallback to general search
            return {
                'type': 'general_search',
                'constraints': constraints
            }

    def extract_constraints(self, query_text: str) -> Dict[str, Any]:
        """NEW: Extract color, type, strategy, and budget constraints from query"""
        query_lower = query_text.lower()
        constraints = {
            'colors': [],
            'card_types': [],
            'strategies': [],
            'budget': None,
            'deck_context': None
        }

        # Extract colors from full words
        for color_word, color_codes in self.category_patterns['colors'].items():
            if color_word in query_lower:
                if isinstance(color_codes, list):
                    constraints['colors'].extend(color_codes)
                elif color_codes == 'mono':
                    constraints['mono'] = True
                elif color_codes == 'multi':
                    constraints['multicolor'] = True

        # NEW: Extract color abbreviations like "UB" or "WUG"
        abbr_matches = re.findall(r'\b[WUBRG]{1,5}\b', query_text.upper())
        for abbr in abbr_matches:
            for letter in abbr:
                if letter not in constraints['colors']:
                    constraints['colors'].append(letter)

        # Extract card types
        for type_word, type_name in self.category_patterns['types'].items():
            if type_word in query_lower:
                if type_name not in constraints['card_types']:
                    constraints['card_types'].append(type_name)

        # Extract strategies
        for strategy, keywords in self.category_patterns['strategies'].items():
            if any(keyword in query_lower for keyword in keywords):
                constraints['strategies'].append(strategy)

        # Extract budget constraints
        budget_match = re.search(r'(?:under|less\s+than|below|budget\s+of|max|maximum|cheap.*)\s*\$?(\d+(?:\.\d+)?)', query_lower)
        if budget_match:
            constraints['budget'] = float(budget_match.group(1))
        else:
            # Check for budget keywords
            for budget_word, default_amount in self.category_patterns['budget'].items():
                if budget_word in query_lower:
                    constraints['budget'] = default_amount
                    break

        # Extract deck context
        deck_match = re.search(r'(?:for|with|in)\s+(?:my\s+)?([a-z]+)\s+deck', query_lower)
        if deck_match:
            constraints['deck_context'] = deck_match.group(1)

        return constraints

    def extract_card_names(self, query_text: str, top_k=5, quick_check=False) -> list:
        """FIXED: Extract card names using improved regex and model validation"""
        potential_cards = set()
        logger.info(f"Extracting card names from: {query_text}")

        # QUICK CHECK: Skip detailed extraction for category queries
        if quick_check:
            # Only do basic regex extraction for speed
            # <<<<<<<<<<<<<<<<<<<< APPLIED CHANGE HERE >>>>>>>>>>>>>>>>>>>>
            for pattern in self.card_detection_patterns[:4]:  # Now checks top 4 patterns
                matches = re.finditer(pattern, query_text, re.IGNORECASE)
                for match in matches:
                    for group in match.groups():
                        if group and len(group.strip()) > 2:
                            cleaned = self._clean_card_name(group.strip())
                            if cleaned and self._looks_like_card_name(cleaned):
                                if self._validate_card_exists(cleaned):
                                    potential_cards.add(cleaned)
                                    if len(potential_cards) >= 2:  # Early exit for quick check
                                        return list(potential_cards)
            return list(potential_cards)

        # FULL EXTRACTION
        logger.info("Starting improved regex-based card extraction...")

        for pattern in self.card_detection_patterns:
            matches = re.finditer(pattern, query_text, re.IGNORECASE)
            for match in matches:
                for group in match.groups():
                    if group and len(group.strip()) > 2:
                        cleaned = self._clean_card_name(group.strip())
                        if cleaned and self._looks_like_card_name(cleaned):
                            potential_cards.add(cleaned)
                            logger.info(f"Regex extracted: '{cleaned}'")

        # Use model for VALIDATION instead of extraction
        logger.info("Starting model-based validation...")
        if self.model and len(potential_cards) > 0:
            validated_cards = self._validate_cards_with_model(list(potential_cards))
            potential_cards.update(validated_cards)

        # FIXED: Better RAG-based extraction focusing on exact matches
        logger.info("Starting targeted RAG-based extraction...")
        try:
            # Use key terms from the query to find cards
            key_terms = self._extract_key_terms(query_text)
            for term in key_terms:
                if len(term) > 3:  # Only meaningful terms
                    rag_results = self.rag.search_cards_by_keyword(term, top_k=3)
                    for card in rag_results:
                        card_name = card.get('name', '')
                        if card_name:
                            potential_cards.add(card_name)
                            logger.info(f"RAG extracted via keyword '{term}': '{card_name}'")
        except Exception as e:
            logger.error(f"RAG extraction failed: {e}")

        # Validation against database with better error handling
        logger.info("Validating extracted candidates...")
        valid_cards = []
        for candidate in potential_cards:
            try:
                if self._validate_card_exists(candidate):
                    valid_cards.append(candidate)
                    logger.info(f"Card validated: '{candidate}'")
                else:
                    # Try fuzzy matching for common misspellings
                    fuzzy_match = self._find_fuzzy_match(candidate)
                    if fuzzy_match:
                        valid_cards.append(fuzzy_match)
                        logger.info(f"Fuzzy matched: '{candidate}' -> '{fuzzy_match}'")
                    else:
                        logger.info(f"Candidate not found: '{candidate}'")
            except Exception as e:
                logger.warning(f"Error validating candidate '{candidate}': {e}")

        logger.info(f"Final extracted and validated card names: {valid_cards}")
        return valid_cards

    def _extract_key_terms(self, query_text: str) -> List[str]:
        """Extract key terms that might be part of card names"""
        # Remove common words and extract potential card name components
        stop_words = {'deck', 'synergy', 'synergies', 'with', 'for', 'my', 'the', 'and', 'or', 'need', 'cards', 'token', 'what', 'which', 'work', 'well', 'good', 'best', 'find', 'show', 'artifacts', 'creatures'}

        words = re.findall(r'\b[A-Za-z]+\b', query_text)
        key_terms = []

        for word in words:
            if word.lower() not in stop_words and len(word) > 2:
                key_terms.append(word)

        # Also look for multi-word terms that could be card names
        title_case_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]*)+\b', query_text)
        key_terms.extend(title_case_phrases)

        return key_terms

    def _looks_like_card_name(self, text: str) -> bool:
        """Check if text looks like it could be a Magic card name"""
        if not text or len(text) < 3 or len(text) > 50:
            return False

        # FIXED: Don't accept common category words as card names
        category_words = {'what', 'which', 'colorless', 'artifacts', 'creatures', 'work', 'well', 'good', 'best', 'find', 'show'}
        if text.lower() in category_words:
            return False

        # Don't accept phrases that start with question words
        if text.lower().startswith(('what ', 'which ', 'how ', 'when ', 'where ')):
            return False

        # Must have at least one capital letter
        if not any(c.isupper() for c in text):
            return False

        # Check against known patterns
        for pattern in self.mtg_name_indicators:
            if re.match(pattern, text):
                return True

        # Allow some flexibility for card names with special characters
        if re.match(r'^[A-Z][A-Za-z\s,\'\-]{2,49}$', text):
            return True

        return False

    def _validate_cards_with_model(self, candidates: List[str]) -> List[str]:
        """FIXED: Use model to validate which candidates are actual Magic cards"""
        validated = []

        try:
            if not self.model:
                return candidates

            # For now, let's use a simpler approach - check each candidate individually
            for candidate in candidates:
                # Simple validation: check if it's in our known card set
                if candidate.lower() in self.card_name_set:
                    validated.append(candidate)
                    logger.info(f"Model validated: '{candidate}'")

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return candidates  # Return original candidates if model fails

        return validated

    def _clean_card_name(self, raw_name: str) -> Optional[str]:
        """FIXED: Clean and validate a potential card name"""
        if not raw_name:
            return None

        # Remove unwanted text patterns
        cleaned = raw_name.strip()

        # Remove trailing words that are not part of card names
        stop_suffixes = ['deck', 'commander', 'strategy', 'synergy', 'synergies', 'work', 'well']
        for suffix in stop_suffixes:
            if cleaned.lower().endswith(' ' + suffix):
                cleaned = cleaned[:-len(suffix)-1].strip()

        # Remove leading articles
        if cleaned.lower().startswith('the '):
            alt_cleaned = cleaned[4:]
            # Only remove 'the' if the remaining text still looks like a card name
            if self._looks_like_card_name(alt_cleaned):
                cleaned = alt_cleaned

        # Final validation
        if len(cleaned) < 3 or len(cleaned) > 50:
            return None

        return cleaned

    def _validate_card_exists(self, card_name: str) -> bool:
        """FIXED: Check if card exists in database with better error handling"""
        try:
            if not self.rag.db.is_connected:
                return False

            # Check against our loaded card name set first (faster)
            if card_name.lower() in self.card_name_set:
                return True

            # Fallback to database query with error handling
            response = self.rag.db.client.table("mtg_cards").select("id").eq("name", card_name).limit(1).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error validating card {card_name}: {e}")
            return False

    def _find_fuzzy_match(self, card_name: str) -> Optional[str]:
        """FIXED: Find fuzzy matches for potential misspellings"""
        try:
            if not self.rag.db.is_connected:
                return None

            # Try partial matches with error handling
            response = self.rag.db.client.table("mtg_cards").select("name").ilike("name", f"%{card_name}%").limit(3).execute()
            if response.data:
                # Return the closest match (first one for now)
                return response.data[0]['name']
            return None
        except Exception as e:
            logger.error(f"Error finding fuzzy match for {card_name}: {e}")
            return None

    def _fetch_color_identity_online(self, card_name: str) -> List[str]:
        """Fetch a card's color identity from Scryfall with local caching"""
        cache_file = os.path.join("data", "color_identity_cache.json")

        if not hasattr(self, "_color_cache"):
            self._color_cache = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        self._color_cache = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load color cache: {e}")

        if card_name in self._color_cache:
            return self._color_cache[card_name]

        try:
            import requests

            encoded = requests.utils.quote(card_name)
            resp = requests.get(
                f"https://api.scryfall.com/cards/named?exact={encoded}", timeout=10
            )
            if resp.status_code != 200:
                resp = requests.get(
                    f"https://api.scryfall.com/cards/named?fuzzy={encoded}", timeout=10
                )

            if resp.status_code == 200:
                data = resp.json()
                colors = data.get("color_identity", [])
                self._color_cache[card_name] = colors
                try:
                    with open(cache_file, "w") as f:
                        json.dump(self._color_cache, f)
                except Exception as e:
                    logger.warning(f"Failed to save color cache: {e}")
                return colors
        except Exception as e:
            logger.warning(f"Error fetching color identity for {card_name}: {e}")

        return []

    def _search_scryfall(self, query: str, limit: int = 30) -> List[Dict]:
        """Search Scryfall API as a fallback when local results are empty"""
        if requests is None:
            return []
        try:
            encoded = requests.utils.quote(query)
            resp = requests.get(
                f"https://api.scryfall.com/cards/search?q={encoded}", timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                cards = data.get("data", [])
                results = []
                for card in cards[:limit]:
                    prices = card.get("prices", {})
                    results.append({
                        "id": card.get("id"),
                        "name": card.get("name"),
                        "type_line": card.get("type_line"),
                        "oracle_text": card.get("oracle_text"),
                        "color_identity": card.get("color_identity", []),
                        "prices_usd": prices.get("usd")
                    })
                return results
        except Exception as e:  # pragma: no cover - network failures
            logger.warning(f"Scryfall search failed: {e}")
        return []

    def validate_and_lookup_cards(self, potential_cards: List[str]) -> Tuple[List[Dict], List[str], List[Dict]]:
        """FIXED: Validate card names against database with better error handling"""
        found_cards = []
        not_found = []
        suggestions = []

        for card_name in potential_cards:
            try:
                # Try exact match first
                card_data = self._lookup_exact_card(card_name)

                if card_data:
                    found_cards.append(card_data)
                    logger.info(f"Found exact match for: {card_name}")
                else:
                    # Try fuzzy matching for suggestions
                    similar_cards = self._find_similar_card_names(card_name)
                    if similar_cards:
                        suggestions.extend(similar_cards[:3])  # Top 3 suggestions per not-found card
                        logger.info(f"No exact match for '{card_name}', found suggestions: {[c['name'] for c in similar_cards[:3]]}")
                    not_found.append(card_name)
            except Exception as e:
                logger.error(f"Error processing card '{card_name}': {e}")
                not_found.append(card_name)

        return found_cards, not_found, suggestions

    def _lookup_exact_card(self, card_name: str) -> Optional[Dict]:
        """FIXED: Look up a card by exact name with better error handling"""
        if not self.rag.db.is_connected:
            return None

        try:
            # Try exact match
            response = self.rag.db.client.table("mtg_cards").select(
                "id, name, type_line, color_identity, oracle_text, mana_cost, prices_usd"
            ).eq("name", card_name).limit(1).execute()

            if response.data:
                return response.data[0]


            # Try case-insensitive match
            response = (
                self.rag.db.client.table("mtg_cards")
                .select(
                    "id, name, type_line, color_identity, oracle_text, mana_cost, prices_usd"
                )
                .ilike("name", card_name)
                .limit(1)
                .execute()
            )

            if response.data:
                return response.data[0]

            # Handle front-face names of double-faced cards
            response = (
                self.rag.db.client.table("mtg_cards")
                .select(
                    "id, name, type_line, color_identity, oracle_text, mana_cost, prices_usd"
                )
                .ilike("name", f"{card_name} //%")
                .limit(1)
                .execute()
            )

            if response.data:
                return response.data[0]

        except Exception as e:
            logger.error(f"Error looking up card '{card_name}': {e}")

        return None

    def _find_similar_card_names(self, card_name: str) -> List[Dict]:
        """FIXED: Find cards with similar names for suggestions"""
        if not self.rag.db.is_connected:
            return []

        try:
            # Try partial matches
            response = self.rag.db.client.table("mtg_cards").select(
                "name, type_line"
            ).ilike("name", f"%{card_name}%").limit(10).execute()

            if response.data:
                return response.data

        except Exception as e:
            logger.error(f"Error finding similar cards for '{card_name}': {e}")

        return []

    def get_combined_color_identity(self, cards: List[Dict]) -> List[str]:
        """Get the combined color identity from multiple cards"""
        all_colors = set()

        for card in cards:
            color_identity = card.get('color_identity')

            # If color identity missing, attempt to fetch from cache/online
            if not color_identity and card.get('name'):
                fetched = self._fetch_color_identity_online(card['name'])
                if fetched:
                    color_identity = fetched
                    card['color_identity'] = fetched

            if isinstance(color_identity, list):
                all_colors.update(color_identity)
            elif isinstance(color_identity, str):
                # Handle string format like "W,U,B"
                colors = [c.strip() for c in color_identity.split(',') if c.strip()]
                all_colors.update(colors)

        return list(all_colors)

    def perform_category_search(self, constraints: Dict[str, Any], query_context: str) -> Dict[str, Any]:
        """NEW: Perform a category-based search using constraints"""
        try:
            logger.info(f"Performing category search with constraints: {constraints}")

            # Build search parameters
            search_params = {}

            # Handle colors
            colors = constraints.get('colors', [])
            if colors:
                search_params['colors'] = colors

            # Handle card types
            card_types = constraints.get('card_types', [])
            if card_types:
                search_params['card_type'] = card_types[0]  # Use first type for now

            # Handle budget
            budget = constraints.get('budget')
            if budget:
                search_params['max_price'] = budget

            # Create search query based on constraints and context
            search_terms = []

            # Add strategy terms
            strategies = constraints.get('strategies', [])
            if strategies:
                search_terms.extend(strategies)

            # Add deck context
            deck_context = constraints.get('deck_context')
            if deck_context:
                search_terms.append(deck_context)

            # Add type context if no specific strategies
            if not strategies and card_types:
                search_terms.extend(card_types)

            # Build the search query
            if search_terms:
                search_query = ' '.join(search_terms)
            else:
                search_query = query_context

            logger.info(f"Using search query: '{search_query}' with params: {search_params}")

            # Perform the search
            if self.rag.embedding_available and self.rag.text_index is not None:
                # Use semantic search
                matching_cards = self.rag.retrieve_cards_by_text(search_query, top_k=30)
            else:
                # Use keyword search
                matching_cards = self.rag.search_cards_by_keyword(search_query, top_k=30)

            # Apply filters
            filtered_cards = self._apply_category_filters(matching_cards, constraints)

            # Fallback to Scryfall search if no results
            if not filtered_cards:
                remote_cards = self._search_scryfall(search_query, limit=40)
                if remote_cards:
                    filtered_cards = self._apply_category_filters(remote_cards, constraints)

            # Add relevance scores and explanations
            for card in filtered_cards:
                card['relevance_score'] = card.get('similarity', 0.7)
                card['category_match_explanation'] = self._generate_category_explanation(card, constraints)

            # Sort by relevance
            filtered_cards.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            return {
                'success': True,
                'query_type': 'category_search',
                'cards': filtered_cards[:40],  # Return top 40
                'constraints_applied': constraints,
                'explanation': self._generate_category_search_explanation(constraints, len(filtered_cards))
            }

        except Exception as e:
            logger.error(f"Error in category search: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _apply_category_filters(self, cards: List[Dict], constraints: Dict[str, Any]) -> List[Dict]:
        """Apply category constraints to filter cards"""
        filtered = []

        for card in cards:
            # Apply color filter
            colors = constraints.get('colors', [])
            if colors:
                card_colors = card.get('color_identity', [])
                if isinstance(card_colors, str):
                    card_colors = [c.strip() for c in card_colors.split(',') if c.strip()]
                elif not isinstance(card_colors, list):
                    card_colors = []

                # Handle colorless
                if 'C' in colors:
                    if card_colors:  # Skip non-colorless cards
                        continue
                else:
                    # For colored constraints, check if card colors are subset of allowed colors
                    if card_colors and not all(color in colors for color in card_colors):
                        continue

            # Apply type filter
            card_types = constraints.get('card_types', [])
            if card_types:
                type_line = card.get('type_line', '')
                if not any(card_type in type_line for card_type in card_types):
                    continue

            # Apply budget filter
            budget = constraints.get('budget')
            if budget:
                price = card.get('prices_usd', 0) or 0
                try:
                    if float(price) > budget:
                        continue
                except (ValueError, TypeError):
                    pass  # Include cards with unknown prices

            filtered.append(card)

        return filtered

    def _generate_category_explanation(self, card: Dict, constraints: Dict[str, Any]) -> str:
        """Generate explanation for why a card matches the category"""
        explanations = []

        # Color explanation
        colors = constraints.get('colors', [])
        if colors:
            if 'C' in colors:
                explanations.append("Colorless card")
            else:
                color_names = {'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': 'Red', 'G': 'Green'}
                color_text = '/'.join([color_names.get(c, c) for c in colors])
                explanations.append(f"Fits {color_text} color identity")

        # Type explanation
        card_types = constraints.get('card_types', [])
        if card_types:
            explanations.append(f"{card_types[0]} card")

        # Budget explanation
        budget = constraints.get('budget')
        if budget:
            price = card.get('prices_usd', 0) or 0
            explanations.append(f"${price} (within budget)")

        return "; ".join(explanations) if explanations else "Matches search criteria"

    def _generate_category_search_explanation(self, constraints: Dict[str, Any], result_count: int) -> str:
        """Generate explanation for category search results"""
        parts = [f"Found {result_count} cards"]

        # Add constraint explanations
        colors = constraints.get('colors', [])
        if colors:
            if 'C' in colors:
                parts.append("that are colorless")
            else:
                color_names = {'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': 'Red', 'G': 'Green'}
                color_text = '/'.join([color_names.get(c, c) for c in colors])
                parts.append(f"in {color_text}")

        card_types = constraints.get('card_types', [])
        if card_types:
            parts.append(f"of type {'/'.join(card_types)}")

        strategies = constraints.get('strategies', [])
        if strategies:
            parts.append(f"for {'/'.join(strategies)} strategies")

        budget = constraints.get('budget')
        if budget:
            parts.append(f"under ${budget}")

        return " ".join(parts) + "."

    def find_synergistic_cards(self, seed_cards: List[Dict], color_identity: List[str], query_context: str) -> List[Dict]:
        """FIXED: Find cards that synergize with the seed cards"""
        all_synergistic = []

        try:
            # Method 1: Text-based similarity
            text_similar = self._find_text_similar_cards(seed_cards, query_context)
            all_synergistic.extend(text_similar)

            # Method 2: Rules-based synergy (with error handling)
            try:
                rules_synergistic = self._find_rules_synergistic_cards(seed_cards)
                all_synergistic.extend(rules_synergistic)
            except Exception as e:
                logger.error(f"Error finding rules synergistic cards: {e}")

            # Method 3: Mechanic-based synergy (with error handling)
            try:
                mechanic_synergistic = self._find_mechanic_synergistic_cards(seed_cards)
                all_synergistic.extend(mechanic_synergistic)
            except Exception as e:
                logger.error(f"Error finding mechanic synergistic cards: {e}")

        except Exception as e:
            logger.error(f"Error in find_synergistic_cards: {e}")
            return []

        # Remove duplicates and filter by color identity
        unique_cards = self._deduplicate_and_filter_cards(all_synergistic, color_identity, seed_cards)

        # Calculate combined synergy scores
        scored_cards = self._calculate_combined_synergy_scores(unique_cards, seed_cards, query_context)

        # Sort by combined score and return top results
        scored_cards.sort(key=lambda x: x.get('combined_synergy_score', 0), reverse=True)

        return scored_cards[:40]  # Return top 40 synergistic cards

    def _find_text_similar_cards(self, seed_cards: List[Dict], query_context: str) -> List[Dict]:
        """FIXED: Find cards with similar text to seed cards"""
        similar_cards = []

        try:
            # Combine oracle texts from seed cards (with None handling)
            combined_texts = []
            for card in seed_cards:
                oracle_text = card.get('oracle_text', '')
                if oracle_text and isinstance(oracle_text, str):
                    combined_texts.append(oracle_text)

            combined_text = " ".join(combined_texts)
            if query_context and isinstance(query_context, str):
                combined_text += " " + query_context

            # Use RAG system to find similar cards (if text available)
            if combined_text.strip() and self.rag.embedding_available:
                text_results = self.rag.retrieve_cards_by_text(combined_text, top_k=30)
                for card in text_results:
                    if card and isinstance(card, dict):  # Safety check
                        card['text_similarity_score'] = card.get('similarity', 0.5)
                        similar_cards.append(card)

        except Exception as e:
            logger.error(f"Error finding text similar cards: {e}")

        return similar_cards

    def _find_rules_synergistic_cards(self, seed_cards: List[Dict]) -> List[Dict]:
        """FIXED: Find cards that synergize based on rules interactions"""
        synergistic_cards = []

        try:
            # Import synergy calculation with error handling
            from src.update_api_server import calculate_card_synergies # Ensure this import is valid in your project structure

            seed_card_names = []
            for card in seed_cards:
                name = card.get('name')
                if name and isinstance(name, str):
                    seed_card_names.append(name)

            if not seed_card_names:
                return []

            synergy_results = calculate_card_synergies(seed_card_names, count=25)

            for result in synergy_results:
                if result and isinstance(result, dict):  # Safety check
                    card_item = result.get('card') # Renamed to avoid conflict
                    if card_item and isinstance(card_item, dict):
                        card_item['rules_synergy_score'] = result.get('total_score', 0)
                        card_item['synergy_explanations'] = result.get('explanations', [])
                        synergistic_cards.append(card_item)

        except ImportError:
            logger.error("Could not import calculate_card_synergies. Rules-based synergy will be skipped.")
        except Exception as e:
            logger.error(f"Error finding rules synergistic cards: {e}")

        return synergistic_cards

    def _find_mechanic_synergistic_cards(self, seed_cards: List[Dict]) -> List[Dict]:
        """FIXED: Find cards that share mechanics with seed cards"""
        mechanic_cards = []

        try:
            # Extract key mechanics from seed cards (with None handling)
            seed_mechanics = set()
            for card in seed_cards:
                oracle_text = card.get('oracle_text', '')
                type_line = card.get('type_line', '')

                # Safety checks for None values
                if oracle_text is None:
                    oracle_text = ''
                if type_line is None:
                    type_line = ''

                text_to_analyze = oracle_text.lower() + " " + type_line.lower()
                mechanics = self._extract_mechanics_from_text(text_to_analyze)
                seed_mechanics.update(mechanics)

            if seed_mechanics:
                # Search for cards with similar mechanics
                mechanic_query = " OR ".join(list(seed_mechanics)[:5])  # Limit to 5 mechanics
                mechanic_results = self.rag.search_cards_by_keyword(mechanic_query, top_k=25)

                for card_item in mechanic_results: # Renamed to avoid conflict
                    if card_item and isinstance(card_item, dict):  # Safety check
                        oracle_text_item = card_item.get('oracle_text', '') or '' # Renamed
                        type_line_item = card_item.get('type_line', '') or '' # Renamed

                        card_text = oracle_text_item.lower() + " " + type_line_item.lower()
                        card_mechanics = self._extract_mechanics_from_text(card_text)

                        # Calculate mechanic overlap score
                        overlap = len(seed_mechanics.intersection(card_mechanics))
                        card_item['mechanic_synergy_score'] = overlap / max(len(seed_mechanics), 1)
                        mechanic_cards.append(card_item)

        except Exception as e:
            logger.error(f"Error finding mechanic synergistic cards: {e}")

        return mechanic_cards

    def _extract_mechanics_from_text(self, text: str) -> set:
        """FIXED: Extract game mechanics from card text with None handling"""
        if not text or not isinstance(text, str):
            return set()

        mechanics = set()

        # Common MTG mechanics patterns
        mechanic_patterns = {
            'artifact': r'\bartifact\b',
            'token': r'\btoken\b|create[s]* [a-z0-9/+\s]*\b(?:creature|artifact|enchantment)\b',
            'counter': r'\bcounter\b(?!\s+target)|counters?|proliferate',
            'draw': r'\bdraw[s]* (?:a )?card[s]*\b',
            'graveyard': r'\bgraveyard\b|from your graveyard',
            'sacrifice': r'\bsacrifice\b',
            'tutor': r'search your library',
            'ramp': r'search your library for.*land|add.*mana',
            'flying': r'\bflying\b',
            'trample': r'\btrample\b',
            'lifelink': r'\blifelink\b',
            'deathtouch': r'\bdeathtouch\b',
            'haste': r'\bhaste\b',
            'vigilance': r'\bvigilance\b',
            'first_strike': r'\bfirst strike\b',
            'double_strike': r'\bdouble strike\b',
            'hexproof': r'\bhexproof\b',
            'indestructible': r'\bindestructible\b',
            'flash': r'\bflash\b',
            'enters_battlefield': r'enters the battlefield|when .* enters',
            'leaves_battlefield': r'leaves the battlefield|when .* leaves',
            'triggered': r'\bwhen\b|\bwhenever\b|at the beginning',
            'activated': r':\s*[^:]*$',  # Activated abilities
        }

        try:
            for mechanic, pattern in mechanic_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    mechanics.add(mechanic)
        except Exception as e:
            logger.error(f"Error extracting mechanics from text: {e}")

        return mechanics

    def _deduplicate_and_filter_cards(self, all_cards: List[Dict], color_identity: List[str], seed_cards: List[Dict]) -> List[Dict]:
        """FIXED: Remove duplicates and filter by color identity"""
        seen_names = set()
        seed_names = {card.get('name', '') for card in seed_cards if card.get('name')}
        filtered_cards = []

        for card_item in all_cards: # Renamed to avoid conflict
            if not card_item or not isinstance(card_item, dict):  # Safety check
                continue

            card_name = card_item.get('name', '')

            # Skip duplicates and seed cards
            if card_name in seen_names or card_name in seed_names or not card_name:
                continue

            # Check color identity compatibility
            if self._is_color_compatible(card_item, color_identity):
                seen_names.add(card_name)
                filtered_cards.append(card_item)

        return filtered_cards

    def _is_color_compatible(self, card: Dict, allowed_colors: List[str]) -> bool:
        """Check if card can be played in the given color identity"""
        card_colors = card.get('color_identity')

        if card_colors is None:
            return False

        # Handle different color identity formats
        if isinstance(card_colors, str):
            card_colors = [c.strip() for c in card_colors.split(',') if c.strip()]
        elif not isinstance(card_colors, list):
            return False

        # If still empty after fetching, treat as unknown (incompatible)
        if not card_colors:
            return False

        # Check if all card colors are in allowed colors
        return all(color in allowed_colors for color in card_colors)

    def _calculate_combined_synergy_scores(self, cards: List[Dict], seed_cards: List[Dict], query_context: str) -> List[Dict]:
        """FIXED: Calculate combined synergy scores with better error handling"""
        for card in cards:
            try:
                # Get individual scores (default to 0 if not present)
                text_score = float(card.get('text_similarity_score', 0) or 0)
                rules_score = float(card.get('rules_synergy_score', 0) or 0)
                mechanic_score = float(card.get('mechanic_synergy_score', 0) or 0)

                # Normalize scores to 0-1 range
                text_score = min(max(text_score, 0), 1)
                rules_score = min(max(rules_score / 2, 0), 1)  # Rules scores can be > 1
                mechanic_score = min(max(mechanic_score, 0), 1)

                # Weighted combination (you can adjust these weights)
                combined_score = (
                    0.3 * text_score +      # Text similarity
                    0.5 * rules_score +     # Rules interactions (most important)
                    0.2 * mechanic_score    # Mechanic overlap
                )

                card['combined_synergy_score'] = combined_score
                card['score_breakdown'] = {
                    'text_similarity': text_score,
                    'rules_synergy': rules_score,
                    'mechanic_synergy': mechanic_score,
                    'combined': combined_score
                }
            except Exception as e:
                logger.error(f"Error calculating scores for card {card.get('name', 'unknown')}: {e}")
                # Set default scores
                card['combined_synergy_score'] = 0.0
                card['score_breakdown'] = {
                    'text_similarity': 0.0,
                    'rules_synergy': 0.0,
                    'mechanic_synergy': 0.0,
                    'combined': 0.0
                }

        return cards

    def process_universal_query(self, query_text: str, context: Dict = None) -> Dict:
        """FIXED: Process a universal natural language query with improved logic"""
        try:
            logger.info(f"Processing universal query: {query_text}")

            # Step 1: Detect query type and extract constraints
            query_analysis = self.detect_query_type(query_text)
            query_type = query_analysis['type']
            constraints = query_analysis.get('constraints', {})

            logger.info(f"Query type detected: {query_type}")
            logger.info(f"Constraints extracted: {constraints}")

            # Step 2: Handle based on query type
            if query_type == 'card_specific':
                # This is a card-specific query
                potential_cards = query_analysis.get('cards', [])

                if not potential_cards:
                    logger.info("No valid cards found in card-specific query, falling back to category search")
                    return self.perform_category_search(constraints, query_text)

                # Validate and lookup cards
                found_cards, not_found_cards, suggestions = self.validate_and_lookup_cards(potential_cards)

                if not found_cards:
                    logger.info(f"No valid cards found, providing suggestions")
                    return {
                        'success': True,
                        'query_type': 'card_not_found',
                        'original_query': query_text,
                        'searched_for': potential_cards,
                        'suggestions': suggestions,
                        'cards': suggestions[:10],
                        'explanation': f"Could not find cards: {', '.join(not_found_cards)}. Here are some similar cards you might have meant."
                    }

                # Get combined color identity
                color_identity = self.get_combined_color_identity(found_cards)

                # Find synergistic cards
                synergistic_cards = self.find_synergistic_cards(found_cards, color_identity, query_text)

                # Format response
                response = {
                    'success': True,
                    'query_type': 'card_synergy_search',
                    'original_query': query_text,
                    'detected_cards': [card['name'] for card in found_cards],
                    'color_identity': color_identity,
                    'cards': synergistic_cards,
                    'explanation': self._generate_explanation(found_cards, synergistic_cards, not_found_cards, suggestions)
                }

                if not_found_cards:
                    response['not_found'] = not_found_cards
                    response['suggestions'] = suggestions

                logger.info(f"Successfully processed card-specific query, found {len(synergistic_cards)} synergistic cards")
                return response

            elif query_type == 'category_search':
                # This is a category search query
                logger.info("Processing as category search")
                return self.perform_category_search(constraints, query_text)

            else:
                # General search fallback
                logger.info("Processing as general search")
                return self._fallback_general_search(query_text, context)

        except Exception as e:
            logger.error(f"Error processing universal query: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_query': query_text
            }

    def _fallback_general_search(self, query_text: str, context: Dict = None) -> Dict:
        """FIXED: Fallback to general search when no cards are detected"""
        try:
            # Use semantic search if available, otherwise keyword search
            matching_cards = []

            if self.rag.embedding_available and self.rag.text_index is not None:
                logger.info(f"Using semantic search for general query: {query_text}")
                matching_cards = self.rag.retrieve_cards_by_text(query_text, top_k=20)
            else:
                # Fallback to keyword search
                logger.info(f"Using keyword search for general query")
                key_terms = self._extract_key_terms(query_text)
                if key_terms:
                    for term in key_terms[:3]:  # Use top 3 terms
                        keyword_matches = self.rag.search_cards_by_keyword(term, top_k=8)
                        matching_cards.extend(keyword_matches)

                    # Remove duplicates
                    seen_ids = set()
                    unique_cards = []
                    for card in matching_cards:
                        if card and card.get('id') not in seen_ids:
                            seen_ids.add(card['id'])
                            unique_cards.append(card)
                    matching_cards = unique_cards[:30]
                else:
                    # Last resort: search for "artifact"
                    matching_cards = self.rag.search_cards_by_keyword("artifact", top_k=30)

            # If still no results, try Scryfall
            if not matching_cards:
                matching_cards = self._search_scryfall(query_text, limit=30)

            # Apply context filters if provided
            if context and matching_cards:
                budget = context.get('budget')
                if budget:
                    matching_cards = [
                        card for card in matching_cards
                        if float(card.get('prices_usd', 0) or 0) <= budget
                    ]

            # Add relevance scores
            for card in matching_cards:
                card['relevance_score'] = card.get('similarity', 0.5)

            # Sort by relevance
            matching_cards.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            return {
                'success': True,
                'query_type': 'general_search',
                'original_query': query_text,
                'cards': matching_cards[:30],
                'explanation': f"Found {len(matching_cards[:30])} cards matching your search."
            }

        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_query': query_text
            }

    def _generate_explanation(self, found_cards: List[Dict], synergistic_cards: List[Dict],
                            not_found_cards: List[str], suggestions: List[Dict]) -> str:
        """Generate an explanation of the search results"""
        explanation_parts = []

        if found_cards:
            card_names = [card['name'] for card in found_cards]
            explanation_parts.append(f"Found synergies for: {', '.join(card_names)}")

            # Get color identity info
            color_identity = self.get_combined_color_identity(found_cards)
            if color_identity:
                color_text = '/'.join(color_identity)
                explanation_parts.append(f"Filtered for {color_text} color identity")
            else:
                explanation_parts.append("Filtered for colorless cards")

        if synergistic_cards:
            explanation_parts.append(f"Showing {len(synergistic_cards)} synergistic cards ranked by compatibility")

        if not_found_cards:
            explanation_parts.append(f"Note: Could not find: {', '.join(not_found_cards)}")
            if suggestions:
                suggestion_names = [s['name'] for s in suggestions[:3]]
                explanation_parts.append(f"Did you mean: {', '.join(suggestion_names)}?")

        return ". ".join(explanation_parts) + "."