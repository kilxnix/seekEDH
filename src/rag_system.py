# src/rag_system.py - Enhanced RAG System with Image and Dual-Faced Card Support
import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import faiss

# Configure logging
logger = logging.getLogger("RAGSystem")

# Try importing with better error handling
try:
    import torch
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
    logger.info("SentenceTransformer successfully imported")
except Exception as e:
    logger.error(f"Error importing SentenceTransformer: {e}")
    EMBEDDING_AVAILABLE = False

from src.db_interface import DatabaseInterface

class MTGRetrievalSystem:
    """Enhanced retrieval system for MTG cards using FAISS with dual-faced card support"""
    
    def __init__(self, db_interface=None, embedding_model_name="all-MiniLM-L6-v2"):
        """
        Initialize the retrieval system
        
        Args:
            db_interface: DatabaseInterface instance
            embedding_model_name: Name of the sentence transformer model
        """
        self.db = db_interface or DatabaseInterface()
        
        # Initialize embedding model
        self.embedding_model = None
        self.embedding_available = False
        
        if EMBEDDING_AVAILABLE and embedding_model_name:
            try:
                # Check for GPU
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Using device: {device} for embeddings")
                
                # Load model
                self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
                self.embedding_available = True
                logger.info(f"Loaded embedding model: {embedding_model_name}")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
        
        self.text_index = None
        self.price_index = None
        self.card_ids = []
        self.text_card_ids = []
        self.embedding_metadata = []  # Store metadata about each embedding (for dual-faced cards)
        
        # Initialize indexes
        if self.db.is_connected:
            self.initialize_indexes()
    
    def initialize_indexes(self):
        """Initialize the FAISS indexes from the database"""
        try:
            # Fetch cards with price embeddings
            logger.info("Fetching cards with price embeddings from database")
            response = self.db.client.table("mtg_cards").select("id, scryfall_id, name, price_embedding").not_.is_("price_embedding", "null").limit(10000).execute()
            
            if not response.data:
                logger.warning("No cards with price embeddings found in database")
            else:
                cards_with_embeddings = response.data
                logger.info(f"Found {len(cards_with_embeddings)} cards with price embeddings")
                
                # Extract card IDs and price embeddings
                self.card_ids = [card['id'] for card in cards_with_embeddings]
                
                # Parse string representations of embeddings if needed
                price_embeddings = []
                for card in cards_with_embeddings:
                    embedding = card.get('price_embedding')
                    if embedding is not None:
                        # Check if embedding is a string and parse it
                        if isinstance(embedding, str):
                            try:
                                # Remove brackets and split by commas
                                embedding = embedding.strip('[]').split(',')
                                embedding = [float(x) for x in embedding]
                            except Exception as e:
                                logger.warning(f"Could not parse embedding string: {e}")
                                continue
                        price_embeddings.append(embedding)
                
                if not price_embeddings:
                    logger.warning("No valid price embeddings found in database")
                else:
                    # Convert to numpy array
                    price_embeddings_array = np.array(price_embeddings).astype('float32')
                    
                    # Create price index
                    dimension = price_embeddings_array.shape[1]
                    self.price_index = faiss.IndexFlatL2(dimension)
                    self.price_index.add(price_embeddings_array)
                    
                    logger.info(f"Initialized price index with {self.price_index.ntotal} embeddings of dimension {dimension}")
            
            # Initialize text index if embedding model is available
            if self.embedding_available and self.embedding_model is not None:
                self._initialize_text_index()
                
            return True
            
        except Exception as e:
            logger.error(f"Error initializing indexes: {e}")
            return False
    
    def _initialize_text_index(self):
        """FIXED: Initialize text index with dimension validation"""
        try:
            # Fetch cards with text embeddings
            logger.info("Fetching cards with text embeddings from database")
            response = self.db.client.table("mtg_cards").select(
                "id, name, oracle_text, text_embedding"
            ).not_.is_("text_embedding", "null").limit(5000).execute()
            
            if not response.data:
                logger.warning("No cards with text embeddings found in database")
                return False
            
            cards_with_text = response.data
            logger.info(f"Found {len(cards_with_text)} cards with text embeddings")
            
            # Extract and validate embeddings
            valid_embeddings = []
            valid_card_ids = []
            expected_dimension = 384  # Dimension for all-MiniLM-L6-v2
            
            for i, card in enumerate(cards_with_text):
                try:
                    embedding = card.get('text_embedding')
                    if embedding is None:
                        continue
                    
                    # Handle different embedding formats
                    if isinstance(embedding, list):
                        embedding_array = np.array(embedding, dtype=np.float32)
                    elif isinstance(embedding, str):
                        # Parse string representation
                        try:
                            # Remove brackets and split by commas
                            embedding_str = embedding.strip('[]')
                            embedding_values = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
                            embedding_array = np.array(embedding_values, dtype=np.float32)
                        except:
                            logger.warning(f"Could not parse string embedding for card {card.get('name', 'unknown')}")
                            continue
                    else:
                        logger.warning(f"Unknown embedding format for card {card.get('name', 'unknown')}: {type(embedding)}")
                        continue
                    
                    # Validate dimension
                    if embedding_array.shape == (expected_dimension,):
                        valid_embeddings.append(embedding_array)
                        valid_card_ids.append(card['id'])
                    else:
                        logger.warning(f"Invalid embedding dimension for card {card.get('name', 'unknown')}: "
                                    f"expected {expected_dimension}, got {embedding_array.shape}")
                        
                        # Debug: Log first few problematic cards
                        if len(valid_embeddings) < 5:
                            logger.debug(f"Card {i}: {card.get('name', 'unknown')} - "
                                    f"embedding type: {type(embedding)}, shape: {embedding_array.shape}")
                            if hasattr(embedding, '__len__') and len(embedding) < 10:
                                logger.debug(f"Embedding content: {embedding}")
                
                except Exception as e:
                    logger.warning(f"Error processing embedding for card {card.get('name', 'unknown')}: {e}")
                    continue
            
            if not valid_embeddings:
                logger.error("No valid text embeddings found")
                self.text_index = None
                return False
            
            logger.info(f"Validated {len(valid_embeddings)} embeddings out of {len(cards_with_text)} total")
            
            # Create embeddings array
            text_embeddings_array = np.stack(valid_embeddings).astype('float32')
            
            # Store valid card IDs
            self.text_card_ids = valid_card_ids
            
            # Create text index optimized for similarity search (cosine similarity)
            dimension = text_embeddings_array.shape[1]
            self.text_index = faiss.IndexFlatIP(dimension)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(text_embeddings_array)
            self.text_index.add(text_embeddings_array)
            
            logger.info(f"Successfully initialized text index with {self.text_index.ntotal} embeddings of dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing text index: {e}")
            self.text_index = None
            return False
    
    def retrieve_cards_by_text(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """FIXED: Retrieve cards with better error handling"""
        if not self.db.is_connected:
            logger.error("Not connected to database")
            return []
        
        # Use embedding-based search if available
        if self.embedding_available and self.embedding_model is not None and self.text_index is not None:
            try:
                # Generate embedding for query text
                query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True)[0]
                query_embedding = query_embedding.reshape(1, -1).astype('float32')
                
                # Normalize query for cosine similarity
                faiss.normalize_L2(query_embedding)
                
                # Search the index
                distances, indices = self.text_index.search(query_embedding, min(top_k, self.text_index.ntotal))
                
                # Check if we got valid results
                if len(indices) == 0 or len(indices[0]) == 0:
                    logger.warning("No results from embedding search, falling back to keyword search")
                    return self.search_cards_by_keyword(query_text, top_k)
                
                # Get the result indices
                result_indices = indices[0]
                similar_cards = []
                
                for i, idx in enumerate(result_indices):
                    if idx >= 0 and idx < len(self.text_card_ids):  # Validate index
                        card_id = self.text_card_ids[idx]
                        
                        card_response = self.db.client.table("mtg_cards").select(
                            "name, oracle_text, type_line, mana_cost, colors, color_identity, prices_usd"
                        ).eq("id", card_id).limit(1).execute()
                        
                        if card_response.data:
                            card = card_response.data[0]
                            card["similarity"] = float(distances[0][i]) if i < len(distances[0]) else 0.5
                            similar_cards.append(card)
                
                logger.info(f"Found {len(similar_cards)} similar cards using embedding search")
                return similar_cards
                
            except Exception as e:
                logger.error(f"Error in embedding-based search: {e}")
                logger.info("Falling back to keyword search")
        
        # Fallback to keyword search
        return self.search_cards_by_keyword(query_text, top_k)
    
    def retrieve_cards_by_price(self, query_card: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve cards with similar price characteristics
        
        Args:
            query_card: Name of the card to use as query
            top_k: Number of results to return
            
        Returns:
            List of similar cards
        """
        if not self.db.is_connected:
            logger.error("Not connected to database")
            return []
        
        if not self.price_index:
            logger.error("Price index not initialized")
            return []
        
        try:
            # Fetch the query card
            response = self.db.client.table("mtg_cards").select("id, price_embedding").eq("name", query_card).limit(1).execute()
            
            if not response.data:
                logger.error(f"Card '{query_card}' not found")
                return []
            
            query_card_data = response.data[0]
            query_embedding = query_card_data.get('price_embedding')
            
            if not query_embedding:
                logger.error(f"No price embedding found for card '{query_card}'")
                return []
            
            # Parse string embedding if needed
            if isinstance(query_embedding, str):
                try:
                    # Remove brackets and split by commas
                    query_embedding = query_embedding.strip('[]').split(',')
                    query_embedding = [float(x) for x in query_embedding]
                except Exception as e:
                    logger.error(f"Could not parse query embedding string: {e}")
                    return []
            
            # Convert to numpy array
            query_embedding_array = np.array([query_embedding]).astype('float32')
            
            # Search the index
            distances, indices = self.price_index.search(query_embedding_array, top_k + 1)  # +1 to account for the query card itself
            
            # Get the result indices
            result_indices = indices[0]
            
            # Fetch the cards
            similar_cards = []
            for idx in result_indices:
                if idx < len(self.card_ids):
                    card_id = self.card_ids[idx]
                    # Skip the query card
                    if card_id == query_card_data.get('id'):
                        continue
                    
                    card_response = self.db.client.table("mtg_cards").select("name, prices_usd, prices_usd_foil, rarity, set_name").eq("id", card_id).limit(1).execute()
                    if card_response.data:
                        card = card_response.data[0]
                        similar_cards.append({
                            "id": card_id,
                            "name": card["name"],
                            "price_usd": card.get("prices_usd"),
                            "price_usd_foil": card.get("prices_usd_foil"),
                            "rarity": card.get("rarity"),
                            "set": card.get("set_name"),
                            "distance": float(distances[0][len(similar_cards)])
                        })
                        
                    # Stop when we have enough results
                    if len(similar_cards) >= top_k:
                        break
            
            return similar_cards
            
        except Exception as e:
            logger.error(f"Error retrieving cards by price: {e}")
            return []
    
    def search_cards_by_keyword(self, keyword: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for cards by keyword in the oracle text
        
        Args:
            keyword: Keyword to search for
            top_k: Number of results to return
            
        Returns:
            List of matching cards
        """
        if not self.db.is_connected:
            logger.error("Not connected to database")
            return []
        
        try:
            # Search for cards containing the keyword
            response = self.db.client.table("mtg_cards").select("id, name, oracle_text, type_line, mana_cost, color_identity, prices_usd").ilike("oracle_text", f"%{keyword}%").limit(top_k).execute()
            
            if not response.data:
                logger.warning(f"No cards found matching keyword '{keyword}'")
                return []
            
            # Format the results
            matching_cards = [{
                "id": card["id"],
                "name": card["name"],
                "oracle_text": card.get("oracle_text"),
                "type_line": card.get("type_line"),
                "mana_cost": card.get("mana_cost"),
                "color_identity": card.get("color_identity"),
                "prices_usd": card.get("prices_usd")
            } for card in response.data]
            
            return matching_cards
            
        except Exception as e:
            logger.error(f"Error searching cards by keyword: {e}")
            return []
    
    def get_card_with_all_embeddings(self, card_name: str) -> Optional[Dict]:
        """Get a card with all its embeddings (main + faces)"""
        if not self.db.is_connected:
            return None
            
        try:
            response = self.db.client.rpc(
                'get_card_with_embeddings',
                {'p_card_name': card_name}
            ).execute()
            
            if response.data:
                return response.data[0]
                
        except Exception as e:
            logger.error(f"Error getting card embeddings for {card_name}: {e}")
            
        return None

    def search_by_embedding_with_faces(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search using the database RPC function that handles dual-faced cards"""
        if not self.db.is_connected:
            return []
            
        try:
            response = self.db.client.rpc(
                'search_card_embeddings',
                {
                    'query_embedding': query_embedding.tolist(),
                    'match_threshold': 0.3,
                    'match_count': top_k,
                    'include_faces': True
                }
            ).execute()
            
            if response.data:
                # Convert the results to the expected format
                results = []
                for row in response.data:
                    # Fetch full card data
                    card_response = self.db.client.table("mtg_cards").select("*").eq("id", row['card_id']).limit(1).execute()
                    
                    if card_response.data:
                        card = card_response.data[0]
                        card['similarity'] = row['similarity']
                        card['embedding_type'] = row['embedding_type']
                        card['matched_face'] = row['face_name']
                        results.append(card)
                
                return results
                
        except Exception as e:
            logger.error(f"Error in database embedding search: {e}")
            
        return []
    
    def generate_deck_recommendation(self, strategy: str, commander: Optional[str] = None, budget: Optional[float] = None, colors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a deck recommendation based on strategy and constraints
        
        Args:
            strategy: Deck strategy description
            commander: Optional commander name
            budget: Optional budget constraint
            colors: Optional color identity constraint
            
        Returns:
            Deck recommendation
        """
        if not self.db.is_connected:
            logger.error("Not connected to database")
            return {"error": "Not connected to database"}
        
        try:
            # Extract keywords from the strategy
            keywords = [word.lower() for word in strategy.split() if len(word) > 3]
            
            # Try to use embedding-based search for keywords if available
            all_matches = []
            if self.embedding_available and self.embedding_model is not None and self.text_index is not None:
                try:
                    # Use the full strategy text for better semantic matching
                    matching_cards = self.retrieve_cards_by_text(strategy, top_k=50)
                    all_matches.extend(matching_cards)
                except Exception as e:
                    logger.error(f"Error using embedding search for strategy: {e}")
            
            # Fallback or supplement with keyword search
            if not all_matches:
                for keyword in keywords[:5]:  # Limit to first 5 keywords
                    matches = self.search_cards_by_keyword(keyword, top_k=10)
                    all_matches.extend(matches)
            
            # Remove duplicates
            seen_ids = set()
            unique_matches = []
            for card in all_matches:
                if card["id"] not in seen_ids:
                    seen_ids.add(card["id"])
                    unique_matches.append(card)
            
            if not unique_matches:
                return {"error": "No cards found matching the strategy"}
            
            # If commander is specified, get its details
            commander_details = None
            if commander:
                response = self.db.client.table("mtg_cards").select("id, color_identity").eq("name", commander).limit(1).execute()
                if response.data:
                    commander_details = response.data[0]
                    
                    # Use commander's color identity if colors not specified
                    if not colors and commander_details.get('color_identity'):
                        colors = commander_details.get('color_identity')
            
            # Filter cards by color identity
            if colors is not None:  # Changed from "if colors:"
                filtered_cards = []
                for card in unique_matches:
                    card_response = self.db.client.table("mtg_cards").select("color_identity").eq("id", card["id"]).limit(1).execute()
                    if card_response.data:
                        card_colors = card_response.data[0].get('color_identity', [])
                        # For colorless (empty list), only allow colorless cards
                        if not colors and not card_colors:
                            filtered_cards.append(card)
                        # For colored, check subset
                        elif colors and all(color in colors for color in card_colors):
                            filtered_cards.append(card)
                unique_matches = filtered_cards
            
            # Filter by budget if specified
            if budget is not None:
                budget_cards = []
                total_cost = 0
                
                # Sort by relevance/similarity
                if unique_matches and "similarity" in unique_matches[0]:
                    unique_matches.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                
                # Add cards until we hit budget
                for card in unique_matches:
                    # Get card price
                    price = card.get("prices_usd")
                    if price is None:
                        # Try to fetch price if not available
                        card_response = self.db.client.table("mtg_cards").select("prices_usd").eq("id", card["id"]).limit(1).execute()
                        if card_response.data:
                            price = card_response.data[0].get('prices_usd')
                    
                    # Add card if it fits in budget
                    if price is not None and isinstance(price, (int, float)):
                        if total_cost + price <= budget:
                            budget_cards.append(card)
                            total_cost += price
                    else:
                        # Include cards with unknown price but track them
                        card["price_unknown"] = True
                        budget_cards.append(card)
                
                unique_matches = budget_cards
            
            # Build a simple deck recommendation
            deck = {
                "commander": commander,
                "strategy": strategy,
                "budget": budget,
                "cards": unique_matches[:60],  # Limit to 60 cards
                "total_cards": len(unique_matches[:60])
            }
            
            return deck
            
        except Exception as e:
            logger.error(f"Error generating deck recommendation: {e}")
            return {"error": str(e)}