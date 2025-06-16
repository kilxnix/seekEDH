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
    """Retrieval system for MTG cards using FAISS"""
    
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
                return False
            
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
                return False
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
        """Initialize the text embedding index from the database"""
        try:
            # Fetch cards with oracle text
            logger.info("Fetching cards with oracle text from database")
            response = self.db.client.table("mtg_cards").select(
                "id, name, oracle_text"
            ).not_.is_("oracle_text", "null").limit(5000).execute()
            
            if not response.data:
                logger.warning("No cards with oracle text found in database")
                return False
            
            cards_with_text = response.data
            logger.info(f"Found {len(cards_with_text)} cards with oracle text")
            
            # Extract texts and ids
            texts = [card.get('oracle_text', '') for card in cards_with_text if card.get('oracle_text')]
            self.text_card_ids = [card['id'] for card in cards_with_text if card.get('oracle_text')]
            
            if not texts:
                logger.warning("No valid oracle texts found in database")
                return False
            
            # Generate embeddings in batches
            batch_size = 128
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                logger.info(f"Generating embeddings for batch {i+1}/{total_batches}")
                batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)
            
            text_embeddings_array = np.vstack(all_embeddings).astype('float32')
            
            # Create text index optimized for similarity search (cosine similarity)
            dimension = text_embeddings_array.shape[1]
            self.text_index = faiss.IndexFlatIP(dimension)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(text_embeddings_array)
            self.text_index.add(text_embeddings_array)
            
            logger.info(f"Initialized text index with {self.text_index.ntotal} embeddings of dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing text index: {e}")
            return False
    
    def retrieve_cards_by_text(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve cards with similar text/oracle text to the query
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            
        Returns:
            List of similar cards
        """
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
                distances, indices = self.text_index.search(query_embedding, top_k)
                
                # Get the result indices
                result_indices = indices[0]
                
                # Fetch the cards
                similar_cards = []
                for i, idx in enumerate(result_indices):
                    if idx < len(self.text_card_ids):
                        card_id = self.text_card_ids[idx]
                        
                        card_response = self.db.client.table("mtg_cards").select(
                            "name, oracle_text, type_line, mana_cost, colors"
                        ).eq("id", card_id).limit(1).execute()
                        
                        if card_response.data:
                            card = card_response.data[0]
                            similar_cards.append({
                                "id": card_id,
                                "name": card["name"],
                                "oracle_text": card.get("oracle_text"),
                                "type_line": card.get("type_line"),
                                "mana_cost": card.get("mana_cost"),
                                "colors": card.get("colors"),
                                "similarity": float(distances[0][i])  # Similarity score
                            })
                
                logger.info(f"Found {len(similar_cards)} similar cards using embedding search")
                return similar_cards
            except Exception as e:
                logger.error(f"Error in embedding-based search: {e}")
        
        # Fallback to keyword search
        logger.info(f"Using keyword search for query: {query_text}")
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
            response = self.db.client.table("mtg_cards").select("id, name, oracle_text, type_line, mana_cost").ilike("oracle_text", f"%{keyword}%").limit(top_k).execute()
            
            if not response.data:
                logger.warning(f"No cards found matching keyword '{keyword}'")
                return []
            
            # Format the results
            matching_cards = [{
                "id": card["id"],
                "name": card["name"],
                "oracle_text": card.get("oracle_text"),
                "type_line": card.get("type_line"),
                "mana_cost": card.get("mana_cost")
            } for card in response.data]
            
            return matching_cards
            
        except Exception as e:
            logger.error(f"Error searching cards by keyword: {e}")
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
            if colors:
                filtered_cards = []
                for card in unique_matches:
                    # Get card's color identity
                    card_response = self.db.client.table("mtg_cards").select("color_identity").eq("id", card["id"]).limit(1).execute()
                    if card_response.data:
                        card_colors = card_response.data[0].get('color_identity', [])
                        # Check if card color identity is a subset of allowed colors
                        if all(color in colors for color in card_colors):
                            filtered_cards.append(card)
                
                unique_matches = filtered_cards
            
            # Filter by budget if specified
            if budget is not None:
                budget_cards = []
                total_cost = 0
                
                # Sort by relevance/similarity
                if "similarity" in unique_matches[0]:
                    unique_matches.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                
                # Add cards until we hit budget
                for card in unique_matches:
                    # Get card price
                    price = card.get("price_usd")
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