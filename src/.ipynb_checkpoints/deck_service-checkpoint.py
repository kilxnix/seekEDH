# src/deck_service.py
import os
import re
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from supabase import create_client
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeckGenerationService:
    def __init__(self):
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(supabase_url, supabase_key)
        
        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize embedding model
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load embeddings and build FAISS index
        self.initialize_search_index()
    
    def initialize_search_index(self):
        """Initialize the FAISS search index with card embeddings"""
        try:
            # Query for all card embeddings
            # In a real application, you'd batch this or handle it differently for large datasets
            response = self.supabase.table("mtg_cards").select("id,embedding").execute()
            cards = response.data
            
            if not cards:
                logger.warning("No cards found in database")
                self.search_index = None
                self.card_ids = []
                return
            
            # Extract card IDs and embeddings
            self.card_ids = [card['id'] for card in cards]
            embeddings = np.array([card['embedding'] for card in cards])
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.search_index = faiss.IndexFlatL2(dimension)
            self.search_index.add(embeddings)
            
            logger.info(f"Search index initialized with {len(self.card_ids)} cards")
            
        except Exception as e:
            logger.error(f"Error initializing search index: {str(e)}")
            self.search_index = None
            self.card_ids = []
    
    def search_cards(self, regex_pattern=None, query_text=None, price_limit=None, max_results=50):
        """Search for cards using regex pattern and/or semantic search"""
        try:
            # Start with a base query
            card_query = self.supabase.table("mtg_cards").select("*")
            
            # Apply regex filter if provided
            if regex_pattern:
                # Note: This is a simplification. In a real application, you'd need to
                # handle regex in a more sophisticated way, possibly with a custom function
                card_query = card_query.ilike("oracle_text", f"%{regex_pattern}%")
            
            # Apply price limit if provided
            if price_limit is not None and price_limit > 0:
                card_query = card_query.lte("prices_usd", price_limit)
            
            # Execute the query
            response = card_query.execute()
            
            if not response.data:
                return []
            
            matched_cards = response.data
            
            # If query_text is provided, perform semantic search
            if query_text and self.search_index:
                # Get subset of cards that matched the filters
                filtered_ids = [card['id'] for card in matched_cards]
                
                # Create a mapping from ID to index
                id_to_index = {id_val: idx for idx, id_val in enumerate(filtered_ids)}
                
                # Extract embeddings for filtered cards
                filtered_embeddings = np.array([card['embedding'] for card in matched_cards])
                
                # Create temporary index for filtered cards
                temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
                temp_index.add(filtered_embeddings)
                
                # Encode the query text
                query_embedding = self.embed_model.encode([query_text])[0].reshape(1, -1)
                
                # Search
                _, indices = temp_index.search(query_embedding, min(max_results, len(filtered_ids)))
                
                # Get the top matching cards
                result_cards = [matched_cards[idx] for idx in indices[0]]
                
                return result_cards
            else:
                # Just return the filtered cards
                return matched_cards[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching cards: {str(e)}")
            return []
    
    def generate_deck(self, strategy_description, matched_cards, price_limit=None):
        """Generate a deck based on strategy and matched cards"""
        try:
            # Format the matched cards for the prompt
            card_list = "\n".join([
                f"{card.get('name', 'Unknown')} - {card.get('oracle_text', '')[:100]}..."
                for card in matched_cards[:20]  # Limit to 20 cards for reasonable prompt size
            ])
            
            price_constraint = f"The total deck price should be under ${price_limit}." if price_limit else ""
            
            # Create the prompt
            prompt = f"""Generate a Magic: The Gathering commander deck based on this strategy:
Strategy: {strategy_description}

Include these cards if appropriate (you don't have to use all):
{card_list}

{price_constraint}

Output the deck in JSON format with the following structure:
{{
  "commander": "Commander Name",
  "strategy": "Detailed strategy explanation",
  "cards": {{
    "commander": ["Card Name"],
    "creatures": ["Card Name 1", "Card Name 2", ...],
    "instants": ["Card Name 1", "Card Name 2", ...],
    "sorceries": ["Card Name 1", "Card Name 2", ...],
    "artifacts": ["Card Name 1", "Card Name 2", ...],
    "enchantments": ["Card Name 1", "Card Name 2", ...],
    "planeswalkers": ["Card Name 1", "Card Name 2", ...],
    "lands": ["Card Name 1", "Card Name 2", ...]
  }},
  "total_price": estimated_price_in_dollars
}}
"""
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Magic: The Gathering deck builder."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract the response text
            deck_text = response.choices[0].message.content
            
            # Try to parse the JSON
            try:
                # Extract JSON from the response
                match = re.search(r'({.*})', deck_text.replace('\n', ' '), re.DOTALL)
                if match:
                    deck_json = json.loads(match.group(1))
                else:
                    deck_json = json.loads(deck_text)
                
                return deck_json
            except json.JSONDecodeError:
                logger.error("Failed to parse deck JSON")
                return {
                    "error": "Failed to parse generated deck",
                    "raw_response": deck_text
                }
            
        except Exception as e:
            logger.error(f"Error generating deck: {str(e)}")
            return {"error": str(e)}
    
    def save_deck(self, deck_data, user_id=None):
        """Save a generated deck to the database"""
        try:
            # Prepare deck data for storage
            storage_data = {
                "commander": deck_data.get("commander", "Unknown Commander"),
                "strategy": deck_data.get("strategy", ""),
                "decklist": json.dumps(deck_data),
                "user_id": user_id
            }
            
            # Insert into database
            result = self.supabase.table("saved_decks").insert(storage_data).execute()
            
            if result.data:
                deck_id = result.data[0]['id']
                return {
                    "success": True,
                    "deck_id": deck_id,
                    "deck_url": f"/decks/{deck_id}"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to save deck"
                }
            
        except Exception as e:
            logger.error(f"Error saving deck: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_deck_request(self, request_data, user_id=None):
        """Process a complete deck generation request"""
        try:
            # Extract request parameters
            regex_pattern = request_data.get('regex_pattern', '')
            query_text = request_data.get('query_text', '')
            price_limit = float(request_data.get('price_limit', 0))
            strategy_description = request_data.get('strategy_description', '')
            
            # Step 1: Search for matching cards
            matched_cards = self.search_cards(regex_pattern, query_text, price_limit)
            
            if not matched_cards:
                return {
                    "success": False,
                    "error": "No cards found matching your criteria"
                }
            
            # Step 2: Generate the deck
            deck_result = self.generate_deck(strategy_description, matched_cards, price_limit)
            
            if 'error' in deck_result:
                return {
                    "success": False,
                    "error": deck_result['error']
                }
            
            # Step 3: Save the deck
            storage_result = self.save_deck(deck_result, user_id)
            
            if not storage_result['success']:
                return {
                    "success": False,
                    "error": storage_result['error']
                }
            
            # Step 4: Return the complete result
            return {
                "success": True,
                "commander": deck_result.get('commander', 'Unknown Commander'),
                "strategy": strategy_description,
                "price_limit": price_limit,
                "deck_url": storage_result['deck_url'],
                "deck_details": deck_result
            }
            
        except Exception as e:
            logger.error(f"Error processing deck request: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }