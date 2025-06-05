from flask import jsonify, request
from flask_cors import CORS
import json
import logging
import re
import os
import numpy as np
import requests
import faiss
from src.rag_system import MTGRetrievalSystem
from src.api_server import app
from functools import lru_cache
from src.config import DATA_DIR
from typing import List, Dict, Any, Optional # Added Optional
import scipy.spatial.distance as distance # This import was present but not obviously used; kept for now.
from src.rag_instance import rag_system
# from src.universal_search import UniversalSearchHandler # Not used directly, Enhanced is used.
from src.enhanced_universal_search import EnhancedUniversalSearchHandler
import ast # For ast.literal_eval

# Configure logging
logger = logging.getLogger("UpdateAPIServer")
# The enhanced_search_handler is initialized in enhanced_universal_search.py and passed to endpoints if needed.
# This global instance might be for other uses or can be removed if all universal search goes via the endpoint.
# For now, keeping it as it was in the original structure.
enhanced_search_handler_global_instance = EnhancedUniversalSearchHandler(rag_system)


# Initialize rules cache
rules_cache = {
    'loaded': False,
    'rules_data': [],
    'rules_embeddings': None # Should store np.ndarray if loaded
}

# Initialize mechanics mapping (for synergy detection)
mechanics_cache = {
    'loaded': False,
    'mechanics_map': {},  # Maps mechanics to rules
    'card_mechanics_map': {}  # Maps cards to their mechanics
}

# Enable CORS
CORS(app)

# Helper functions
def validate_colors(colors_str: Optional[str]) -> Optional[List[str]]:
    """Validate and normalize color parameters"""
    if not colors_str:
        return None
    valid_colors = ['W', 'U', 'B', 'R', 'G', 'C']
    colors = colors_str.split(',')
    return [c.upper() for c in colors if c.upper() in valid_colors]

def validate_format(format_str: Optional[str]) -> Optional[str]:
    """Validate format parameter"""
    if not format_str:
        return None
    valid_formats = ['standard', 'pioneer', 'modern', 'legacy', 'vintage', 'commander', 'brawl', 'historic', 'pauper']
    return format_str.lower() if format_str.lower() in valid_formats else None

def check_color_match(card_colors: List[str], filter_colors: List[str]) -> bool:
    """Check if a card's colors match the filter colors - FIXED FOR COLORLESS CARDS"""
    if not card_colors:  # Empty color_identity = colorless card
        if 'C' in filter_colors: # Only match if 'C' is explicitly asked for colorless
            return True
        # Or if no specific color filter is applied, colorless matches.
        # If filter_colors is not empty and does not contain 'C', colorless should not match.
        # The original logic: return True (colorless can be played in any deck)
        # Let's refine: if filter_colors is present and doesn't include 'C', then a colorless card (no card_colors) should not match unless filter_colors is empty.
        # However, for deck building, colorless is always an option if not explicitly excluded.
        # For filtering based on "show me X color cards", colorless should only show for "C".
        # For "cards I can play in my X color deck", colorless is fine.
        # Current universal search filter logic might be better. This one is for specific API filters.
        return True # Original broad interpretation for deck inclusion.

    if 'C' in filter_colors and not card_colors: # Explicitly asking for colorless and card is colorless
        return True
    # If asking for specific colors (and not 'C'), card must have at least one of those.
    return any(color in filter_colors for color in card_colors)


def check_for_excluded_terms(card: Dict[str, Any], exclude_terms: List[str]) -> bool:
    """Check if card contains any excluded terms"""
    if not exclude_terms:
        return False
    card_name = card.get('name', '').lower()
    oracle_text = card.get('oracle_text', '').lower()
    for term in exclude_terms:
        term_lower = term.strip().lower() # Renamed for clarity
        if term_lower and (
            re.search(r'\b' + re.escape(term_lower) + r'\b', card_name) or
            re.search(r'\b' + re.escape(term_lower) + r'\b', oracle_text)
        ):
            return True
    return False

def download_rules(url: str = "https://media.wizards.com/2025/downloads/MagicCompRules%2020250404.txt") -> Optional[str]:
    """Download the latest MTG comprehensive rules"""
    try:
        logger.info(f"Attempting to download rules from {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.text
        if len(content) < 1000:  # Adjusted size check for robustness
            logger.error(f"Downloaded content too small ({len(content)} bytes). Might not be valid rules.")
            return None
        logger.info(f"Successfully downloaded {len(content)} bytes of rules content")
        return content
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading rules from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error downloading rules from {url}: {e}")
        return None

def parse_rules(rules_text: str) -> List[Dict[str, Any]]:
    """Parse the rules text into searchable chunks"""
    rules_data = []
    # Split the document into sections more robustly
    # Regex to find a rule number like "1.", "100.", "100.1.", "100.1a."
    # This improved split handles the preamble before the first rule.
    rule_sections = re.split(r'\n(?=\d{1,3}(?:\.\d+)?[a-z]?\.\s)', rules_text)

    # The first part might be a preamble or empty if the text starts directly with a rule.
    # We are interested in sections starting with a rule number.
    current_section_title = "General" # Default for rules before a new section title is found
    section_counter = 0

    for rule_block in rule_sections:
        if not rule_block.strip():
            continue

        # Check for a section header like "Credits", "Glossary", "Oracle Changes" etc. or "1. Game Concepts"
        # This regex looks for lines that are likely section titles (all caps or Title Case, or starting with a number)
        section_header_match = re.match(r'^(?:[A-Z\s]+|[A-Z][a-zÀ-ÖØ-öø-ÿ\s\-]+(?:\s(?:of|the|a|an)\s[A-Z][a-zÀ-ÖØ-öø-ÿ\s\-]+)*|\d+\.\s+[A-Z][a-zÀ-ÖØ-öø-ÿ\s\-]+)\s*$', rule_block.split('\n')[0])

        block_lines = rule_block.strip().split('\n')
        first_line = block_lines[0].strip()

        # Heuristic for section titles (often all caps or title case, and short)
        if (first_line.isupper() and len(first_line.split()) < 7) or \
           (first_line.istitle() and len(first_line.split()) < 7 and not re.match(r'^\d+\.\d+[a-z]?\s', first_line)) or \
           (re.match(r'^\d+\.\s+[A-Z]', first_line)): # e.g., "1. Game Concepts"
            current_section_title = first_line
            section_counter +=1 # Using a simple counter for section ID
            # Remove title line if it's not also a rule itself
            if not re.match(r'^\d+\.\d+[a-z]?\s', first_line): # if it's not like "100.1. ..."
                 rule_text_for_section = "\n".join(block_lines[1:])
            else:
                 rule_text_for_section = "\n".join(block_lines)

        else:
            rule_text_for_section = rule_block.strip()

        # Extract individual rules using the original regex, but on the current rule_text_for_section
        # This pattern finds rule numbers like 100.1 or 100.1a followed by the rule text.
        # It handles multi-line rule text where subsequent lines are indented.
        rules_in_block = re.findall(r'(\d+\.\d+[a-z]?)\s+((?:[^\n]*(?:\n(?!\d+\.\d+[a-z]?\s)[^\n]*)*))', rule_text_for_section)

        for rule_num_str, rule_text_content in rules_in_block: # Renamed for clarity
            cleaned_rule_text = rule_text_content.replace('\n', ' ').strip()
            if cleaned_rule_text: # Ensure there's text
                rules_data.append({
                    'section': section_counter, # Using the simple counter
                    'title': current_section_title, # Title of the current major section
                    'rule_number': rule_num_str,
                    'text': cleaned_rule_text,
                    'id': f"rule_{rule_num_str.replace('.', '_').replace(' ', '_')}" # Ensure ID is valid
                })
    if not rules_data and rules_text: # Fallback if main parsing yields nothing but text exists
        logger.warning("Advanced rule parsing yielded no results, trying simpler full text split.")
        # Simpler fallback for very weird formats: split by rule numbers directly
        # This is a very basic fallback and might not be robust.
        raw_rules = re.findall(r'((\d+\.\d+[a-z]?)\s+([^\n]+(?:\n\s+[^\n]+)*))', rules_text)
        for full_match, rule_num_str, rule_text_content in raw_rules:
            rules_data.append({
                'section': 0, 'title': "General Fallback", 'rule_number': rule_num_str,
                'text': rule_text_content.replace('\n', ' ').strip(),
                'id': f"rule_{rule_num_str.replace('.', '_')}"
            })

    logger.info(f"Parsed {len(rules_data)} rules.")
    return rules_data


def generate_rules_embeddings(rules_data: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Generate embeddings for rules text"""
    if not rag_system.embedding_available:
        logger.warning("Embedding model not available for generating rule embeddings.")
        return None
    if not rules_data:
        logger.warning("No rules data provided to generate_rules_embeddings.")
        return np.array([]) # Return empty array for consistency

    texts = [f"Rule {rule.get('rule_number', '')}: {rule.get('text', '')}" for rule in rules_data] # Added "Rule" prefix for clarity
    batch_size = 32  # As defined in original
    all_embeddings_list = [] # Renamed

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:min(i + batch_size, len(texts))] # Renamed
        try:
            batch_embeddings_np = rag_system.embedding_model.encode(batch_texts) # Renamed
            all_embeddings_list.append(batch_embeddings_np)
        except Exception as e:
            logger.error(f"Error encoding batch of rule texts (index {i}): {e}")
            # Optionally, continue to next batch or return None/partial
            # For now, let's continue and log, result might be partial
    
    if all_embeddings_list:
        try:
            return np.vstack(all_embeddings_list)
        except ValueError as e_vstack: # Handle cases where vstack might fail (e.g. inconsistent shapes if errors occurred)
            logger.error(f"Error during np.vstack of rule embeddings, possibly due to errors in batch encoding: {e_vstack}")
            # Attempt to filter out None or problematic embeddings if any crept in
            valid_batches = [b for b in all_embeddings_list if isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[0] > 0]
            if valid_batches:
                # Check for consistent dimensions (number of columns)
                first_dim = valid_batches[0].shape[1]
                if all(b.shape[1] == first_dim for b in valid_batches):
                    logger.info(f"Attempting vstack with {len(valid_batches)} valid batches after filtering.")
                    return np.vstack(valid_batches)
            return None # Could not form a valid combined embedding matrix
    else:
        logger.warning("No embeddings were generated for rules.")
        return np.array([])


def create_rules_table_if_not_exists() -> bool:
    """Verify the rules table exists in the database"""
    try:
        if not rag_system.db.is_connected:
            logger.warning("Database not connected. Cannot access rules table.")
            return False
        # Just try a simple select to verify the table exists
        try:
            # Check if table exists (Supabase specific or general SQL)
            # This is a Supabase client way:
            rag_system.db.client.from_("mtg_rules").select("id").limit(1).execute()
            logger.info("Table 'mtg_rules' exists.")
            return True
        except Exception as e: # Catch Supabase/PostgREST specific error for non-existent table
            logger.warning(f"'mtg_rules' table might not exist or access denied: {e}")
            # Here you might try to CREATE TABLE if it doesn't exist, if permissions allow
            # For now, just reports it doesn't exist.
            return False
    except Exception as e:
        logger.error(f"Error checking 'mtg_rules' table: {e}")
        return False

def import_rules_to_database(rules_data: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None) -> bool:
    """Import rules data to the database"""
    if not rag_system.db.is_connected:
        logger.error("Database not connected. Cannot import rules.")
        return False
    try:
        if not create_rules_table_if_not_exists():
            # Optionally, attempt to create the table here if it's confirmed not to exist.
            # For now, we assume it must exist or be creatable by other means.
            logger.error("Rules table 'mtg_rules' does not exist, and creation was not attempted. Cannot import rules.")
            return False
        
        # Clear existing rules (be cautious with this in production)
        try:
            # Delete all rows. If 'id' is not suitable for neq, use a different condition or .delete().match({})
            # For safety, one might prefer to delete based on a source/version field if re-importing specific sets of rules.
            delete_response = rag_system.db.client.from_("mtg_rules").delete().gte("section", -1).execute() # Deletes all rows essentially
            logger.info(f"Cleared existing rules from 'mtg_rules'. Response: {delete_response}")
        except Exception as e_delete:
            logger.error(f"Error clearing rules from 'mtg_rules': {e_delete}")
            # Decide if to proceed if clearing fails. For now, we'll proceed.

        batch_size = 50  # As in original
        total_rules = len(rules_data)
        success_count = 0
        
        for i in range(0, total_rules, batch_size):
            batch_end_idx = min(i + batch_size, total_rules) # Renamed
            current_batch_rules = rules_data[i:batch_end_idx] # Renamed
            batch_items_to_insert = [] # Renamed
            
            for j_idx, rule_item_data in enumerate(current_batch_rules): # Renamed
                db_rule_item = { # Renamed
                    "rule_number": rule_item_data["rule_number"],
                    "section": rule_item_data.get("section"), # Use .get for safety
                    "title": rule_item_data.get("title"),
                    "text": rule_item_data["text"]
                    # Assuming 'id' is auto-generated by DB or not explicitly inserted here
                }
                
                if embeddings is not None and (i + j_idx) < embeddings.shape[0]:
                    # Ensure embedding is a list for Supabase JSON/vector storage
                    db_rule_item["embedding"] = embeddings[i + j_idx].tolist()
                
                batch_items_to_insert.append(db_rule_item)
            
            if batch_items_to_insert:
                try:
                    insert_response = rag_system.db.client.from_("mtg_rules").insert(batch_items_to_insert).execute() # Renamed
                    # Supabase insert response might not directly give count of inserted rows in data.
                    # Check for errors in insert_response if possible
                    if not (hasattr(insert_response, 'error') and insert_response.error):
                        success_count += len(batch_items_to_insert)
                        logger.info(f"Imported {success_count}/{total_rules} rules into 'mtg_rules'. Batch {i//batch_size + 1}.")
                    else:
                        logger.error(f"Error inserting batch of rules (index {i}): {getattr(insert_response, 'error', 'Unknown Supabase error')}")

                except Exception as e_insert_batch:
                    logger.error(f"Exception during batch insert of rules (index {i}): {e_insert_batch}")
        
        logger.info(f"Successfully imported a total of {success_count} rules to 'mtg_rules' table.")
        return success_count > 0 # Or success_count == total_rules for stricter success
    except Exception as e:
        logger.error(f"General error importing rules to database: {e}")
        return False

def load_rules_from_database() -> bool:
    """Load rules data from the database into cache"""
    global rules_cache # Ensure we are modifying the global cache
    
    try:
        if not rag_system.db.is_connected:
            logger.warning("Database not connected. Cannot load rules.")
            return False
        
        # Query all rules
        response = rag_system.db.client.from_("mtg_rules").select("id, rule_number, section, title, text, embedding").execute() # Select embedding too
        
        if not response.data:
            logger.warning("No rules found in 'mtg_rules' database table.")
            rules_cache["rules_data"] = []
            rules_cache["rules_embeddings"] = None # Or np.array([])
            rules_cache["loaded"] = True # Loaded, but empty
            return True # Successfully "loaded" an empty set
        
        db_rules_data = [] # Renamed
        db_rule_embeddings_list = [] # Renamed for clarity

        for db_rule_item in response.data: # Renamed
            # Reconstruct the cache format
            cache_rule_item = { # Renamed
                "id": db_rule_item.get("id", f"rule_{db_rule_item['rule_number'].replace('.', '_')}") if db_rule_item.get('id') else f"rule_{db_rule_item['rule_number'].replace('.', '_')}", # Prefer DB id if present
                "rule_number": db_rule_item["rule_number"],
                "section": db_rule_item["section"],
                "title": db_rule_item["title"],
                "text": db_rule_item["text"]
            }
            db_rules_data.append(cache_rule_item)
            
            # Parse embedding if present
            embedding_from_db = db_rule_item.get("embedding") # Renamed
            if embedding_from_db:
                parsed_emb = parse_embedding_string(embedding_from_db) # Use robust parser
                if parsed_emb is not None:
                    db_rule_embeddings_list.append(parsed_emb)
                else: # If parsing fails, append a placeholder or handle inconsistency
                    logger.warning(f"Failed to parse embedding for rule {db_rule_item['rule_number']} from database.")
                    # To maintain alignment, you might append a zero vector of correct dimension or None
                    # For now, this might lead to shorter embeddings list than rules_data list if parsing fails.
                    # Best to ensure embeddings are stored correctly or handle this case.
                    # If a rule has no embedding, db_rule_embeddings_list will be shorter.
                    # This requires careful handling when using rules_data and rules_embeddings together.
                    # For simplicity, if an embedding fails to parse, we might skip adding to db_rule_embeddings_list,
                    # or add a specific marker/None. Let's assume for now that if present, it should parse.

        rules_cache["rules_data"] = db_rules_data
        if db_rule_embeddings_list:
            try:
                # Ensure all embeddings have the same shape before vstack
                if len(set(emb.shape for emb in db_rule_embeddings_list if emb is not None and emb.ndim > 0)) <= 1: # Allows for one shape or empty after Nones
                    valid_embs_for_vstack = [emb for emb in db_rule_embeddings_list if emb is not None and emb.ndim > 0]
                    if valid_embs_for_vstack:
                         rules_cache["rules_embeddings"] = np.vstack(valid_embs_for_vstack)
                         logger.info(f"Loaded {rules_cache['rules_embeddings'].shape[0]} rule embeddings from database.")
                    else:
                        rules_cache["rules_embeddings"] = np.array([])
                        logger.info("No valid rule embeddings loaded from database.")
                else:
                    logger.error("Rule embeddings from database have inconsistent shapes. Cannot stack.")
                    rules_cache["rules_embeddings"] = None # Indicate problem
            except ValueError as e_vstack:
                 logger.error(f"Error stacking rule embeddings from database: {e_vstack}")
                 rules_cache["rules_embeddings"] = None
        else:
            rules_cache["rules_embeddings"] = np.array([]) # No embeddings found/parsed
            logger.info("No rule embeddings found or parsed from database.")

        rules_cache["loaded"] = True
        logger.info(f"Loaded {len(db_rules_data)} rules from 'mtg_rules' database table.")
        return True
    except Exception as e:
        logger.error(f"Error loading rules from database: {e}")
        rules_cache["loaded"] = False # Ensure cache reflects load failure
        return False

def load_rules_data() -> List[Dict[str, Any]]: # Return type is list of rules
    """Load rules data from database or file, or download if needed. Populates cache."""
    # Attempt to load from database first
    if load_rules_from_database(): # This function now populates the cache directly
        logger.info("Rules successfully loaded from database into cache.")
        return rules_cache["rules_data"]
    
    # If database loading failed or was skipped, try local file
    logger.warning("Failed to load rules from database, trying local file.")
    rules_json_path = os.path.join(DATA_DIR, 'processed', 'rules_data.json') # Renamed
    
    if os.path.exists(rules_json_path):
        try:
            with open(rules_json_path, 'r', encoding='utf-8') as f_json: # Renamed
                file_rules_data = json.load(f_json) # Renamed
                rules_cache["rules_data"] = file_rules_data
                # Embeddings are not typically stored in rules_data.json; they'd be generated or loaded separately
                rules_cache["rules_embeddings"] = None # Indicate embeddings need generation if from file
                rules_cache["loaded"] = True
                logger.info(f"Loaded {len(file_rules_data)} rules from JSON file: {rules_json_path}")
                # If loaded from file, consider generating/loading embeddings if needed by application flow
                return file_rules_data
        except Exception as e_file_load:
            logger.error(f"Error loading rules from JSON file {rules_json_path}: {e_file_load}")
    
    # Last resort - download and process rules if not loaded from DB or file
    logger.warning("Failed to load rules from local file, attempting to download and process.")
    rules_text_content = download_rules() # Renamed
    if not rules_text_content:
        logger.error("Failed to download rules text. Cannot load rules.")
        rules_cache["rules_data"] = [] # Ensure cache is empty
        rules_cache["loaded"] = False
        return []
    
    downloaded_rules_data = parse_rules(rules_text_content) # Renamed
    
    # Save parsed rules data to JSON file as backup for next time
    try:
        os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
        with open(rules_json_path, 'w', encoding='utf-8') as f_json_save: # Renamed
            json.dump(downloaded_rules_data, f_json_save, indent=2)
            logger.info(f"Saved newly parsed rules to JSON file: {rules_json_path}")
    except Exception as e_file_save:
        logger.error(f"Error saving newly parsed rules to JSON file {rules_json_path}: {e_file_save}")
    
    # Try to import these newly parsed rules to database (includes generating embeddings)
    if rag_system.db.is_connected:
        logger.info("Attempting to import newly downloaded and parsed rules to database.")
        generated_embeddings = None
        if rag_system.embedding_available:
            generated_embeddings = generate_rules_embeddings(downloaded_rules_data)
        
        import_success = import_rules_to_database(downloaded_rules_data, generated_embeddings)
        if import_success:
            logger.info("Successfully imported newly parsed rules into database.")
            rules_cache["rules_embeddings"] = generated_embeddings # Cache the generated embeddings
        else:
            logger.error("Failed to import newly parsed rules into database.")
            rules_cache["rules_embeddings"] = None # Ensure embeddings cache is cleared if import failed
    else: # DB not connected, embeddings won't be saved to DB, cache what we generated
        if rag_system.embedding_available and not rules_cache.get("rules_embeddings"): # Only if not already set (e.g. by failed DB import)
            rules_cache["rules_embeddings"] = generate_rules_embeddings(downloaded_rules_data)

    rules_cache["rules_data"] = downloaded_rules_data
    rules_cache["loaded"] = True # Loaded, possibly without DB persistence if that failed
    return downloaded_rules_data


def update_rules_embeddings(url: Optional[str] = None) -> bool: # url is optional
    """Update rules data and embeddings, using local file first, then optionally URL."""
    rules_text_content = None
    rules_txt_path = os.path.join(DATA_DIR, 'processed', 'MTGCompRules.txt') # Renamed
    
    if os.path.exists(rules_txt_path):
        try:
            with open(rules_txt_path, 'r', encoding='utf-8') as f_txt: # Renamed
                rules_text_content = f_txt.read()
                logger.info(f"Loaded rules from local TXT file ({len(rules_text_content)} bytes): {rules_txt_path}")
        except Exception as e_load_txt:
            logger.error(f"Error loading local rules TXT file {rules_txt_path}: {e_load_txt}")
            rules_text_content = None # Ensure it's None if loading fails

    if not rules_text_content and url: # If local file failed or not present, and URL is provided
        logger.info(f"Local rules TXT not found or failed to load. Attempting download from URL: {url}")
        rules_text_content = download_rules(url)
        if rules_text_content: # Save downloaded content to local TXT file for next time
            try:
                os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
                with open(rules_txt_path, 'w', encoding='utf-8') as f_txt_save: # Renamed
                    f_txt_save.write(rules_text_content)
                    logger.info(f"Saved downloaded rules to local TXT file: {rules_txt_path}")
            except Exception as e_save_txt:
                logger.error(f"Error saving downloaded rules to local TXT file: {e_save_txt}")
    
    if not rules_text_content:
        logger.error("Failed to obtain rules text from local file or URL. Cannot update rules embeddings.")
        return False
    
    # Parse rules
    parsed_rules_data = parse_rules(rules_text_content) # Renamed
    if not parsed_rules_data:
        logger.error("Parsing rules text yielded no data. Cannot update embeddings.")
        return False

    # Generate embeddings
    generated_embeddings_np = None # Renamed
    if rag_system.embedding_available:
        generated_embeddings_np = generate_rules_embeddings(parsed_rules_data)
    
    # Update the database
    import_db_success = False # Renamed
    if rag_system.db.is_connected: # Only attempt DB import if connected
        import_db_success = import_rules_to_database(parsed_rules_data, generated_embeddings_np)
    else:
        logger.warning("Database not connected. Skipping import of rules to database.")
        # Even if DB not connected, we can update the cache if rules were parsed.
        import_db_success = True # Consider it a "success" in terms of getting data, just not DB stored.

    # Update cache
    global rules_cache # Ensure modification of global
    rules_cache["rules_data"] = parsed_rules_data
    rules_cache["rules_embeddings"] = generated_embeddings_np # Cache the newly generated embeddings
    rules_cache["loaded"] = True # Cache is now considered loaded with this data
    
    logger.info(f"Rules cache updated. Rules count: {len(parsed_rules_data)}. Embeddings generated: {'Yes' if generated_embeddings_np is not None else 'No'}.")
    return import_db_success # Success depends on DB import if it was attempted.


# --- Robust Embedding Parsing Function ---
_parse_embedding_logger = logging.getLogger("UpdateAPIServer.ParseEmbedding")

def parse_embedding_string(embedding_input: Any) -> Optional[np.ndarray]:
    """
    Robustly parses various embedding inputs (string, list, tuple, np.ndarray)
    into a NumPy float32 array.
    """
    if isinstance(embedding_input, np.ndarray):
        return embedding_input.astype(np.float32) # Ensure correct dtype if already ndarray

    if isinstance(embedding_input, (list, tuple)):
        try:
            if not embedding_input:
                _parse_embedding_logger.debug("Input is an empty list/tuple.")
                return np.array([], dtype=np.float32) # Consistent return for empty
            
            # Check if all items are already numbers (int/float)
            if all(isinstance(x, (int, float)) for x in embedding_input):
                return np.array(embedding_input, dtype=np.float32)
            # Check if all items are strings that can be converted to float
            elif all(isinstance(x, str) for x in embedding_input):
                # Filter out empty strings that might result from bad splits if this list came from a string
                numeric_items_str = [item for item in embedding_input if item.strip()]
                if not numeric_items_str and embedding_input: # Had items, but all were whitespace strings
                     _parse_embedding_logger.warning(f"List/tuple contained only whitespace strings: {embedding_input}")
                     return None # Or np.array([], dtype=np.float32)
                return np.array([float(x) for x in numeric_items_str], dtype=np.float32)
            else:
                _parse_embedding_logger.warning(f"Input list/tuple contains mixed/non-numeric/non-string types. Example item type: {type(embedding_input[0])}")
                return None
        except ValueError as e_val:
            _parse_embedding_logger.warning(f"Could not convert list/tuple to float array: {str(embedding_input)[:200]}. Error: {e_val}")
            return None
        except Exception as e_gen_list: # Catch any other unexpected errors during list processing
            _parse_embedding_logger.error(f"Unexpected error processing list/tuple embedding {str(embedding_input)[:200]}: {e_gen_list}")
            return None

    if isinstance(embedding_input, str):
        cleaned_str = embedding_input.strip()
        if not cleaned_str:
            _parse_embedding_logger.debug("Input is an empty string.")
            return np.array([], dtype=np.float32)
        try:
            # ast.literal_eval is safer for evaluating string representations of Python literals
            evaluated_data = ast.literal_eval(cleaned_str)

            # After ast.literal_eval, data could be list, tuple, int, float, etc.
            # We need to recursively call parse_embedding_string in case ast_literal_eval itself returns a list of strings
            # or a single string number.
            if isinstance(evaluated_data, (list, tuple, str, int, float)): # Check if type is something parse_embedding_string can handle
                return parse_embedding_string(evaluated_data) # Recursive call to handle nested structures or convert single items
            else:
                 _parse_embedding_logger.warning(f"ast.literal_eval on string '{cleaned_str}' resulted in an unsupported type: {type(evaluated_data)}")
                 return None

        except (ValueError, SyntaxError, TypeError) as e_ast:
            # Fallback for strings not parsable by ast.literal_eval (e.g., "1.0,2.0,3.0" without brackets)
            _parse_embedding_logger.info(f"ast.literal_eval failed for string '{cleaned_str}': {e_ast}. Attempting simple comma split.")
            try:
                values_from_split_str = [val.strip() for val in cleaned_str.split(',') if val.strip()] # Renamed
                if not values_from_split_str:
                    _parse_embedding_logger.warning(f"Simple comma split of '{cleaned_str}' yielded no values.")
                    return np.array([], dtype=np.float32)
                
                float_values_list = [float(val) for val in values_from_split_str] # Renamed
                return np.array(float_values_list, dtype=np.float32)
            except ValueError as e_split_val: # Renamed
                 _parse_embedding_logger.error(f"Fallback comma split also failed for '{cleaned_str}': {e_split_val}")
                 return None
            except Exception as e_split_general: # Renamed and more general
                 _parse_embedding_logger.error(f"Unexpected error during fallback comma split for '{cleaned_str}': {e_split_general}")
                 return None
    
    _parse_embedding_logger.warning(f"Input type '{type(embedding_input)}' is not supported for embedding parsing. Value: {str(embedding_input)[:100]}")
    return None
# --- End of Robust Embedding Parsing Function ---


# --- Synergy Detection Functions (using robust parsing) ---
def extract_key_mechanics(card_data: Dict[str, Any]) -> List[str]:
    """Extract key game mechanics from card text"""
    mechanics_set = set() # Renamed
    oracle_text = card_data.get('oracle_text', '')
    if not oracle_text:
        return list(mechanics_set)
    
    # Define mechanic patterns (condensed for brevity, ensure these are effective)
    mechanic_patterns = {
        'replacement_effect': r'instead of|would|instead|skip|enters with|enters as', 'triggered_ability': r'when|whenever|at the beginning',
        'trample': r'\btrample\b', 'flying': r'\bflying\b', 'first_strike': r'\bfirst strike\b', 'double_strike': r'\bdouble strike\b',
        'deathtouch': r'\bdeathtouch\b', 'lifelink': r'\blifelink\b', 'token': r'\btoken\b|create[s]* (?:a|an|two|three|\d+)?',
        'sacrifice': r'\bsacrifice\b', 'counter_spell': r'\bcounter target\b|\bcounterspell\b', # Renamed to avoid confusion with +1/+1 counters
        'counters_on_card': r'\b\+1/\+1 counter|\b\-1/\-1 counter|counter on it', # Renamed
        'discard': r'\bdiscard\b', 'graveyard_interaction': r'from (?:a|the|your) graveyard|return[s]* to|from graveyard', # Renamed
        'exile': r'\bexile\b', 'tutor': r'search your library', 'draw_card': r'draw[s]* (?:a|an|\d+)? card[s]*', # Renamed
        'ramp': r'search your library for a[n]* land|additional land|add[s]* mana', 'removal': r'destroy target|exile target|damage to target'
    }
    for mechanic_key, pattern_str in mechanic_patterns.items(): # Renamed
        if re.search(pattern_str, oracle_text, re.IGNORECASE):
            mechanics_set.add(mechanic_key)
    
    type_line = card_data.get('type_line', '').lower()
    if 'creature' in type_line: mechanics_set.add('creature_type') # Renamed for clarity
    if 'artifact' in type_line: mechanics_set.add('artifact_type')
    if 'enchantment' in type_line: mechanics_set.add('enchantment_type')
    if 'planeswalker' in type_line: mechanics_set.add('planeswalker_type')
    if 'land' in type_line: mechanics_set.add('land_type')
    
    return list(mechanics_set)


def initialize_mechanics_mappings() -> bool:
    """Initialize mappings between mechanics, rules, and cards. Populates cache."""
    global mechanics_cache # Ensure global cache modification
    
    if mechanics_cache['loaded']:
        return True
    try:
        # First ensure rules are loaded into cache
        if not rules_cache["loaded"] or not rules_cache["rules_data"]: # Check if data is actually there
            logger.info("Rules not loaded in cache for mechanics mapping, attempting to load.")
            load_rules_data() # This populates rules_cache
            if not rules_cache["rules_data"]: # Still no data after load attempt
                logger.error("Failed to load rules data for mechanics mapping.")
                return False
        
        new_mechanics_map = {} # Renamed
        # Keywords for mechanics (ensure these align with extract_key_mechanics output)
        mechanic_keywords_for_rules = {
            # Keyword Abilities (often in Rule 702)
            'trample': [
                r'\b702\.19\b',  # Rule number for Trample
                r'\bTrample\b is a static ability',
                r'\bassign damage .* Trample\b'
            ],
            'flying': [
                r'\b702\.9\b',   # Rule number for Flying
                r'\bFlying\b is a static ability',
                r'can block .* Flying\b',
                r'can be blocked .* Flying\b'
            ],
            'first_strike': [
                r'\b702\.7\b',   # Rule number for First Strike
                r'\bFirst strike\b is a static ability',
                r'\bcombat damage step preceded by one for first strike\b'
            ],
            'double_strike': [
                r'\b702\.4\b',   # Rule number for Double Strike
                r'\bDouble strike\b is a static ability',
                r'\bdeals damage in both the first and second combat damage steps\b'
            ],
            'deathtouch': [
                r'\b702\.2\b',   # Rule number for Deathtouch
                r'\bDeathtouch\b is a static ability',
                r'\bdamage dealt by a source with deathtouch .* is considered lethal damage\b'
            ],
            'lifelink': [
                r'\b702\.15\b',  # Rule number for Lifelink
                r'\bLifelink\b is a static ability',
                r'\bdamage dealt by a source with lifelink causes that source’s controller .* to gain that much life\b'
            ],

            # Game Actions & Concepts
            'replacement_effect': [
                r'\b614\.\d+[a-z]?\b',  # Rule section 614 for Replacement Effects
                r'\b[Rr]eplacement [Ee]ffects?\b',
                r'\binstead\b', r'\bwould\b.*\bdo\b.*\binstead\b', # Common phrasing
                r'\bskip part of an event\b',
                r'\bas .* enters the battlefield\b'
            ],
            'triggered_ability': [
                r'\b603\.\d+[a-z]?\b',  # Rule section 603 for Triggered Abilities
                r'\b[Tt]riggered [Aa]bilit(y|ies)\b',
                r'\b[Ww]hen\b', r'\b[Ww]henever\b', r'\b[Aa]t the beginning\b', # Core trigger words
                r'\btriggers once\b', r'\btrigger event\b'
            ],
            'token': [
                r'\b111\.\d+[a-z]?\b',  # Rule section 111 for Tokens
                r'\b[Tt]oken\b(?:s)?\b', # "token" or "tokens"
                r'\bcreate[s]? .* token\b',
                r'\bput .* token[s]? onto the battlefield\b',
                r'\btoken that.s a copy\b'
            ],
            'sacrifice': [
                r'\b701\.17\b',  # Rule number for Sacrifice (Keyword Action)
                r'\b[Ss]acrifice\b is a keyword action',
                r'\bto sacrifice a permanent\b',
                r'player sacrifices'
            ],
            'counter_spell': [ # For countering spells/abilities
                r'\b701\.5\b',   # Rule number for Counter (Keyword Action)
                r'\b[Cc]ounter target spell\b',
                r'\b[Cc]ounter target .*ability\b',
                r'\bcountered by game rules\b',
                r'\bspell or ability that.s countered\b'
            ],
            'counters_on_card': [ # For +1/+1, -1/-1, loyalty, poison etc.
                r'\b122\.\d+[a-z]?\b',  # Rule section 122 for Counters
                r'\b[Cc]ounter[s]?\b on a permanent\b', # General phrase
                r'\bput[s]? .* counter[s]? on\b',
                r'\bmove[s]? .* counter[s]? from\b',
                r'\b\+1/\+1 counter[s]?\b',
                r'\b\-1/\-1 counter[s]?\b',
                r'\b[Ll]oyalty counter[s]?\b',
                r'\b[Pp]oison counter[s]?\b',
                r'\b[Cc]harge counter[s]?\b' # Common type
            ],
            'discard': [
                r'\b701\.8\b',   # Rule number for Discard (Keyword Action)
                r'\b[Dd]iscard\b is a keyword action',
                r'\bto discard a card\b',
                r'player discards'
            ],
            'graveyard_interaction': [
                r'\b404\.\d+[a-z]?\b',  # Rule section 404 for Graveyard
                r'\b[Gg]raveyard\b',
                r'\bfrom .* graveyard\b',
                r'\bto the graveyard\b',
                r'\bput[s]? .* into .* graveyard\b',
                r'\breturn .* from .* graveyard to the battlefield\b',
                r'\breturn .* from .* graveyard to .* hand\b',
                r'\bdies\b' # Common shorthand for creature going to graveyard from battlefield (700.4)
            ],
            'exile': [
                r'\b406\.\d+[a-z]?\b',  # Rule section 406 for Exile zone
                r'\b[Ee]xile\b zone\b',
                r'\b[Ee]xile target\b',
                r'\bput[s]? .* into exile\b',
                r'\bexiled card[s]?\b',
                r'\b[Ee]xile .* face down\b'
            ],
            'tutor': [ # Player term for searching library
                r'\b701\.19\b',  # Rule number for Search (Keyword Action)
                r'\b[Ss]earch\b is a keyword action',
                r'\bsearch .* library for a card\b', # Common phrasing
                r'\bsearch that player.s library\b',
                r'\bshuffle .* library\b' # Often follows a search
            ],
            'draw_card': [
                r'\b121\.\d+[a-z]?\b',  # Rule section 121 for Drawing a Card
                r'\b701\.9\b', # Rule for Draw keyword action (though less common)
                r'\b[Dd]raw[s]? a card\b',
                r'\b[Dd]raw[s]? cards\b',
                r'\bplayer who drew a card\b',
                r'\bif a player would draw a card\b' # For replacement effects related to drawing
            ],
            'ramp': [ # Player term for mana acceleration
                r'\b[Aa]dd[s]? (?:one|two|three|\d+|X) mana\b', # Specific mana addition
                r'\b[Aa]dd[s]? {.*?}\b', # E.g. {R}{G}
                r'\bfor each\b.*\badd mana\b',
                r'\bplay an additional land\b', # Rule 305.2b
                r'\bput[s]? .* land .* onto the battlefield\b(?: from .* hand)?', # From hand or library
                r'\b[Mm]ana [Pp]ool\b',
                r'\b605\.\d+[a-z]?\b',  # Rule section 605 for Mana Abilities
                r'\b[Mm]ana [Aa]bilit(y|ies)\b'
            ],
            'removal': [ # Player term for removing threats
                r'\b[Dd]estroy target\b (?:creature|planeswalker|artifact|enchantment|permanent)\b',
                r'\b[Dd]estroy all creatures\b', # Board wipes
                r'\b[Ee]xile target\b (?:creature|planeswalker|artifact|enchantment|permanent)\b', # Already covered by 'exile' but good for specificity
                r'\bdeals? \d+ damage to target creature\b',
                r'\bdeals? \d+ damage to target planeswalker\b',
                r'\bdeals? \d+ damage to any target\b',
                r'\breturn target .* to its owner.s hand\b', # Bounce
                r'\bput target creature .* on top of .* library\b', # Tuck
                r'\bcontroller sacrifices .* creature\b' # Forced sacrifice
            ],

            # Card Types (focus on main rules sections and defining characteristics)
            'creature_type': [
                r'\b302\.\d+[a-z]?\b',  # Rule section 302 for Creatures
                r'\b[Cc]reature type[s]?\b',
                r'\b[Cc]reature card[s]?\b',
                r'\b[Cc]reature spell[s]?\b',
                r'\b[Cc]reature permanent[s]?\b',
                r'\b[Pp]ower and [Tt]oughness\b',
                r'\b[Aa]ttacking creature[s]?\b',
                r'\b[Bb]locking creature[s]?\b',
                r'\b[Dd]eclar(e|ing) attackers step\b', # Relevant to creatures
                r'\b[Dd]eclar(e|ing) blockers step\b'  # Relevant to creatures
            ],
            'artifact_type': [
                r'\b301\.\d+[a-z]?\b',  # Rule section 301 for Artifacts
                r'\b[Aa]rtifact type[s]?\b',
                r'\b[Aa]rtifact card[s]?\b',
                r'\b[Aa]rtifact spell[s]?\b',
                r'\b[Aa]rtifact permanent[s]?\b',
                r'\b[Ee]quipment\b', # Subtype, Rule 301.5
                r'\b[Ff]ortification[s]?\b', # Subtype, Rule 301.6
                r'\b[Vv]ehicle[s]?\b', # Subtype, Rule 301.7
                r'\b[Tt]reasure token[s]?\b' # Common artifact token
            ],
            'enchantment_type': [
                r'\b303\.\d+[a-z]?\b',  # Rule section 303 for Enchantments
                r'\b[Ee]nchantment type[s]?\b',
                r'\b[Ee]nchantment card[s]?\b',
                r'\b[Ee]nchantment spell[s]?\b',
                r'\b[Ee]nchantment permanent[s]?\b',
                r'\b[Aa]ura[s]?\b', # Subtype, Rule 303.4
                r'\b[Cc]urse[s]?\b', # Subtype
                r'\b[Ss]aga[s]?\b'   # Subtype
            ],
            'planeswalker_type': [
                r'\b306\.\d+[a-z]?\b',  # Rule section 306 for Planeswalkers
                r'\b[Pp]laneswalker type[s]?\b', # e.g. "Jace", "Liliana"
                r'\b[Pp]laneswalker card[s]?\b',
                r'\b[Pp]laneswalker spell[s]?\b',
                r'\b[Pp]laneswalker permanent[s]?\b',
                r'\b[Ll]oyalty abilit(y|ies)\b', # Rule 606
                r'\b[Ll]oyalty counter[s]?\b' # Rule 306.5b
            ],
            'land_type': [
                r'\b305\.\d+[a-z]?\b',  # Rule section 305 for Lands
                r'\b[Ll]and type[s]?\b', # e.g. "Forest", "Island"
                r'\b[Ll]and card[s]?\b',
                r'\bplay a land\b',
                r'\b[Bb]asic [Ll]and type[s]?\b', # Rule 205.3i, Rule 305.6
                r'\bmana abilit(y|ies) of lands\b'
            ],
            
            # Additional common mechanics not explicitly in your list but good for future or if extract_key_mechanics expands
            'activated_ability': [ # If you add this key to extract_key_mechanics
                r'\b602\.\d+[a-z]?\b',  # Rule section 602 for Activated Abilities
                r'\b[Aa]ctivated [Aa]bilit(y|ies)\b',
                r'cost\s*:\s*effect', # General format shown in rules for abilities
                r'\bactivate this ability\b'
            ],
            'static_ability': [ # Broad category, might be useful
                r'\b604\.\d+[a-z]?\b', # Rule section 604 for Static Abilities
                r'\b[Ss]tatic [Aa]bilit(y|ies)\b',
                r'\b[Aa]s long as\b', # Common phrasing for static abilities
                r'\bhas\b (?:flying|trample|deathtouch|etc\.)' # Defines many keyword abilities as static
            ],
            'keyword_action': [ # General rules for keyword actions
                r'\b701\.\d+[a-z]?\b', # Rule section 701, which covers many keyword actions
                r'\b[Kk]eyword [Aa]ction\b'
            ]
        }
        
        # Match rules to mechanics
        for rule_item_for_mech in rules_cache["rules_data"]: # Renamed
            rule_text_lower = rule_item_for_mech["text"].lower() # Renamed
            for mechanic_key_rules, patterns_list in mechanic_keywords_for_rules.items(): # Renamed
                for pattern_str_rules in patterns_list: # Renamed
                    if re.search(pattern_str_rules, rule_text_lower, re.IGNORECASE):
                        new_mechanics_map.setdefault(mechanic_key_rules, []).append(rule_item_for_mech)
                        break  # Found a pattern for this mechanic in this rule, move to next mechanic
        
        mechanics_cache['mechanics_map'] = new_mechanics_map
        mechanics_cache['card_mechanics_map'] = {} # Initialize fresh
        mechanics_cache['loaded'] = True
        logger.info(f"Initialized mechanics mappings. Found rules for {len(new_mechanics_map)} mechanics.")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing mechanics mappings: {e}")
        mechanics_cache['loaded'] = False # Ensure this reflects failure
        return False


def get_card_embedding(card_data: Dict[str, Any]) -> Optional[np.ndarray]:
    """Get the embedding for a card, generating if needed. Returns np.ndarray or None."""
    card_id = card_data.get('id')
    if not card_id: # Card data must have an ID
        logger.warning(f"Card data missing 'id' for card '{card_data.get('name', 'Unknown')}'. Cannot fetch embedding by ID.")
        # Fallback to generating if possible, but this indicates a data issue.
    elif rag_system.db.is_connected:
        try:
            response = rag_system.db.client.from_("mtg_cards").select("text_embedding").eq("id", card_id).limit(1).execute()
            if response.data and response.data[0].get('text_embedding') is not None: # Check for explicit None too
                embedding_from_db = response.data[0]['text_embedding'] # Renamed
                parsed_db_embedding = parse_embedding_string(embedding_from_db) # Renamed
                if parsed_db_embedding is not None and parsed_db_embedding.size > 0:
                    return parsed_db_embedding
                else:
                    logger.warning(f"Failed to parse 'text_embedding' for card ID {card_id} from database. Value: {str(embedding_from_db)[:100]}")
            # else: logger.debug(f"No 'text_embedding' found in DB for card ID {card_id}") # If you want to log misses
        except Exception as e_db_emb: # Renamed
            logger.warning(f"Error retrieving card embedding from database for card ID {card_id}: {e_db_emb}")
    
    # If DB fetch failed or not connected, try generating
    if rag_system.embedding_available and hasattr(rag_system, 'embedding_model') and rag_system.embedding_model:
        oracle_text = card_data.get('oracle_text', '')
        card_name_for_emb = card_data.get('name', '') # Renamed
        card_type_for_emb = card_data.get('type_line', '') # Renamed
        
        text_for_embedding = f"Card: {card_name_for_emb}. Type: {card_type_for_emb}. Text: {oracle_text}".strip() # Added prefixes
        if not text_for_embedding or text_for_embedding == "Card: . Type: . Text:": # Check if effectively empty
            logger.warning(f"Not enough text data to generate embedding for card '{card_name_for_emb}'.")
            return None
        try:
            generated_embedding = rag_system.embedding_model.encode([text_for_embedding])[0] # Renamed
            return generated_embedding # This is already a numpy array
        except Exception as e_gen_emb:
            logger.error(f"Error generating embedding for card '{card_name_for_emb}': {e_gen_emb}")
            return None
    else:
        logger.warning(f"Embedding model not available, cannot generate embedding for card '{card_data.get('name', 'Unknown')}'.")
        return None

def get_card_data(card_name: str) -> Optional[Dict[str, Any]]: # Return Optional[Dict]
    """Get data for a card by name. Returns None if not found."""
    try:
        if rag_system.db.is_connected:
            response = rag_system.db.client.from_("mtg_cards").select("*").eq("name", card_name).limit(1).execute()
            if response.data:
                return response.data[0]
            else: # Card not found by exact name
                logger.warning(f"Card '{card_name}' not found by exact match in database.")
                # Optionally, try case-insensitive or fuzzy match here if desired for get_card_data
                return None
        else: # DB not connected
            logger.warning("Database not connected. Cannot get card data.")
            return None # Or perhaps {"name": card_name, "error": "DB not connected"}
    except Exception as e:
        logger.error(f"Error getting card data for '{card_name}': {e}")
        return None


def get_card_mechanics(card_data: Dict[str, Any]) -> List[str]:
    """Get the mechanics for a card, extracting if needed. card_data must be a valid card dict."""
    global mechanics_cache # Ensure global
    
    # Ensure mappings are initialized
    if not mechanics_cache['loaded']:
        if not initialize_mechanics_mappings(): # If init fails, return empty
            logger.warning("Mechanics mappings failed to initialize. Cannot get card mechanics.")
            return []
    
    card_name_for_mech_cache = card_data.get('name') # Renamed
    if not card_name_for_mech_cache:
        logger.warning("Card data has no name for mechanics cache lookup.")
        return extract_key_mechanics(card_data) # Extract fresh if no name for cache

    # Return cached mechanics if available
    if card_name_for_mech_cache in mechanics_cache['card_mechanics_map']:
        return mechanics_cache['card_mechanics_map'][card_name_for_mech_cache]
    
    # Extract mechanics if not cached
    extracted_mechanics = extract_key_mechanics(card_data) # Renamed
    
    # Cache the result
    mechanics_cache['card_mechanics_map'][card_name_for_mech_cache] = extracted_mechanics
    return extracted_mechanics


def find_related_rules(mechanics_list_input: List[str]) -> List[Dict[str, Any]]: # Renamed
    """Find rules related to a list of mechanics."""
    global mechanics_cache # Ensure global
    
    if not mechanics_list_input: return []

    if not mechanics_cache['loaded']:
        if not initialize_mechanics_mappings():
            logger.warning("Mechanics mappings failed to initialize. Cannot find related rules.")
            return []
            
    all_related_rules = [] # Renamed
    # For each mechanic, add related rules from the 'mechanics_map' (rules associated with mechanic keywords)
    for mechanic_key_input in mechanics_list_input: # Renamed
        if mechanic_key_input in mechanics_cache['mechanics_map']:
            all_related_rules.extend(mechanics_cache['mechanics_map'][mechanic_key_input])
    
    # Remove duplicates based on rule 'id'
    unique_rules_list = [] # Renamed
    seen_rule_ids = set() # Renamed
    for rule_item_to_check in all_related_rules: # Renamed
        rule_id_val = rule_item_to_check.get('id') # Renamed
        if rule_id_val and rule_id_val not in seen_rule_ids:
            seen_rule_ids.add(rule_id_val)
            unique_rules_list.append(rule_item_to_check)
    
    return unique_rules_list


def find_similar_cards(input_embeddings_list: List[Optional[np.ndarray]], exclude_names: Optional[List[str]] = None, limit: int = 50) -> List[Dict[str, Any]]: # Renamed
    """
    Finds cards similar to the given list of query embeddings.
    input_embeddings_list should contain np.ndarray objects or None.
    parse_embedding_string should have already been called on any raw data before this.
    """
    if exclude_names is None:
        exclude_names = []
    
    # Filter out None embeddings and ensure they are numpy arrays
    valid_query_embeddings = [emb for emb in input_embeddings_list if isinstance(emb, np.ndarray) and emb.size > 0] # Renamed

    if not valid_query_embeddings:
        logger.warning("No valid query embeddings provided to find_similar_cards after filtering.")
        return []
    
    try:
        # Ensure all embeddings have consistent dimensions.
        first_emb_dim = valid_query_embeddings[0].shape[0]
        if not all(emb.shape[0] == first_emb_dim for emb in valid_query_embeddings if emb.ndim > 0): # Check ndim too
            logger.error(f"Embedding dimension mismatch in find_similar_cards. Expected {first_emb_dim}, found others.")
            # Attempt to filter to most common dimension or just use first_emb_dim ones
            # This case should ideally be prevented by consistent embedding generation.
            # For now, we proceed hoping the RAG system's index can handle it or we average.
            # A more robust solution would be to ensure all input embeddings are conformant before this step.
            # Let's filter to the first dimension found.
            valid_query_embeddings = [emb for emb in valid_query_embeddings if emb.ndim > 0 and emb.shape[0] == first_emb_dim]
            if not valid_query_embeddings:
                logger.error("No embeddings remain after dimension consistency filtering.")
                return []
            logger.info(f"Proceeding with {len(valid_query_embeddings)} embeddings of dimension {first_emb_dim}.")

        # Average the embeddings if there are multiple to create a single query vector
        # Ensure all are 1D before averaging
        embeddings_1d_for_avg = [] # Renamed
        for emb_item_for_avg in valid_query_embeddings: # Renamed
            if emb_item_for_avg.ndim > 1: embeddings_1d_for_avg.append(emb_item_for_avg.flatten())
            elif emb_item_for_avg.ndim == 1: embeddings_1d_for_avg.append(emb_item_for_avg)
        
        if not embeddings_1d_for_avg:
            logger.error("No 1D embeddings found for averaging in find_similar_cards.")
            return []

        final_query_embedding = np.mean(embeddings_1d_for_avg, axis=0) if len(embeddings_1d_for_avg) > 1 else embeddings_1d_for_avg[0] # Renamed
        
        # Normalize the final query embedding
        embedding_norm = np.linalg.norm(final_query_embedding)
        if embedding_norm == 0:
            logger.error("Final query embedding has zero norm in find_similar_cards.")
            return []
        final_query_embedding_normalized = final_query_embedding / embedding_norm # Renamed
        
        # Check RAG system's text index
        if not rag_system.text_index or rag_system.text_index.ntotal == 0:
            logger.warning("RAG system text index not available or empty in find_similar_cards.")
            return []
        
        # Reshape for FAISS (ensure contiguous and proper dtype)
        faiss_query_vector = np.ascontiguousarray(final_query_embedding_normalized.reshape(1, -1).astype('float32')) # Renamed
        
        # Search using RAG system's FAISS index
        try:
            # Determine k for search, ensuring it's not more than ntotal
            k_search_val = min(limit * 2, rag_system.text_index.ntotal) # Renamed
            if k_search_val == 0 : # No items in index to search
                 logger.warning("FAISS index is empty (ntotal=0). Cannot search.")
                 return []

            distances_from_search, indices_from_search = rag_system.text_index.search(faiss_query_vector, k_search_val) # Renamed
        except Exception as e_faiss:
            logger.error(f"FAISS search failed in find_similar_cards: {e_faiss}")
            return []
        
        # Process search results
        retrieved_similar_cards = [] # Renamed
        if indices_from_search is not None and len(indices_from_search) > 0:
            for i_idx, faiss_idx_val in enumerate(indices_from_search[0]): # Renamed
                if 0 <= faiss_idx_val < len(rag_system.text_card_ids): # Check bounds
                    card_id_from_faiss = rag_system.text_card_ids[faiss_idx_val] # Renamed
                    try:
                        # Fetch card data from database
                        card_db_response = rag_system.db.client.from_("mtg_cards").select(
                            "id, name, oracle_text, type_line, color_identity, mana_cost, prices_usd" # Ensure all needed fields
                        ).eq("id", card_id_from_faiss).limit(1).execute()
                        
                        if card_db_response.data:
                            card_data_item = card_db_response.data[0] # Renamed
                            if card_data_item['name'] in exclude_names: # Check against exclude list
                                continue
                            # Add similarity score (convert FAISS distance to similarity)
                            card_data_item['similarity'] = float(1.0 - distances_from_search[0][i_idx]) if distances_from_search is not None and i_idx < len(distances_from_search[0]) else 0.5
                            retrieved_similar_cards.append(card_data_item)
                            if len(retrieved_similar_cards) >= limit: # Stop if limit is reached
                                break
                    except Exception as e_db_fetch:
                        logger.error(f"Error fetching card data for ID {card_id_from_faiss} in find_similar_cards: {e_db_fetch}")
                        continue # Skip this card and continue with others
        
        return retrieved_similar_cards
        
    except Exception as e_general_find:
        logger.error(f"General error in find_similar_cards: {e_general_find}")
        return []


def find_cards_by_rules(input_rule_embeddings: List[np.ndarray], exclude_names: Optional[List[str]] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Finds cards related to a list of rule embeddings.
    Assumes input_rule_embeddings is a list of valid np.ndarray objects.
    """
    if exclude_names is None:
        exclude_names = []
    
    # Filter for valid numpy arrays, though the input type hint already suggests this.
    valid_rule_embeddings = [emb for emb in input_rule_embeddings if isinstance(emb, np.ndarray) and emb.size > 0] # Renamed

    if not valid_rule_embeddings:
        logger.warning("No valid rule embeddings provided to find_cards_by_rules after filtering.")
        return []
    
    try:
        # Ensure consistency and prepare query embedding (similar to find_similar_cards)
        # Assuming rule embeddings are already consistent in dimension from their source (generate_rules_embeddings)
        first_emb_dim_rules = valid_rule_embeddings[0].shape[0] # Renamed
        # Filter for consistency (optional, but good for safety)
        valid_rule_embeddings = [emb for emb in valid_rule_embeddings if emb.ndim > 0 and emb.shape[0] == first_emb_dim_rules]
        if not valid_rule_embeddings:
             logger.error("No rule embeddings remain after dimension consistency for find_cards_by_rules.")
             return []

        # Average rule embeddings
        embeddings_1d_for_avg_rules = [] # Renamed
        for emb_item_rules in valid_rule_embeddings: # Renamed
            if emb_item_rules.ndim > 1: embeddings_1d_for_avg_rules.append(emb_item_rules.flatten())
            elif emb_item_rules.ndim == 1: embeddings_1d_for_avg_rules.append(emb_item_rules)

        if not embeddings_1d_for_avg_rules:
            logger.error("No 1D rule embeddings found for averaging in find_cards_by_rules.")
            return []

        final_rule_query_embedding = np.mean(embeddings_1d_for_avg_rules, axis=0) if len(embeddings_1d_for_avg_rules) > 1 else embeddings_1d_for_avg_rules[0] # Renamed
        
        # Normalize
        rule_embedding_norm = np.linalg.norm(final_rule_query_embedding) # Renamed
        if rule_embedding_norm == 0:
            logger.error("Final rule query embedding has zero norm in find_cards_by_rules.")
            return []
        final_rule_query_embedding_normalized = final_rule_query_embedding / rule_embedding_norm # Renamed
        
        # Check RAG text index
        if not rag_system.text_index or rag_system.text_index.ntotal == 0:
            logger.warning("RAG system text index not available or empty for rule-based card search.")
            return []
        
        # Prepare for FAISS
        faiss_rule_query_vector = np.ascontiguousarray(final_rule_query_embedding_normalized.reshape(1, -1).astype('float32')) # Renamed
        
        # FAISS search
        try:
            k_rules_search = min(limit * 2, rag_system.text_index.ntotal) # Renamed
            if k_rules_search == 0:
                logger.warning("FAISS index empty for rule-based card search.")
                return []
            distances_rules, indices_rules = rag_system.text_index.search(faiss_rule_query_vector, k_rules_search) # Renamed
        except Exception as e_faiss_rules: # Renamed
            logger.error(f"FAISS search failed for rule-based card search: {e_faiss_rules}")
            return []
        
        # Process results
        retrieved_rule_related_cards = [] # Renamed
        if indices_rules is not None and len(indices_rules) > 0:
            for i_rules_idx, faiss_idx_rules_val in enumerate(indices_rules[0]): # Renamed
                if 0 <= faiss_idx_rules_val < len(rag_system.text_card_ids):
                    card_id_from_rules_faiss = rag_system.text_card_ids[faiss_idx_rules_val] # Renamed
                    try:
                        card_db_response_rules = rag_system.db.client.from_("mtg_cards").select(
                            "id, name, oracle_text, type_line, color_identity, mana_cost, prices_usd"
                        ).eq("id", card_id_from_rules_faiss).limit(1).execute()
                        
                        if card_db_response_rules.data:
                            card_data_rules_item = card_db_response_rules.data[0] # Renamed
                            if card_data_rules_item['name'] in exclude_names:
                                continue
                            card_data_rules_item['similarity'] = float(1.0 - distances_rules[0][i_rules_idx]) if distances_rules is not None and i_rules_idx < len(distances_rules[0]) else 0.5
                            retrieved_rule_related_cards.append(card_data_rules_item)
                            if len(retrieved_rule_related_cards) >= limit:
                                break
                    except Exception as e_db_fetch_rules:
                        logger.error(f"Error fetching card data for ID {card_id_from_rules_faiss} in find_cards_by_rules: {e_db_fetch_rules}")
                        continue
        
        return retrieved_rule_related_cards
    except Exception as e_general_find_rules:
        logger.error(f"General error in find_cards_by_rules: {e_general_find_rules}")
        return []


def calculate_card_synergies(seed_card_names: List[str], count: int = 10) -> List[Dict[str, Any]]: # Renamed input
    """Calculate synergy scores between seed cards and potential matches."""
    try:
        card_embeddings_list = []  # Stores np.ndarray objects from get_card_embedding
        all_mechanics_set = set()    # Renamed
        seed_card_data_objects = []# Renamed

        for card_name_seed in seed_card_names: # Renamed
            _card_data_obj = get_card_data(card_name_seed) # Renamed
            if not _card_data_obj: # Check if card_data is None (not found)
                logger.warning(f"Could not retrieve data for seed card: {card_name_seed}. Skipping.")
                continue
            seed_card_data_objects.append(_card_data_obj)
            
            # Get card embedding (already returns np.ndarray or None)
            embedding_np = get_card_embedding(_card_data_obj) # Renamed
            if embedding_np is not None:
                card_embeddings_list.append(embedding_np)
            else:
                logger.warning(f"Could not get/parse embedding for seed card: {card_name_seed}")
            
            mechanics_list_seed = get_card_mechanics(_card_data_obj) # Renamed
            all_mechanics_set.update(mechanics_list_seed)
        
        if not seed_card_data_objects: # No valid seed cards were processed
            logger.warning("No valid seed card data available for synergy calculation.")
            return []

        # Find rules related to these mechanics
        related_rules_list_synergy = find_related_rules(list(all_mechanics_set)) # Renamed
        
        # Extract rule embeddings (list of np.ndarray or empty)
        rule_embeddings_list_synergy = [] # Renamed
        for rule_item_synergy in related_rules_list_synergy: # Renamed
            if rag_system.db.is_connected:
                try:
                    rule_db_response = rag_system.db.client.from_("mtg_rules").select("embedding").eq("rule_number", rule_item_synergy['rule_number']).limit(1).execute() # Renamed
                    if rule_db_response.data and rule_db_response.data[0].get('embedding') is not None:
                        embedding_data_from_db = rule_db_response.data[0]['embedding']
                        parsed_rule_embedding_np = parse_embedding_string(embedding_data_from_db) # Renamed
                        if parsed_rule_embedding_np is not None and parsed_rule_embedding_np.size > 0:
                            rule_embeddings_list_synergy.append(parsed_rule_embedding_np)
                        # else: logger.warning(f"Failed to parse rule embedding for {rule_item_synergy['rule_number']}") # Logged in parser
                except Exception as e_rule_emb_fetch:
                    logger.warning(f"Error retrieving/parsing rule embedding for rule {rule_item_synergy['rule_number']} in synergy calc: {e_rule_emb_fetch}")
        
        logger.info(f"Synergy Calc: Retrieved {len(rule_embeddings_list_synergy)} valid rule embeddings out of {len(related_rules_list_synergy)} related rules.")
        
        # card_embeddings_list is now a list of np.ndarrays (or empty)
        similar_cards_result = find_similar_cards(card_embeddings_list, exclude_names=seed_card_names, limit=count * 2) # Renamed
        
        rule_related_cards_result = [] # Renamed
        if rule_embeddings_list_synergy: # This is a list of np.ndarrays
            rule_related_cards_result = find_cards_by_rules(rule_embeddings_list_synergy, exclude_names=seed_card_names, limit=count * 2)
        else:
            logger.info("Synergy Calc: No rule embeddings available, skipping rules-based card similarity search.")
        
        # Calculate combined scores
        final_synergy_results = calculate_combined_scores(similar_cards_result, rule_related_cards_result, seed_card_data_objects) # Renamed
        
        # Return top results
        return final_synergy_results[:count]
        
    except Exception as e_calc_synergy:
        logger.error(f"General error calculating card synergies: {e_calc_synergy}")
        return []


def calculate_combined_scores(similar_cards_list: List[Dict], rule_related_cards_list: List[Dict], seed_card_data_list_input: List[Dict]) -> List[Dict]: # Renamed
    """Calculate combined synergy scores with robust error handling."""
    combined_scores_map = {} # Renamed
    
    # Add text similarity scores (weight: 0.4)
    for card_item_sim in similar_cards_list: # Renamed
        try:
            card_name_sim = card_item_sim.get('name') # Renamed
            if not card_name_sim:
                logger.debug("Similar card item missing name in combined scores.")
                continue
            
            similarity_value = card_item_sim.get('similarity', 0.5) # This should be a float from find_similar_cards
            if not isinstance(similarity_value, (int, float)): # Defensive check
                logger.warning(f"Similarity for {card_name_sim} is not a number: {similarity_value}. Defaulting to 0.5.")
                similarity_value = 0.5
            
            normalized_similarity = max(0.0, min(1.0, float(similarity_value))) # Ensure 0-1
            
            combined_scores_map[card_name_sim] = {
                'card': card_item_sim,
                'similarity_score': normalized_similarity * 0.4, # Weighted score
                'rules_score': 0.0, # Initialize rules score
                'total_score': normalized_similarity * 0.4
            }
        except Exception as e_sim_score:
            logger.warning(f"Error processing similar card '{card_item_sim.get('name', 'unknown')}' for combined scores: {e_sim_score}")
            continue # Skip this card
    
    # Add rules-based scores (weight: 0.6)
    for card_item_rule in rule_related_cards_list: # Renamed
        try:
            card_name_rule = card_item_rule.get('name') # Renamed
            if not card_name_rule:
                logger.debug("Rule-related card item missing name in combined scores.")
                continue
            
            rule_similarity_value = card_item_rule.get('similarity', 0.5) # Should be float from find_cards_by_rules
            if not isinstance(rule_similarity_value, (int, float)): # Defensive
                logger.warning(f"Rule similarity for {card_name_rule} is not a number: {rule_similarity_value}. Defaulting to 0.5.")
                rule_similarity_value = 0.5

            normalized_rule_similarity = max(0.0, min(1.0, float(rule_similarity_value))) # Ensure 0-1
            weighted_rules_score = normalized_rule_similarity * 0.6 # Weighted score
            
            if card_name_rule in combined_scores_map:
                combined_scores_map[card_name_rule]['rules_score'] = weighted_rules_score
                # Ensure total_score is float before adding
                current_total = combined_scores_map[card_name_rule].get('total_score', 0.0)
                combined_scores_map[card_name_rule]['total_score'] = float(current_total) + weighted_rules_score
            else: # Card was found by rules but not by text similarity
                combined_scores_map[card_name_rule] = {
                    'card': card_item_rule,
                    'similarity_score': 0.0, # No text similarity score
                    'rules_score': weighted_rules_score,
                    'total_score': weighted_rules_score
                }
        except Exception as e_rule_score:
            logger.warning(f"Error processing rule-related card '{card_item_rule.get('name', 'unknown')}' for combined scores: {e_rule_score}")
            continue

    # Add explanations
    for card_name_explain, data_item_explain in combined_scores_map.items(): # Renamed
        try:
            card_obj_explain = data_item_explain.get('card', {}) # Renamed
            explanations_list = [] # Renamed
            
            card_mechanics_list = get_card_mechanics(card_obj_explain) if card_obj_explain else [] # Renamed
            
            # Aggregate all mechanics from all seed cards
            seed_card_all_mechanics_set = set() # Renamed
            for seed_card_obj in seed_card_data_list_input: # Renamed
                if seed_card_obj and isinstance(seed_card_obj, dict): # Check if it's a valid card dict
                    seed_card_all_mechanics_set.update(get_card_mechanics(seed_card_obj))
            
            # Find shared mechanics
            shared_mechanics_list = [m for m in card_mechanics_list if m in seed_card_all_mechanics_set] # Renamed
            if shared_mechanics_list:
                explanations_list.append(f"Shares mechanics: {', '.join(shared_mechanics_list)}")
            
            # Add explanations based on scores
            text_sim_score = data_item_explain.get('similarity_score', 0.0) # Renamed
            rules_based_score = data_item_explain.get('rules_score', 0.0) # Renamed
            
            if isinstance(text_sim_score, (int, float)) and text_sim_score > 0.1: # Adjusted threshold, actual score is already weighted
                explanations_list.append("Similar card text/function to seed cards.")
            if isinstance(rules_based_score, (int, float)) and rules_based_score > 0.15: # Adjusted threshold
                explanations_list.append("Interacts with similar game mechanics/rules as seed cards.")
            
            if not explanations_list: # Default explanation if none specific
                explanations_list = ["Potentially synergizes with seed cards."]
            
            data_item_explain['explanations'] = explanations_list
            
        except Exception as e_explain:
            logger.error(f"Error adding explanations for card '{card_name_explain}': {e_explain}")
            # Ensure 'explanations' key exists even if an error occurs
            data_item_explain['explanations'] = data_item_explain.get('explanations', ["Error generating synergy explanation."])
    
    # Convert map to list and sort by total_score
    final_result_list = [] # Renamed
    for card_name_final, card_data_final in combined_scores_map.items(): # Renamed
        try:
            total_score_val = card_data_final.get('total_score', 0.0) # Renamed
            if not isinstance(total_score_val, (int, float)): # Defensive type check
                total_score_val = 0.0
            
            # Ensure total_score is float and clamp (max possible is 0.4 + 0.6 = 1.0 from these two sources)
            card_data_final['total_score'] = max(0.0, min(1.0, float(total_score_val))) 
            final_result_list.append(card_data_final)
        except Exception as e_final_proc:
            logger.warning(f"Error processing final result for card '{card_name_final}' before sort: {e_final_proc}")
            continue
    
    # Safe sorting
    try:
        final_result_list.sort(key=lambda x_sort: x_sort.get('total_score', 0.0), reverse=True) # Renamed
    except Exception as e_sort:
        logger.error(f"Error sorting combined score results: {e_sort}")
        # Return unsorted if sort fails, rather than empty or erroring out
    
    return final_result_list


@lru_cache(maxsize=100) # Keep cache for frequent keyword searches
def cached_card_search(keyword: str, limit: int) -> List[Dict[str, Any]]:
    """Cache frequent keyword searches to improve performance."""
    logger.debug(f"Performing cached card search for keyword: '{keyword}', limit: {limit}")
    return rag_system.search_cards_by_keyword(keyword, limit)


# ========= API Endpoints =========
# (The rest of the API endpoints from your original file should follow here.)
# I will re-insert them, ensuring variable names are clear and they use
# the updated helper functions correctly where applicable.

@app.route('/api/rag/status', methods=['GET'])
def rag_status_endpoint(): # Renamed endpoint function
    """Get the status of the RAG system"""
    try:
        price_index_loaded = rag_system.price_index is not None # Renamed
        text_index_loaded = rag_system.text_index is not None and rag_system.text_index.ntotal > 0 # Renamed, added ntotal check
        
        current_rules_status = { # Renamed
            "cached_rules_loaded": rules_cache["loaded"],
            "cached_rules_count": len(rules_cache["rules_data"]) if rules_cache["loaded"] else 0,
            "cached_embeddings_present": rules_cache.get("rules_embeddings") is not None and rules_cache["rules_embeddings"].size > 0 if rules_cache.get("rules_embeddings") is not None else False
        }
        
        if rag_system.db.is_connected:
            try:
                # A more efficient way to count in Supabase/Postgres:
                count_query = rag_system.db.client.from_("mtg_rules").select("id", count="exact", head=True).execute()
                current_rules_status["database_rules_count"] = count_query.count if hasattr(count_query, 'count') else "Error/Unknown"
            except Exception as e_db_rules_count:
                logger.error(f"Error checking database rules count: {e_db_rules_count}")
                current_rules_status["database_rules_count"] = "Error querying table"
        else:
            current_rules_status["database_rules_count"] = "DB not connected"

        current_mechanics_status = { # Renamed
            "cached_mappings_loaded": mechanics_cache["loaded"],
            "mapped_mechanics_count": len(mechanics_cache["mechanics_map"]) if mechanics_cache["loaded"] else 0,
            "cards_with_cached_mechanics": len(mechanics_cache["card_mechanics_map"]) if mechanics_cache["loaded"] else 0
        }
        
        return jsonify({
            "success": True,
            "status": {
                "price_index_loaded": price_index_loaded,
                "text_index_loaded": text_index_loaded,
                "db_connection_active": rag_system.db.is_connected, # Renamed
                "embedding_model_available": rag_system.embedding_available, # Renamed
                "rules_module": current_rules_status, # Renamed
                "mechanics_module": current_mechanics_status # Renamed
            }
        })
    except Exception as e_status: # Renamed
        logger.error(f"Error in RAG status endpoint: {e_status}")
        return jsonify({"success": False, "error": str(e_status)}), 500


@app.route('/api/rag/synergy-search', methods=['POST'])
def find_synergistic_cards_endpoint(): # This was the original name, it's fine.
    """Find cards that synergize with seed cards"""
    try:
        request_data = request.json # Renamed
        if not request_data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        seed_card_names_input = request_data.get('cards', []) # Renamed
        if not seed_card_names_input or not isinstance(seed_card_names_input, list): # Added type check
            return jsonify({"success": False, "error": "No seed cards (list) provided"}), 400
        
        # Ensure all seed cards are strings
        if not all(isinstance(name, str) for name in seed_card_names_input):
            return jsonify({"success": False, "error": "All seed cards must be strings (card names)."}), 400

        count_requested = min(int(request_data.get('count', 10)), 50) # Renamed & cap
        
        synergistic_cards_results = calculate_card_synergies(seed_card_names_input, count_requested) # Renamed
        
        # Format results for the API response
        formatted_api_results = [] # Renamed
        for item_synergy_data in synergistic_cards_results: # Renamed
            card_object = item_synergy_data.get('card', {}) # Renamed, ensure card object exists
            # Ensure all keys are present with defaults if necessary
            formatted_api_results.append({
                "name": card_object.get("name", "Unknown Card"),
                "similarity_score": item_synergy_data.get("similarity_score", 0.0),
                "rules_score": item_synergy_data.get("rules_score", 0.0),
                "total_score": item_synergy_data.get("total_score", 0.0),
                "oracle_text": card_object.get("oracle_text", ""),
                "type_line": card_object.get("type_line", ""),
                "mana_cost": card_object.get("mana_cost", ""),
                "color_identity": card_object.get("color_identity", []),
                "price": card_object.get("prices_usd", 0.0), # Default to float
                "explanations": item_synergy_data.get("explanations", [])
            })
        
        return jsonify({
            "success": True,
            "seed_cards": seed_card_names_input,
            "synergistic_cards": formatted_api_results
        })
    
    except ValueError as e_val: # Specific error for int conversion
        logger.warning(f"Invalid 'count' parameter in synergy-search: {e_val}")
        return jsonify({"success": False, "error": "Invalid 'count' parameter. Must be an integer."}), 400
    except Exception as e_synergy_endpoint: # Renamed
        logger.error(f"Error in synergy-search endpoint: {e_synergy_endpoint}")
        return jsonify({"success": False, "error": str(e_synergy_endpoint)}), 500


@app.route('/api/rag/similar-price', methods=['GET'])
def similar_price_endpoint(): # Renamed
    """Get cards with similar price characteristics"""
    card_name_query_param = request.args.get('card', '') # Renamed
    if not card_name_query_param:
        return jsonify({"success": False, "error": "No card name provided in 'card' parameter."}), 400
    
    try:
        limit_val = min(int(request.args.get('limit', 10)), 100) # Renamed
        page_num = int(request.args.get('page', 1)) # Renamed
    except ValueError:
        return jsonify({"success": False, "error": "Invalid 'limit' or 'page' parameter. Must be integers."}), 400

    # Filters
    rarity_filter = request.args.get('rarity') # Renamed
    colors_filter = validate_colors(request.args.get('colors')) # Renamed
    format_legality_filter = validate_format(request.args.get('format')) # Renamed
    # exact_phrase not typically relevant for price similarity, but kept from original if other logic uses it
    # exact_phrase_filter = request.args.get('exact_phrase') == 'true' 
    
    card_type_filter = request.args.get('type') # Renamed
    exclude_terms_str = request.args.get('exclude', '') # Renamed
    exclude_terms_list = [term.strip() for term in exclude_terms_str.split(',') if term.strip()] # Renamed
    # banned_filter = request.args.get('banned') == 'true' # Renamed # Replaced with more specific logic below

    try:
        # Retrieve cards based on price similarity (this function needs to exist in rag_system)
        # Assuming retrieve_cards_by_price returns a list of card dicts
        initial_similar_cards = rag_system.retrieve_cards_by_price(card_name_query_param, limit_val * 3) # Renamed, get extra for filtering
        
        # Apply filters
        filtered_price_cards = [] # Renamed
        for card_data_item_price in initial_similar_cards: # Renamed
            if check_for_excluded_terms(card_data_item_price, exclude_terms_list):
                continue
            if rarity_filter and card_data_item_price.get('rarity', '').lower() != rarity_filter.lower(): # Case-insensitive rarity
                continue
            if card_type_filter and card_type_filter.lower() not in card_data_item_price.get('type_line', '').lower():
                continue
            
            # Color filter (using color_identity as it's more standard for deck building)
            card_color_identity = card_data_item_price.get('color_identity', [])
            if colors_filter and not check_color_match(card_color_identity, colors_filter):
                 continue

            # Format legality filter
            if format_legality_filter:
                legalities = card_data_item_price.get('legalities', {})
                # Ensure legalities is a dict
                if isinstance(legalities, str):
                    try: legalities = json.loads(legalities) if legalities else {}
                    except json.JSONDecodeError: legalities = {}
                if not isinstance(legalities, dict): legalities = {} # Default to empty if not dict after attempt

                if legalities.get(format_legality_filter) not in ['legal', 'restricted']:
                    continue
            
            filtered_price_cards.append(card_data_item_price)
        
        # Apply pagination
        start_idx = (page_num - 1) * limit_val # Renamed
        paginated_price_results = filtered_price_cards[start_idx : start_idx + limit_val] # Renamed
        
        return jsonify({
            "success": True,
            "card_query": card_name_query_param, # Renamed
            "similar_cards": paginated_price_results,
            "total_filtered": len(filtered_price_cards), # Renamed
            "page": page_num,
            "limit": limit_val,
            "filters_applied": {
                "excluded_terms": exclude_terms_list,
                "colors": colors_filter,
                "rarity": rarity_filter,
                "card_type": card_type_filter,
                "format_legality": format_legality_filter,
                # "banned_in_commander": banned_filter, # Original 'banned' was generic
            }
        })
        
    except Exception as e_price_sim: # Renamed
        logger.error(f"Error finding similar price cards for '{card_name_query_param}': {e_price_sim}")
        return jsonify({"success": False, "error": str(e_price_sim)}), 500


@app.route('/api/rag/similar-text', methods=['GET'])
def similar_text_endpoint(): # Renamed
    """Find cards with similar text content"""
    query_text_param = request.args.get('query') # Renamed
    if not query_text_param:
        return jsonify({"success": False, "error": "No query text provided in 'query' parameter."}), 400

    try:
        limit_val = min(int(request.args.get('limit', 10)), 100)
        page_num = int(request.args.get('page', 1))
    except ValueError:
        return jsonify({"success": False, "error": "Invalid 'limit' or 'page' parameter. Must be integers."}), 400

    exact_phrase_match = request.args.get('exact_phrase') == 'true' # Renamed
    
    # Filters
    colors_filter_text = validate_colors(request.args.get('colors')) # Renamed
    card_type_filter_text = request.args.get('type') # Renamed
    rarity_filter_text = request.args.get('rarity') # Renamed
    format_legality_filter_text = validate_format(request.args.get('format')) # Renamed
    exclude_terms_str_text = request.args.get('exclude', '') # Renamed
    exclude_terms_list_text = [term.strip() for term in exclude_terms_str_text.split(',') if term.strip()] # Renamed
    # banned_filter_text = request.args.get('banned') == 'true' # Renamed: covered by format legality more generally

    try:
        initial_text_results = [] # Renamed
        if exact_phrase_match:
            # Escape special characters for SQL LIKE pattern more carefully
            # search_pattern = query.replace('%', '\\%').replace('_', '\\_') # Original
            # Python's re.escape is good for regex, for SQL LIKE, it's simpler:
            search_pattern_sql = query_text_param.replace('%', '%%').replace('_', '\\_') # Standard SQL escaping for LIKE
            
            # Query the database directly for exact phrase matches
            # Ensure all necessary fields for filtering are selected
            db_text_response = rag_system.db.client.from_("mtg_cards").select(
                "*" # Select all to ensure all filterable fields are present
            ).ilike("oracle_text", f"%{search_pattern_sql}%").limit(limit_val * 3).execute() # Renamed
            initial_text_results = db_text_response.data if db_text_response.data else []
        else:
            # Use embedding-based semantic search
            initial_text_results = rag_system.retrieve_cards_by_text(query_text_param, limit_val * 3)
        
        # Apply filters
        filtered_text_cards = [] # Renamed
        for card_data_text_item in initial_text_results: # Renamed
            if check_for_excluded_terms(card_data_text_item, exclude_terms_list_text):
                continue
            
            card_colors_for_check = card_data_text_item.get('color_identity', card_data_text_item.get('colors', []))
            if colors_filter_text and not check_color_match(card_colors_for_check, colors_filter_text):
                continue
            if card_type_filter_text and card_type_filter_text.lower() not in card_data_text_item.get('type_line', '').lower():
                continue
            if rarity_filter_text and card_data_text_item.get('rarity', '').lower() != rarity_filter_text.lower():
                continue
            
            # Format legality filter
            if format_legality_filter_text:
                legalities = card_data_text_item.get('legalities', {})
                if isinstance(legalities, str): # Ensure dict
                    try: legalities = json.loads(legalities) if legalities else {}
                    except json.JSONDecodeError: legalities = {}
                if not isinstance(legalities, dict): legalities = {}

                if legalities.get(format_legality_filter_text) not in ['legal', 'restricted']:
                    continue
            
            filtered_text_cards.append(card_data_text_item)
        
        # Apply pagination
        start_idx_text = (page_num - 1) * limit_val # Renamed
        paginated_text_results = filtered_text_cards[start_idx_text : start_idx_text + limit_val] # Renamed
        
        return jsonify({
            "success": True,
            "query": query_text_param,
            "similar_cards": paginated_text_results,
            "total_filtered": len(filtered_text_cards), # Renamed
            "page": page_num,
            "limit": limit_val,
            "filters_applied": {
                "excluded_terms": exclude_terms_list_text,
                "colors": colors_filter_text,
                "rarity": rarity_filter_text,
                "card_type": card_type_filter_text,
                "format_legality": format_legality_filter_text,
                # "banned": banned_filter_text, # Covered by format_legality generally or specific banned check
                "exact_phrase": exact_phrase_match
            }
        })
        
    except Exception as e_text_sim: # Renamed
        logger.error(f"Error searching similar text for query '{query_text_param}': {e_text_sim}")
        return jsonify({"success": False, "error": str(e_text_sim)}), 500

@app.route('/api/rag/deck-recommendation', methods=['POST'])
def deck_recommendation_endpoint(): # Renamed
    """Get a deck recommendation based on strategy and constraints"""
    request_data_deck = request.json # Renamed
    
    if not request_data_deck:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    strategy_input = request_data_deck.get('strategy', '') # Renamed
    if not strategy_input:
        return jsonify({"success": False, "error": "No strategy provided"}), 400
    
    commander_input = request_data_deck.get('commander', None) # Renamed
    budget_input = request_data_deck.get('budget', None) # Renamed
    if budget_input is not None: # Validate budget if provided
        try: budget_input = float(budget_input)
        except ValueError: return jsonify({"success": False, "error": "Budget must be a valid number."}), 400
    
    # Enhanced filtering options
    colors_raw_input = request_data_deck.get('colors', None) # Renamed
    # Validate colors whether it's a list of strings or a comma-separated string
    final_colors_filter = [] # Renamed
    if isinstance(colors_raw_input, list):
        final_colors_filter = validate_colors(','.join(colors_raw_input)) if all(isinstance(c, str) for c in colors_raw_input) else []
    elif isinstance(colors_raw_input, str):
        final_colors_filter = validate_colors(colors_raw_input)
    
    color_combo_input = request_data_deck.get('color_combo', None) # Renamed
    # Color combination mapping (condensed for clarity)
    color_map_guilds_shards_wedges = {
        'colorless': ['C'], 'mono': 'mono', # Mono needs special handling with final_colors_filter
        'azorius': ['W', 'U'], 'dimir': ['U', 'B'], 'rakdos': ['B', 'R'], 'gruul': ['R', 'G'], 'selesnya': ['G', 'W'],
        'orzhov': ['W', 'B'], 'izzet': ['U', 'R'], 'golgari': ['B', 'G'], 'boros': ['R', 'W'], 'simic': ['G', 'U'],
        'esper': ['W', 'U', 'B'], 'grixis': ['U', 'B', 'R'], 'jund': ['B', 'R', 'G'], 'naya': ['R', 'G', 'W'], 'bant': ['G', 'W', 'U'],
        'abzan': ['W', 'B', 'G'], 'jeskai': ['U', 'R', 'W'], 'sultai': ['B', 'G', 'U'], 'mardu': ['R', 'W', 'B'], 'temur': ['G', 'U', 'R'],
        'five_color': ['W', 'U', 'B', 'R', 'G'] # Added five_color for clarity
    }
    if color_combo_input:
        if color_combo_input == 'mono':
            if final_colors_filter and len(final_colors_filter) == 1: # Mono and a single color is specified
                pass # Already correct
            elif final_colors_filter and len(final_colors_filter) > 1: # Mono but multiple colors given, use first
                logger.warning(f"Mono color combo selected with multiple colors {final_colors_filter}, using first: {final_colors_filter[0]}")
                final_colors_filter = [final_colors_filter[0]]
            # If no colors specified but mono, it's ambiguous, let RAG handle general mono if possible
        elif color_combo_input in color_map_guilds_shards_wedges:
            final_colors_filter = color_map_guilds_shards_wedges[color_combo_input]
    
    try:
        card_limit_val = min(int(request_data_deck.get('card_limit', 60)), 100) # Renamed
    except ValueError: return jsonify({"success": False, "error": "Card limit must be an integer."}), 400

    include_lands_bool = request_data_deck.get('include_lands', True) # Renamed
    format_legality_deck = validate_format(request_data_deck.get('format', 'commander')) # Renamed
    # banned_deck_filter = request_data_deck.get('banned') == True # Renamed: 'banned' is not a direct filter for recommendation usually
    
    themes_input = request_data_deck.get('themes', '') # Renamed
    if themes_input:
        strategy_input += f" with themes: {themes_input}" # Make it clear it's themes
    
    exclude_terms_str_deck = request_data_deck.get('exclude', '') # Renamed
    exclude_terms_list_deck = [term.strip() for term in exclude_terms_str_deck.split(',') if term.strip()] # Renamed
    card_type_filter_deck = request_data_deck.get('type') # Renamed
    
    logger.info(f"Deck recommendation request: strategy='{strategy_input}', commander='{commander_input}', colors='{final_colors_filter}', budget='{budget_input}', exclude_terms={exclude_terms_list_deck}")
    
    try:
        # Generate enhanced deck recommendation
        # rag_system.generate_deck_recommendation needs to be robust
        deck_recommendation_result = rag_system.generate_deck_recommendation(
            strategy_input, commander_input, budget_input, final_colors_filter # Pass validated colors
        )
        
        logger.info(f"Retrieved {len(deck_recommendation_result.get('cards', []))} initial cards for recommendation.")
        
        # Filter out cards based on new filters if 'cards' key exists
        if 'cards' in deck_recommendation_result and isinstance(deck_recommendation_result['cards'], list):
            filtered_deck_cards_list = [] # Renamed
            for card_item_deck in deck_recommendation_result['cards']: # Renamed
                if not isinstance(card_item_deck, dict): continue # Skip non-dict items

                # Ensure legality data is a dict
                legalities = card_item_deck.get('legalities', {})
                if isinstance(legalities, str):
                    try: legalities = json.loads(legalities) if legalities else {}
                    except json.JSONDecodeError: legalities = {}
                if not isinstance(legalities, dict): legalities = {}
                card_item_deck['legalities'] = legalities # Update card with parsed legalities

                if check_for_excluded_terms(card_item_deck, exclude_terms_list_deck):
                    continue
                if card_type_filter_deck and card_type_filter_deck.lower() not in card_item_deck.get('type_line', '').lower():
                    continue
                
                # Apply format legality (don't typically filter for 'banned' in recommendations unless specifically requested)
                if format_legality_deck and legalities.get(format_legality_deck) not in ['legal', 'restricted']:
                    continue
                
                if not include_lands_bool and 'Land' in card_item_deck.get('type_line', ''): # Case-sensitive 'Land'
                    continue
                
                filtered_deck_cards_list.append(card_item_deck)
            
            logger.info(f"After filtering, {len(filtered_deck_cards_list)} cards remain for deck recommendation.")
            
            # Fallback if no cards remain after filtering but there were initial results
            if not filtered_deck_cards_list and deck_recommendation_result['cards']:
                logger.warning(f"No cards remained after filtering for strategy: {strategy_input}. Using a portion of initial results.")
                # Take a slice of the original unfiltered (but strategy-matched) cards
                deck_recommendation_result['cards'] = deck_recommendation_result['cards'][:card_limit_val]
                deck_recommendation_result['note'] = "Filters were too restrictive; showing top unfiltered strategy-matched cards."
            else:
                deck_recommendation_result['cards'] = filtered_deck_cards_list[:card_limit_val]

            deck_recommendation_result['total_cards'] = len(deck_recommendation_result['cards'])
        
        # Add filter information to response for clarity
        deck_recommendation_result['filters_applied'] = {
            "strategy": strategy_input, "commander": commander_input, "final_colors_filter": final_colors_filter,
            "color_combo_input": color_combo_input, "themes_input": themes_input, "exclude_terms_list": exclude_terms_list_deck,
            "card_type_filter": card_type_filter_deck, "format_legality_deck": format_legality_deck,
            # "banned_filter": banned_deck_filter, # Not directly applied as a filter here
            "include_lands": include_lands_bool, "budget_input": budget_input, "card_limit": card_limit_val
        }
        
        return jsonify({"success": True, "deck": deck_recommendation_result})
        
    except Exception as e_deck_rec: # Renamed
        logger.exception(f"Error generating deck recommendation: {e_deck_rec}") # Use logger.exception for stack trace
        return jsonify({"success": False, "error": str(e_deck_rec)}), 500


@app.route('/api/rag/search-keyword', methods=['GET'])
def search_keyword_endpoint(): # Renamed
    """Search cards by keyword with enhanced filtering options"""
    keyword_param = request.args.get('keyword') or request.args.get('query') # Renamed
    if not keyword_param:
        return jsonify({"success": False, "error": "No keyword/query provided"}), 400
    
    try:
        limit_val = min(int(request.args.get('limit', 10)), 100)
        page_num = int(request.args.get('page', 1))
    except ValueError:
        return jsonify({"success": False, "error": "Invalid 'limit' or 'page' parameter."}), 400

    exact_phrase_keyword = request.args.get('exact_phrase') == 'true' # Renamed
    
    colors_filter_kw = validate_colors(request.args.get('colors')) # Renamed
    card_type_filter_kw = request.args.get('type') # Renamed
    rarity_filter_kw = request.args.get('rarity') # Renamed
    format_legality_kw = validate_format(request.args.get('format')) # Renamed
    exclude_terms_str_kw = request.args.get('exclude', '') # Renamed
    exclude_terms_list_kw = [term.strip() for term in exclude_terms_str_kw.split(',') if term.strip()] # Renamed
    # banned_filter_kw = request.args.get('banned') == 'true' # Renamed

    try:
        initial_keyword_results = [] # Renamed
        if exact_phrase_keyword:
            search_pattern_sql_kw = keyword_param.replace('%', '%%').replace('_', '\\_') # Renamed
            db_keyword_response = rag_system.db.client.from_("mtg_cards").select("*").ilike("oracle_text", f"%{search_pattern_sql_kw}%").limit(limit_val * 3).execute() # Renamed
            initial_keyword_results = db_keyword_response.data if db_keyword_response.data else []
        else:
            # Use regular keyword search (cached if frequent)
            initial_keyword_results = cached_card_search(keyword_param, limit_val * 3)
        
        # Apply filters
        filtered_keyword_cards = [] # Renamed
        for card_item_kw in initial_keyword_results: # Renamed
            if not isinstance(card_item_kw, dict): continue # Ensure item is a dict

            if check_for_excluded_terms(card_item_kw, exclude_terms_list_kw):
                continue
            
            card_colors_for_check_kw = card_item_kw.get('color_identity', card_item_kw.get('colors', [])) # Renamed
            if colors_filter_kw and not check_color_match(card_colors_for_check_kw, colors_filter_kw):
                continue
            if card_type_filter_kw and card_type_filter_kw.lower() not in card_item_kw.get('type_line', '').lower():
                continue
            if rarity_filter_kw and card_item_kw.get('rarity', '').lower() != rarity_filter_kw.lower():
                continue
            
            # Format legality
            if format_legality_kw:
                legalities_kw = card_item_kw.get('legalities', {}) # Renamed
                if isinstance(legalities_kw, str):
                    try: legalities_kw = json.loads(legalities_kw) if legalities_kw else {}
                    except json.JSONDecodeError: legalities_kw = {}
                if not isinstance(legalities_kw, dict): legalities_kw = {}

                if legalities_kw.get(format_legality_kw) not in ['legal', 'restricted']:
                    continue
            
            filtered_keyword_cards.append(card_item_kw)
        
        # Apply pagination
        start_idx_kw = (page_num - 1) * limit_val # Renamed
        paginated_keyword_results = filtered_keyword_cards[start_idx_kw : start_idx_kw + limit_val] # Renamed
        
        return jsonify({
            "success": True,
            "keyword": keyword_param,
            "cards": paginated_keyword_results,
            "total_filtered": len(filtered_keyword_cards), # Renamed
            "page": page_num,
            "limit": limit_val,
            "filters_applied": {
                "exclude_terms": exclude_terms_list_kw,
                "colors": colors_filter_kw,
                "rarity": rarity_filter_kw,
                "card_type": card_type_filter_kw,
                "format_legality": format_legality_kw,
                # "banned": banned_filter_kw, # Not explicitly applied as a separate filter here
                "exact_phrase": exact_phrase_keyword
            }
        })
        
    except Exception as e_kw_search: # Renamed
        logger.error(f"Error searching by keyword '{keyword_param}': {e_kw_search}")
        return jsonify({"success": False, "error": str(e_kw_search)}), 500


@app.route('/api/rag/rules-search', methods=['GET'])
def search_rules_endpoint(): # Renamed
    """Search MTG rules using semantic search, now with database support"""
    query_text_rules = request.args.get('query') # Renamed
    if not query_text_rules:
        return jsonify({"success": False, "error": "No query provided for rules search."}), 400
    
    try:
        limit_val = min(int(request.args.get('limit', 10)), 50)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid 'limit' parameter for rules search."}), 400
    
    exact_rule_num_param = request.args.get('rule_number') # Renamed
    
    try:
        if query_text_rules == "status_check": # Handle status check query
            if not rules_cache["loaded"]: load_rules_data() # Ensure cache is attempted to be loaded
            return jsonify({
                "success": True,
                "rules_status": {
                    "cache_loaded": rules_cache["loaded"],
                    "rules_in_cache": len(rules_cache["rules_data"]),
                    "embeddings_in_cache": rules_cache["rules_embeddings"].shape[0] if rules_cache.get("rules_embeddings") is not None and rules_cache["rules_embeddings"].size > 0 else 0,
                    "database_connected_for_rules": rag_system.db.is_connected
                }
            })
        
        # Database Search Logic (if connected)
        if rag_system.db.is_connected:
            try:
                if exact_rule_num_param:
                    # Direct lookup by rule number in database
                    exact_rule_response = rag_system.db.client.from_("mtg_rules").select("*").eq("rule_number", exact_rule_num_param).limit(limit_val).execute()
                    if exact_rule_response.data:
                        formatted_rules = [{"rule_number": r["rule_number"], "section": r.get("section","N/A"), "title": r.get("title","N/A"), "text": r["text"], "id": r.get("id", f"rule_{r['rule_number'].replace('.', '_')}")} for r in exact_rule_response.data]
                        return jsonify({"success": True, "query": query_text_rules, "rules": formatted_rules, "total": len(formatted_rules), "search_method": "database_exact_number"})
                
                # Semantic vector search in database if embedding available
                if rag_system.embedding_available and hasattr(rag_system.embedding_model, 'encode'):
                    query_embedding_list = rag_system.embedding_model.encode([query_text_rules])[0].tolist()
                    vector_search_response = rag_system.db.client.rpc(
                        'match_rules_by_embedding', # Assuming this RPC function exists and is set up in Supabase
                        {
                            'query_embedding_param': query_embedding_list, # Ensure param name matches RPC
                            'match_threshold_param': 0.5,  # Ensure param name matches RPC
                            'match_count_param': limit_val    # Ensure param name matches RPC
                        }
                    ).execute()
                    if hasattr(vector_search_response, 'data') and vector_search_response.data: # Check for data attribute
                        formatted_rules_vec = [{"rule_number": r["rule_number"], "section": r.get("section","N/A"), "title": r.get("title","N/A"), "text": r["text"], "id": r.get("id", f"rule_{r['rule_number'].replace('.', '_')}"), "relevance_score": r.get("similarity", 0.0)} for r in vector_search_response.data]
                        return jsonify({"success": True, "query": query_text_rules, "rules": formatted_rules_vec, "total": len(formatted_rules_vec), "search_method": "database_vector_search"})
                
                # Fallback to text search in database if vector search not available/failed or no embeddings
                search_keywords = [k for k in query_text_rules.lower().split() if len(k) > 2] # Min keyword length
                if search_keywords:
                    # Build a text search query (e.g., using FTS if configured, or multiple ILIKEs)
                    # For simplicity, using ILIKE on 'text' field with first keyword. A real FTS is better.
                    # This is a simplified text search.
                    # search_condition = " & ".join(f"text.fts.{k}" for k in search_keywords) # Example for FTS
                    # For ILIKE, one might just use the first significant keyword or build ORs
                    # This is a very basic text search using first keyword
                    db_text_search_response = rag_system.db.client.from_("mtg_rules").select("*").ilike("text", f"%{search_keywords[0]}%").limit(limit_val * 2).execute() # Get more for potential client-side filtering
                    if db_text_search_response.data:
                        # Further client-side filtering might be needed if DB search is too broad
                        # For now, assume results are somewhat relevant
                        formatted_rules_text_db = [{"rule_number": r["rule_number"], "section": r.get("section","N/A"), "title": r.get("title","N/A"), "text": r["text"], "id": r.get("id", f"rule_{r['rule_number'].replace('.', '_')}")} for r in db_text_search_response.data[:limit_val]]
                        return jsonify({"success": True, "query": query_text_rules, "rules": formatted_rules_text_db, "total": len(formatted_rules_text_db), "search_method": "database_text_search"})
            except Exception as db_rule_search_error:
                logger.error(f"Database rule search error for query '{query_text_rules}': {db_rule_search_error}")
                # Fall through to in-memory search if DB search has issues
        
        # Fallback to In-Memory Search (if DB search fails or is not available)
        logger.info(f"Falling back to in-memory rule search for query: {query_text_rules}")
        if not rules_cache["loaded"]: load_rules_data() # Ensure cache is loaded
        
        cached_rules_data_list = rules_cache["rules_data"] # Renamed
        if not cached_rules_data_list:
             return jsonify({"success": True, "query": query_text_rules, "rules": [], "total":0, "search_method": "in_memory_no_data"})

        in_memory_results = [] # Renamed
        
        if exact_rule_num_param:
            in_memory_results = [r for r in cached_rules_data_list if r['rule_number'] == exact_rule_num_param][:limit_val]
        elif rag_system.embedding_available and rules_cache.get("rules_embeddings") is not None:
            rules_embeddings_np_cache = rules_cache["rules_embeddings"] # Renamed
            if rules_embeddings_np_cache is not None and rules_embeddings_np_cache.size > 0 and rules_embeddings_np_cache.shape[0] == len(cached_rules_data_list): # Ensure alignment
                query_embedding_rules_mem = rag_system.embedding_model.encode([query_text_rules]) # Renamed
                
                # FAISS search on cached embeddings
                dimension_rules_mem = rules_embeddings_np_cache.shape[1] # Renamed
                faiss_index_mem = faiss.IndexFlatIP(dimension_rules_mem) # Renamed
                
                # Normalize for cosine similarity (IP on normalized vectors)
                rules_embeddings_norm_mem = rules_embeddings_np_cache.copy() # Renamed
                faiss.normalize_L2(rules_embeddings_norm_mem)
                faiss_index_mem.add(rules_embeddings_norm_mem)
                
                query_embedding_norm_mem = query_embedding_rules_mem.copy() # Renamed
                faiss.normalize_L2(query_embedding_norm_mem)
                
                distances_mem, indices_mem = faiss_index_mem.search(query_embedding_norm_mem, limit_val) # Renamed
                
                for i_mem, idx_mem in enumerate(indices_mem[0]): # Renamed
                    if 0 <= idx_mem < len(cached_rules_data_list):
                        rule_obj_mem = cached_rules_data_list[idx_mem] # Renamed
                        rule_with_score_mem = dict(rule_obj_mem) # Renamed
                        rule_with_score_mem['relevance_score'] = float(distances_mem[0][i_mem])
                        in_memory_results.append(rule_with_score_mem)
            else: # Fallback if embeddings in cache are problematic
                logger.warning("In-memory rule search: Embeddings available but cache issue. Falling to keyword.")
                search_keywords_mem = query_text_rules.lower().split() # Renamed
                in_memory_results = [r for r in cached_rules_data_list if all(kw in r['text'].lower() for kw in search_keywords_mem)][:limit_val]
        else: # Fallback to keyword search if no embeddings
            search_keywords_mem_noemb = query_text_rules.lower().split() # Renamed
            in_memory_results = [r for r in cached_rules_data_list if all(kw in r['text'].lower() for kw in search_keywords_mem_noemb)][:limit_val]
        
        return jsonify({
            "success": True, "query": query_text_rules, "rules": in_memory_results,
            "total_found_in_memory": len(in_memory_results), # Be clear about count source
            "search_method": "in_memory"
        })
        
    except Exception as e_rules_search_main: # Renamed
        logger.error(f"General error searching rules for query '{query_text_rules}': {e_rules_search_main}")
        return jsonify({"success": False, "error": str(e_rules_search_main)}), 500


@app.route('/api/rag/rules-update', methods=['POST'])
def update_rules_endpoint_func(): # Renamed endpoint function
    """Update the rules data and embeddings in the database"""
    request_data_rules_update = request.json or {} # Renamed
    rules_source_url = request_data_rules_update.get('url') # Renamed (can be None)
    
    try:
        # Ensure DB table exists if DB is connected (update_rules_embeddings also calls import_rules_to_database which checks)
        # if rag_system.db.is_connected:
        #     create_rules_table_if_not_exists() # This is implicitly handled by import_rules_to_database
        
        update_success_status = update_rules_embeddings(rules_source_url) # Renamed, pass URL
        
        if update_success_status:
            db_rules_count_after_update = "N/A" # Renamed
            if rag_system.db.is_connected:
                try:
                    count_response_db = rag_system.db.client.from_("mtg_rules").select("id", count="exact", head=True).execute() # Renamed
                    db_rules_count_after_update = count_response_db.count if hasattr(count_response_db, 'count') else "Error"
                except Exception as e_count_rules_db: # Renamed
                    logger.error(f"Error checking rules count in DB after update: {e_count_rules_db}")
                    db_rules_count_after_update = "Error querying count"
            
            return jsonify({
                "success": True,
                "message": "Rules update process completed.", # Message reflects process, not just DB success
                "rules_in_cache_after_update": len(rules_cache["rules_data"]), # Renamed
                "rules_in_database_after_update": db_rules_count_after_update, # Renamed
                "embeddings_cached_after_update": rules_cache.get("rules_embeddings") is not None and rules_cache["rules_embeddings"].size > 0 if rules_cache.get("rules_embeddings") is not None else False
            })
        else:
            # update_rules_embeddings logs its own errors, so this indicates a failure in that process
            return jsonify({
                "success": False,
                "error": "Failed to update rules. Check server logs for details."
            }), 500
            
    except Exception as e_update_rules_main: # Renamed
        logger.error(f"Error in /api/rag/rules-update endpoint: {e_update_rules_main}")
        return jsonify({"success": False, "error": str(e_update_rules_main)}), 500


@app.route('/api/rag/mechanics-analysis', methods=['GET'])
def mechanics_analysis_endpoint(): # Renamed
    """Analyze the game mechanics for a specific card with improved rule display"""
    card_name_param_mech = request.args.get('card') # Renamed
    if not card_name_param_mech:
        return jsonify({"success": False, "error": "No card name provided in 'card' parameter."}), 400
    
    try:
        # Get full card data for the specified card
        card_data_for_analysis = get_card_data(card_name_param_mech) # Renamed
        if not card_data_for_analysis: # get_card_data now returns None if not found
            return jsonify({"success": False, "error": f"Card '{card_name_param_mech}' not found in database."}), 404
        
        card_oracle_text = card_data_for_analysis.get('oracle_text', '') # Renamed
        card_type_line_text = card_data_for_analysis.get('type_line', '') # Renamed
        
        # Extract mechanics from the card
        card_key_mechanics = get_card_mechanics(card_data_for_analysis) # Renamed
        
        # Load rules data if not already in cache
        if not rules_cache["loaded"] or not rules_cache["rules_data"]:
            load_rules_data()
        
        # Find rules related to the card's text and mechanics
        all_relevant_rules_for_card = [] # Renamed
        
        # 1. Direct text matching for important terms from oracle text
        important_card_text_terms = set() # Renamed
        if card_oracle_text:
            oracle_keywords = [kw for kw in re.findall(r'\b\w+\b', card_oracle_text.lower()) if len(kw) > 2] # Min length 3
            important_card_text_terms.update(oracle_keywords)
            # Common MTG phrases often tied to rules
            mtg_rule_phrases = [
                "enters the battlefield", "leaves the battlefield", "when you cast", "target creature",
                "whenever this creature attacks", "at the beginning of your upkeep", "draw a card",
                # Keywords often found in rules text that might not be formal mechanics
                "target player", "deals damage", "gains life", "put a counter", "search your library",
                "shuffle", "exile", "sacrifice", "tap", "untap", "counter target spell"
            ]
            for phrase_item in mtg_rule_phrases: # Renamed
                if phrase_item in card_oracle_text.lower():
                    important_card_text_terms.add(phrase_item) # Add phrase as a term
        
        cached_rules_for_analysis = rules_cache.get("rules_data", []) # Renamed
        seen_rule_ids_mech_analysis = set() # Renamed

        for rule_obj_analysis in cached_rules_for_analysis: # Renamed
            rule_text_lower_analysis = rule_obj_analysis['text'].lower() # Renamed
            relevance_score_text = 0
            matched_terms_list = [] # Renamed

            for term_to_match in important_card_text_terms: # Renamed
                if term_to_match in rule_text_lower_analysis:
                    relevance_score_text += (2 if len(term_to_match.split()) > 1 else 1) # Higher score for phrases
                    matched_terms_list.append(term_to_match)
            
            if relevance_score_text > 0 and rule_obj_analysis['id'] not in seen_rule_ids_mech_analysis:
                rule_copy_analysis = dict(rule_obj_analysis) # Renamed
                rule_copy_analysis['relevance_score'] = relevance_score_text
                rule_copy_analysis['source'] = 'direct_text_match'
                rule_copy_analysis['matched_terms'] = matched_terms_list
                all_relevant_rules_for_card.append(rule_copy_analysis)
                seen_rule_ids_mech_analysis.add(rule_obj_analysis['id'])

        # 2. Semantic search for rules related to oracle text (if embeddings are available)
        if rag_system.embedding_available and card_oracle_text and rules_cache.get("rules_embeddings") is not None:
            rules_embeddings_np_mech = rules_cache.get("rules_embeddings") # Renamed
            if rules_embeddings_np_mech is not None and rules_embeddings_np_mech.size > 0 and rules_embeddings_np_mech.shape[0] == len(cached_rules_for_analysis):
                try:
                    oracle_text_embedding = rag_system.embedding_model.encode([card_oracle_text]) # Renamed
                    
                    faiss_index_mech = faiss.IndexFlatIP(rules_embeddings_np_mech.shape[1]) # Renamed
                    rules_embeddings_norm_mech = rules_embeddings_np_mech.copy() # Renamed
                    faiss.normalize_L2(rules_embeddings_norm_mech)
                    faiss_index_mech.add(rules_embeddings_norm_mech)
                    
                    oracle_text_embedding_norm = oracle_text_embedding.copy() # Renamed
                    faiss.normalize_L2(oracle_text_embedding_norm)
                    
                    k_faiss_mech = min(10, rules_embeddings_np_mech.shape[0]) # Renamed
                    distances_semantic, indices_semantic = faiss_index_mech.search(oracle_text_embedding_norm, k_faiss_mech) # Renamed
                    
                    for i_sem, idx_sem in enumerate(indices_semantic[0]): # Renamed
                        if 0 <= idx_sem < len(cached_rules_for_analysis):
                            rule_obj_sem = cached_rules_for_analysis[idx_sem] # Renamed
                            if rule_obj_sem['id'] not in seen_rule_ids_mech_analysis: # Add if not already by text match
                                rule_copy_sem = dict(rule_obj_sem) # Renamed
                                rule_copy_sem['relevance_score'] = float(distances_semantic[0][i_sem]) # Semantic similarity score
                                rule_copy_sem['source'] = 'semantic_search'
                                all_relevant_rules_for_card.append(rule_copy_sem)
                                seen_rule_ids_mech_analysis.add(rule_obj_sem['id'])
                            else: # Rule already present, maybe update score if semantic is higher or add to sources
                                for existing_rule in all_relevant_rules_for_card:
                                    if existing_rule['id'] == rule_obj_sem['id']:
                                        existing_rule['relevance_score'] = max(existing_rule.get('relevance_score',0), float(distances_semantic[0][i_sem]))
                                        existing_rule['source'] += '+semantic_search' if 'semantic_search' not in existing_rule.get('source','') else ''
                                        break
                except Exception as e_semantic_mech:
                    logger.error(f"Error in semantic rule search for mechanics analysis of '{card_name_param_mech}': {e_semantic_mech}")
        
        # Sort all found rules by relevance score
        all_relevant_rules_for_card.sort(key=lambda x_sort_mech: x_sort_mech.get('relevance_score', 0), reverse=True)
        
        # Top N rules overall
        top_n_overall_rules = all_relevant_rules_for_card[:10] # Renamed

        # Group rules by the card's identified mechanics for display
        rules_grouped_by_mechanic = {} # Renamed
        for mechanic_on_card in card_key_mechanics: # Renamed
            rules_for_this_mechanic = [] # Renamed
            for rule_obj_grouping in top_n_overall_rules: # Search within top rules for mechanic relevance
                # A simple check if mechanic name (or part of it) is in rule text or title
                if mechanic_on_card.replace('_type','').replace('_interaction','').replace('_spell','').replace('_card','') in rule_obj_grouping['text'].lower() or \
                   mechanic_on_card.replace('_type','').replace('_interaction','').replace('_spell','').replace('_card','') in rule_obj_grouping.get('title','').lower():
                    rules_for_this_mechanic.append(rule_obj_grouping)
            if rules_for_this_mechanic:
                rules_grouped_by_mechanic[mechanic_on_card] = rules_for_this_mechanic
        
        # Simplified mechanic explanations (as in original)
        generic_mechanic_explanations = { # Renamed
            'creature_type': "This is a creature card that can attack and block.", 'flying': "This creature can only be blocked by creatures with flying or reach.",
            # ... (add more from original or expand)
        }
        
        # Create rule highlights for each identified mechanic on the card
        final_rule_highlights = {} # Renamed
        for mechanic_highlight_key in card_key_mechanics: # Renamed
            # Find the most relevant rule for this specific mechanic from the 'top_n_overall_rules'
            best_rule_for_mechanic = None
            highest_relevance_for_mechanic = -1
            for rule_highlight_candidate in top_n_overall_rules: # Renamed
                # Check if mechanic keyword is in this rule's text
                # Use a simplified version of mechanic key for matching text (e.g. 'creature_type' -> 'creature')
                simplified_mechanic_key = mechanic_highlight_key.split('_')[0]
                if simplified_mechanic_key in rule_highlight_candidate['text'].lower():
                    if rule_highlight_candidate.get('relevance_score',0) > highest_relevance_for_mechanic:
                        best_rule_for_mechanic = rule_highlight_candidate
                        highest_relevance_for_mechanic = rule_highlight_candidate.get('relevance_score',0)
            
            if best_rule_for_mechanic:
                # Helper to extract a relevant sentence or snippet
                def extract_relevant_snippet(rule_text_snippet, keyword_snippet): # Renamed
                    sentences = re.split(r'(?<=[.!?])\s+', rule_text_snippet) # Split into sentences
                    for sent in sentences:
                        if keyword_snippet.lower() in sent.lower():
                            return sent.strip()
                    return (rule_text_snippet[:150] + '...') if len(rule_text_snippet) > 150 else rule_text_snippet # Fallback
                
                final_rule_highlights[mechanic_highlight_key] = {
                    'rule_number': best_rule_for_mechanic['rule_number'],
                    'highlight': extract_relevant_snippet(best_rule_for_mechanic['text'], simplified_mechanic_key),
                    'relevance': highest_relevance_for_mechanic
                }
        
        # Format response
        analysis_result_payload = { # Renamed
            "success": True,
            "card_name": card_name_param_mech, # Renamed
            "oracle_text": card_oracle_text,
            "type_line": card_type_line_text,
            "identified_mechanics": card_key_mechanics,
            "generic_mechanic_explanations": generic_mechanic_explanations, # General explanations
            "rule_highlights_for_mechanics": final_rule_highlights,  # Specific rule snippets for each mechanic
            "top_overall_relevant_rules": top_n_overall_rules[:5],  # Show top 5 overall relevant rules
            "rules_grouped_by_card_mechanic": rules_grouped_by_mechanic # More detailed list per mechanic
        }
        
        return jsonify(analysis_result_payload)
    
    except Exception as e_mech_analysis: # Renamed
        logger.error(f"Error analyzing card mechanics for '{card_name_param_mech}': {e_mech_analysis}")
        return jsonify({"success": False, "error": str(e_mech_analysis)}), 500
       

@app.route('/api/rag/rules-interaction', methods=['POST'])
def rules_interaction_endpoint(): # Renamed
    """Find rules interactions between multiple cards with enhanced RAG capabilities"""
    try:
        request_data_interaction = request.json # Renamed
        if not request_data_interaction:
            return jsonify({"success": False, "error": "No data provided for rules interaction."}), 400
        
        input_card_names_list = request_data_interaction.get('cards', []) # Renamed
        if not isinstance(input_card_names_list, list) or len(input_card_names_list) < 2:
            return jsonify({"success": False, "error": "At least two card names (list) are required for interaction analysis."}), 400
        if not all(isinstance(name, str) for name in input_card_names_list):
             return jsonify({"success": False, "error": "All items in 'cards' list must be strings (card names)."}), 400

        # Get full card data for each card
        card_data_map_interaction = {} # Renamed
        all_oracle_texts_interaction = [] # Renamed
        
        for card_name_interact in input_card_names_list: # Renamed
            _card_data_obj_interact = get_card_data(card_name_interact) # Renamed
            if _card_data_obj_interact: # Returns None if not found
                card_data_map_interaction[card_name_interact] = _card_data_obj_interact
                oracle_text_val = _card_data_obj_interact.get('oracle_text', '') # Renamed
                if oracle_text_val:
                    all_oracle_texts_interaction.append(oracle_text_val)
            else:
                logger.warning(f"Rules Interaction: Card '{card_name_interact}' not found, will be excluded from analysis.")
        
        if len(card_data_map_interaction) < 2: # Not enough valid cards found for interaction
             return jsonify({"success": False, "error": f"Less than two valid cards found for interaction. Found: {list(card_data_map_interaction.keys())}" }), 400


        # Get mechanics for each found card
        aggregated_mechanics_all_cards = set() # Renamed
        mechanics_per_card_map = {} # Renamed
        for card_name_val, card_data_val_interact in card_data_map_interaction.items(): # Renamed
            card_mechanics_list_interact = get_card_mechanics(card_data_val_interact) # Renamed
            mechanics_per_card_map[card_name_val] = card_mechanics_list_interact
            aggregated_mechanics_all_cards.update(card_mechanics_list_interact)
        
        # Load rules data if needed for context
        if not rules_cache["loaded"] or not rules_cache["rules_data"]:
            load_rules_data()
        
        # Find rules potentially relevant to the combination of cards (e.g. based on all_oracle_texts_interaction)
        # This part is similar to mechanics_analysis for finding relevant rules
        relevant_rules_for_interaction = [] # Renamed
        # (This can be a complex step: find rules related to combined text, or individual card texts,
        # or based on the aggregated_mechanics_all_cards. For now, let's focus on interaction explanations.)
        # For a simple version, let's find rules related to the aggregated mechanics:
        if aggregated_mechanics_all_cards:
            relevant_rules_for_interaction = find_related_rules(list(aggregated_mechanics_all_cards))
            relevant_rules_for_interaction.sort(key=lambda x: x.get('relevance_score', 0), reverse=True) # Assuming find_related_rules might add a score

        # Find potential interactions by analyzing overlapping mechanics and direct text references
        interactions_found_list = [] # Renamed
        
        # Check for shared mechanics between pairs of cards
        card_names_processed_pairs = list(card_data_map_interaction.keys()) # Use only names of cards found
        for i_idx_card1, card1_name_val in enumerate(card_names_processed_pairs): # Renamed
            for j_idx_card2, card2_name_val in enumerate(card_names_processed_pairs): # Renamed
                if i_idx_card1 >= j_idx_card2:  # Avoid self-comparison and duplicate pairs
                    continue
                
                mechanics_card1 = mechanics_per_card_map.get(card1_name_val, []) # Renamed
                mechanics_card2 = mechanics_per_card_map.get(card2_name_val, []) # Renamed
                
                shared_mechanics_between_pair = set(mechanics_card1).intersection(set(mechanics_card2)) # Renamed
                if shared_mechanics_between_pair:
                    interaction_explanation = f"Cards '{card1_name_val}' and '{card2_name_val}' share common mechanics: {', '.join(shared_mechanics_between_pair)}."
                    # Add more detailed interaction logic based on specific shared mechanics if needed
                    # Example: if 'sacrifice' and 'token' are shared, suggest specific synergy.
                    interactions_found_list.append({
                        "type": "shared_mechanics",
                        "cards_involved": [card1_name_val, card2_name_val], # Renamed
                        "mechanics": list(shared_mechanics_between_pair),
                        "explanation": interaction_explanation,
                        # "detailed_explanation": "..." # Add more specific logic here
                    })
                
                # Check for direct card text interactions (e.g., one card mentions another's type or name)
                oracle1_interact = card_data_map_interaction[card1_name_val].get('oracle_text', '').lower() # Renamed
                oracle2_interact = card_data_map_interaction[card2_name_val].get('oracle_text', '').lower() # Renamed
                type1_interact = card_data_map_interaction[card1_name_val].get('type_line', '').lower() # Renamed
                type2_interact = card_data_map_interaction[card2_name_val].get('type_line', '').lower() # Renamed
                
                direct_text_interactions = [] # Renamed
                # Simple check: card1's text mentions card2's primary type (e.g. "creature")
                primary_type2 = type2_interact.split('—')[0].split()[-1] if type2_interact else None # e.g. "Creature" from "Legendary Creature - Human"
                if primary_type2 and primary_type2 in oracle1_interact:
                    direct_text_interactions.append(f"'{card1_name_val}' text mentions '{primary_type2}', which '{card2_name_val}' is.")
                primary_type1 = type1_interact.split('—')[0].split()[-1] if type1_interact else None
                if primary_type1 and primary_type1 in oracle2_interact:
                    direct_text_interactions.append(f"'{card2_name_val}' text mentions '{primary_type1}', which '{card1_name_val}' is.")
                
                if direct_text_interactions:
                    interactions_found_list.append({
                        "type": "direct_text_reference",
                        "cards_involved": [card1_name_val, card2_name_val],
                        "explanation": " ".join(direct_text_interactions)
                    })
        
        # Return the results
        return jsonify({
            "success": True,
            "input_cards_queried": input_card_names_list, # Renamed
            "cards_analyzed": list(card_data_map_interaction.keys()), # Actual cards found and used
            "mechanics_per_analyzed_card": mechanics_per_card_map, # Renamed
            "identified_interactions": interactions_found_list, # Renamed
            "potentially_relevant_rules": relevant_rules_for_interaction[:15]  # Limit to 15 for brevity
        })
    
    except Exception as e_rules_interact: # Renamed
        logger.error(f"Error in /api/rag/rules-interaction endpoint: {e_rules_interact}")
        return jsonify({"success": False, "error": str(e_rules_interact)}), 500


@app.route('/api/search-card', methods=['GET']) # For debugging primarily
def search_card_debug_endpoint(): # Renamed
    """Search for a card in the database - for debugging"""
    card_name_query_debug = request.args.get('name', '') # Renamed
    if not card_name_query_debug:
        return jsonify({"success": False, "error": "No card name provided for debug search."}), 400
    
    try:
        search_result_payload = { # Renamed
            "success": True,
            "query": card_name_query_debug,
            "exact_match_found": None, # Renamed
            "similar_matches_found": [] # Renamed
        }
        
        # Exact match first
        exact_match_response = rag_system.db.client.table("mtg_cards").select(
            "id, name, type_line, color_identity, oracle_text" # Key fields
        ).eq("name", card_name_query_debug).limit(1).execute()
        
        if exact_match_response.data:
            search_result_payload["exact_match_found"] = exact_match_response.data[0]
        else: # If no exact match, try case-insensitive and then partial
            ilike_response = rag_system.db.client.table("mtg_cards").select(
                "id, name, type_line, color_identity, oracle_text"
            ).ilike("name", card_name_query_debug).limit(5).execute() # ilike for exact name, different case
            
            if ilike_response.data:
                search_result_payload["similar_matches_found"] = ilike_response.data
            else: # If still no match, try broader partial match
                partial_match_response = rag_system.db.client.table("mtg_cards").select(
                    "id, name, type_line, color_identity, oracle_text"
                ).ilike("name", f"%{card_name_query_debug}%").limit(10).execute()
                if partial_match_response.data:
                    search_result_payload["similar_matches_found"] = partial_match_response.data
        
        return jsonify(search_result_payload)
        
    except Exception as e_search_card_debug: # Renamed
        logger.error(f"Error in debug search-card endpoint for '{card_name_query_debug}': {e_search_card_debug}")
        return jsonify({
            "success": False, "error": str(e_search_card_debug), "query": card_name_query_debug
        }), 500


@app.route('/api/card-stats', methods=['GET'])
def card_stats_endpoint(): # Renamed
    """Get database statistics"""
    try:
        # Get total card count
        # Using head=True with count can be more efficient if supported well by client/DB
        total_cards_response = rag_system.db.client.table("mtg_cards").select("id", count="exact", head=True).execute()
        total_cards_count = total_cards_response.count if hasattr(total_cards_response, 'count') and total_cards_response.count is not None else 0
        
        # Get sample cards
        sample_cards_response = rag_system.db.client.table("mtg_cards").select("name").limit(10).execute()
        sample_card_names = [card['name'] for card in sample_cards_response.data] if sample_cards_response.data else []
        
        # Check for Kozilek specifically (as in original)
        kozilek_variants_response = rag_system.db.client.table("mtg_cards").select("name").ilike("name", "%kozilek%").limit(10).execute() # Added limit
        kozilek_variant_names = [card['name'] for card in kozilek_variants_response.data] if kozilek_variants_response.data else []
        
        return jsonify({
            "success": True,
            "total_cards_in_database": total_cards_count, # Renamed for clarity
            "sample_card_names": sample_card_names, # Renamed
            "kozilek_variant_names_found": kozilek_variant_names, # Renamed
            "database_connection_active": rag_system.db.is_connected # Renamed
        })
        
    except Exception as e_card_stats: # Renamed
        logger.error(f"Error getting card stats: {e_card_stats}")
        return jsonify({"success": False, "error": str(e_card_stats)}), 500
    

@app.route('/api/rag/universal-search', methods=['POST'])
def enhanced_universal_search_post_endpoint(): # Renamed from original name to avoid conflict if script is re-run
    """Enhanced universal natural language search with proper card detection and synergy"""
    try:
        request_data_uni_post = request.json # Renamed
        if not request_data_uni_post:
            return jsonify({"success": False, "error": "No JSON data provided for universal search."}), 400
        
        query_text_uni = request_data_uni_post.get('query', '') # Renamed
        if not query_text_uni.strip(): # Check for empty or whitespace-only query
            return jsonify({"success": False, "error": "No query text provided."}), 400
        
        # Get context information (optional)
        query_context_uni = request_data_uni_post.get('context', {}) # Renamed
        if not isinstance(query_context_uni, dict): # Ensure context is a dict
            logger.warning("Universal search 'context' was not a dictionary. Ignoring context.")
            query_context_uni = {}
            
        # Process the query with enhanced handler (using the global instance or re-instantiating)
        # Using the global instance for consistency if it's meant to hold state (like loaded card names)
        search_result_uni = enhanced_search_handler_global_instance.process_universal_query(query_text_uni, query_context_uni) # Renamed
        
        return jsonify(search_result_uni)
        
    except Exception as e_uni_search_post: # Renamed
        logger.error(f"Error in POST /api/rag/universal-search endpoint: {e_uni_search_post}")
        return jsonify({"success": False, "error": str(e_uni_search_post), "original_query": request.json.get('query', '') if request.is_json else 'N/A'}), 500


@app.route('/api/rag/universal-search', methods=['GET'])
def enhanced_universal_search_get_endpoint(): # Renamed
    """Enhanced universal search via GET for compatibility"""
    query_text_uni_get = request.args.get('query', '') # Renamed
    if not query_text_uni_get.strip():
        return jsonify({"success": False, "error": "No query text provided in 'query' parameter."}), 400
    
    # Parse optional context from query parameters
    query_context_uni_get = {} # Renamed
    commander_context = request.args.get('commander') # Renamed
    budget_context_str = request.args.get('budget') # Renamed
    
    if commander_context:
        query_context_uni_get['commander'] = commander_context
    if budget_context_str:
        try:
            query_context_uni_get['budget'] = float(budget_context_str)
        except ValueError:
            logger.warning(f"Invalid 'budget' parameter in GET universal search: '{budget_context_str}'. Ignoring.")
            pass # Ignore invalid budget
    
    # Process with enhanced handler
    search_result_uni_get = enhanced_search_handler_global_instance.process_universal_query(query_text_uni_get, query_context_uni_get) # Renamed
    return jsonify(search_result_uni_get)


@app.route('/api/debug/check-embeddings', methods=['GET'])
def quick_embedding_check_endpoint(): # Renamed
    """Quick check of embedding dimensions from the database."""
    try:
        # Fetch a few card embeddings from the database
        # Select the raw embedding column to see its stored type and parse it
        db_emb_check_response = rag_system.db.client.table("mtg_cards").select(
            "id, name, text_embedding" # Ensure this is the correct column name for raw embeddings
        ).not_.is_("text_embedding", "null").limit(10).execute()
        
        if not db_emb_check_response.data:
             return jsonify({"success":True, "message": "No embeddings found in mtg_cards to check or DB error.", "data": []})

        dimension_counts_map = {} # Renamed
        embedding_samples_list = [] # Renamed
        
        for card_emb_item in db_emb_check_response.data: # Renamed
            raw_embedding_data = card_emb_item.get('text_embedding') # Renamed
            card_name_emb_check = card_emb_item.get('name', 'Unknown Card') # Renamed
            
            parsed_emb_check = parse_embedding_string(raw_embedding_data) # Use the robust parser
            
            dim_info = "Parse_Failed_or_Empty"
            if parsed_emb_check is not None and parsed_emb_check.size > 0:
                dim_info = str(parsed_emb_check.shape[0]) if parsed_emb_check.ndim == 1 else f"NDim_{parsed_emb_check.ndim}"
            
            dimension_counts_map[dim_info] = dimension_counts_map.get(dim_info, 0) + 1
            embedding_samples_list.append(f"Card: '{card_name_emb_check}', Parsed Dim: {dim_info}, Raw Type: {type(raw_embedding_data).__name__}, Raw Value Snippet: {str(raw_embedding_data)[:70]}")
        
        expected_dim_val = "Unknown" # Renamed
        if rag_system.embedding_available and hasattr(rag_system.embedding_model, 'get_sentence_embedding_dimension'):
            try: expected_dim_val = rag_system.embedding_model.get_sentence_embedding_dimension()
            except: pass # Some models might not have this method

        return jsonify({
            'success': True,
            'embedding_dimension_counts': dimension_counts_map, # Renamed
            'embedding_samples_checked': embedding_samples_list, # Renamed
            'expected_model_dimension': expected_dim_val
        })
        
    except Exception as e_emb_check: # Renamed
        logger.error(f"Error in /api/debug/check-embeddings: {e_emb_check}")
        return jsonify({'success': False, 'error': str(e_emb_check)})


@app.route('/api/rag/enhanced-search', methods=['POST'])
def enhanced_search_endpoint_func(): # Renamed from 'enhanced_search' to avoid conflict if file is run multiple times in some contexts
    """Enhanced search with improved filtering and categorization, uses universal query processor."""
    try:
        request_data_enh = request.json # Renamed
        if not request_data_enh:
            return jsonify({"success": False, "error": "No JSON data provided for enhanced search."}), 400
        
        query_text_enh = request_data_enh.get('query', '') # Renamed
        if not query_text_enh.strip():
            return jsonify({"success": False, "error": "No query text provided for enhanced search."}), 400
        
        filters_context_enh = request_data_enh.get('filters', {}) # Renamed (context for universal_query)
        if not isinstance(filters_context_enh, dict):
             logger.warning("Enhanced search 'filters' (context) was not a dictionary. Ignoring.")
             filters_context_enh = {}

        # Use the enhanced_search_handler's universal query processor
        search_result_enh = enhanced_search_handler_global_instance.process_universal_query(query_text_enh, filters_context_enh) # Renamed
        
        return jsonify(search_result_enh) # process_universal_query already formats the response structure
        
    except Exception as e_enh_search: # Renamed
        logger.error(f"Error in /api/rag/enhanced-search endpoint: {e_enh_search}")
        return jsonify({"success": False, "error": str(e_enh_search), "original_query": request.json.get('query', '') if request.is_json else 'N/A'}), 500


@app.route('/api/rag/card-synergies', methods=['POST'])
def card_synergies_endpoint_func(): # Renamed
    """Find cards that synergize with seed cards using enhanced algorithm"""
    try:
        request_data_card_syn = request.json # Renamed
        if not request_data_card_syn:
            return jsonify({"success": False, "error": "No JSON data provided for card synergies."}), 400
        
        seed_cards_list_syn = request_data_card_syn.get('seed_cards', []) # Renamed
        if not isinstance(seed_cards_list_syn, list) or not seed_cards_list_syn:
            return jsonify({"success": False, "error": "No 'seed_cards' (list) provided or list is empty."}), 400
        if not all(isinstance(name, str) for name in seed_cards_list_syn):
             return jsonify({"success": False, "error": "All items in 'seed_cards' list must be strings (card names)."}), 400
        
        try:
            top_k_syn_val = min(int(request_data_card_syn.get('top_k', 15)), 50) # Renamed, cap at 50
        except ValueError:
             return jsonify({"success": False, "error": "'top_k' must be an integer."}), 400

        # Calculate synergies using the (now robust) calculate_card_synergies function
        synergistic_cards_result_syn = calculate_card_synergies(seed_cards_list_syn, top_k_syn_val) # Renamed
        
        # Format results for enhanced response (already done well by calculate_card_synergies structure)
        # We just need to ensure the items have 'card', 'total_score', and 'explanations'
        formatted_synergy_results = [] # Renamed
        for item_data_syn in synergistic_cards_result_syn: # Renamed
            card_obj_syn = item_data_syn.get('card', {}) # Renamed
            formatted_synergy_results.append({
                "card": card_obj_syn, # Includes all card details
                "synergy_score": item_data_syn.get("total_score", 0.0),
                "synergy_type": "enhanced_calculation", # Clarify source of synergy score
                "synergy_reason_summary": "; ".join(item_data_syn.get("explanations", ["General synergy."])) # Renamed
            })
        
        return jsonify({
            "success": True,
            "seed_cards_queried": seed_cards_list_syn, # Renamed
            "synergistic_cards_found": formatted_synergy_results # Renamed
        })
        
    except Exception as e_card_syn_endpoint: # Renamed
        logger.error(f"Error in /api/rag/card-synergies endpoint: {e_card_syn_endpoint}")
        return jsonify({"success": False, "error": str(e_card_syn_endpoint)}), 500


@app.route('/api/rag/combo-finder', methods=['POST'])
def combo_finder_endpoint(): # Renamed
    """Find potential combos and interactions between cards"""
    try:
        request_data_combo = request.json # Renamed
        if not request_data_combo:
            return jsonify({"success": False, "error": "No JSON data provided for combo finder."}), 400
        
        input_cards_for_combo = request_data_combo.get('cards', []) # Renamed
        if not isinstance(input_cards_for_combo, list) or len(input_cards_for_combo) < 2 :
            return jsonify({"success": False, "error": "At least two 'cards' (list of names) required for combo analysis."}), 400
        if not all(isinstance(name, str) for name in input_cards_for_combo):
             return jsonify({"success": False, "error": "All items in 'cards' list must be strings (card names)."}), 400

        # max_combo_size_param = request_data_combo.get('max_combo_size', 3) # Renamed, not used in current logic
        include_enablers_param = request_data_combo.get('include_enablers', True) # Renamed
        
        # Analyze each card for its mechanics to inform combo potential
        card_analysis_for_combo_map = {} # Renamed
        valid_cards_for_combo_analysis = [] # Store names of cards successfully fetched
        for card_name_combo_item in input_cards_for_combo: # Renamed
            _card_data_obj_combo = get_card_data(card_name_combo_item) # Renamed
            if _card_data_obj_combo: # If card data was found
                valid_cards_for_combo_analysis.append(card_name_combo_item)
                _mechanics_list_combo = get_card_mechanics(_card_data_obj_combo) # Renamed
                card_analysis_for_combo_map[card_name_combo_item] = {
                    "status": "analyzed", # Renamed
                    "mechanics": _mechanics_list_combo,
                    "synergy_potential_rating": "High" if len(_mechanics_list_combo) > 3 else ("Medium" if len(_mechanics_list_combo) > 1 else "Low"), # Adjusted rating
                    "potential_combo_roles": [] # Placeholder for more advanced role analysis
                }
            else:
                card_analysis_for_combo_map[card_name_combo_item] = {"status": "card_not_found"}

        if len(valid_cards_for_combo_analysis) < 2:
            return jsonify({
                "success": False, 
                "error": "Not enough valid cards found to perform combo analysis.",
                "input_cards": input_cards_for_combo,
                "card_analysis": card_analysis_for_combo_map
            }), 400
        
        # Find synergistic cards for the valid input cards (these could be combo pieces or support)
        # Use only valid_cards_for_combo_analysis for synergy calculation
        synergistic_support_cards = calculate_card_synergies(valid_cards_for_combo_analysis, count=20) # Renamed
        
        # Identify potential combos (simplified placeholder logic from original)
        # This needs significant domain knowledge and graph analysis for real combo finding.
        identified_potential_combos = [] # Renamed
        aggregated_mechanics_for_combo = set() # Renamed
        for card_name_agg_mech in valid_cards_for_combo_analysis: # Renamed
            if card_analysis_for_combo_map[card_name_agg_mech]["status"] == "analyzed":
                aggregated_mechanics_for_combo.update(card_analysis_for_combo_map[card_name_agg_mech]["mechanics"])

        if "sacrifice" in aggregated_mechanics_for_combo and ("token" in aggregated_mechanics_for_combo or "creature_type" in aggregated_mechanics_for_combo):
            identified_potential_combos.append({
                "combo_type": "Sacrifice-Token Engine", # Renamed
                "description": "Potential synergy involving creating tokens/creatures and sacrificing them for value.",
                "involved_mechanics": ["sacrifice", "token/creature_type"]
            })
        if "graveyard_interaction" in aggregated_mechanics_for_combo:
            identified_potential_combos.append({
                "combo_type": "Graveyard Value Engine", # Renamed
                "description": "Potential synergy using the graveyard as a resource or for recurring effects.",
                "involved_mechanics": ["graveyard_interaction"]
            })
        # Add more sophisticated combo detection logic here based on patterns of mechanics across multiple input cards.
        
        # Suggest enabler card types if requested
        enabler_card_archetypes = [] # Renamed
        if include_enablers_param:
            enabler_card_archetypes = [
                {"archetype_name": "Tutors/Card Search", "role_type": "search_enabler", "reason": "Helps find specific combo pieces."}, # Renamed
                {"archetype_name": "Protection Spells/Abilities", "role_type": "protection_enabler", "reason": "Protects key combo pieces from removal."},
                {"archetype_name": "Mana Acceleration/Ramp", "role_type": "mana_enabler", "reason": "Enables casting expensive combo pieces or multiple spells sooner."}
            ]
        
        return jsonify({
            "success": True,
            "input_cards_queried_combo": input_cards_for_combo, # Renamed
            "card_by_card_analysis": card_analysis_for_combo_map, # Renamed
            "identified_potential_combos": identified_potential_combos,
            "synergistic_support_cards": synergistic_support_cards, # Cards that synergize with the input set
            "suggested_enabler_archetypes": enabler_card_archetypes # Renamed
        })
        
    except Exception as e_combo_finder: # Renamed
        logger.error(f"Error in /api/rag/combo-finder endpoint: {e_combo_finder}")
        return jsonify({"success": False, "error": str(e_combo_finder)}), 500


# Final log message to confirm module loading
logger.info("UpdateAPIServer module loaded. Enhanced search, robust embedding parsing, and other functionalities active.")