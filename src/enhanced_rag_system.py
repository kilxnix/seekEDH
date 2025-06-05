# src/enhanced_rag_system.py - Enhanced RAG System with Image Support
import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("EnhancedRAGSystem")

class EnhancedMTGRetrievalSystem:
    """Enhanced RAG system that includes image URLs in search results"""
    
    def __init__(self, original_rag_system, image_manager=None):
        self.rag = original_rag_system
        self.image_manager = image_manager
        
        # Initialize image manager if not provided
        if self.image_manager is None:
            try:
                from src.image_manager import MTGImageManager
                self.image_manager = MTGImageManager(self.rag.db)
                logger.info("Image manager initialized successfully")
            except ImportError:
                logger.warning("Image manager not available")
                self.image_manager = None
            except Exception as e:
                logger.error(f"Error initializing image manager: {e}")
                self.image_manager = None
    
    def enhance_cards_with_images(self, cards: List[Dict[str, Any]], 
                                 image_size: str = 'normal',
                                 prefer_storage: bool = True) -> List[Dict[str, Any]]:
        """Enhance card data with image URLs"""
        if not self.image_manager:
            logger.debug("Image manager not available, returning cards without image enhancement")
            return cards
        
        enhanced_cards = []
        
        for card in cards:
            try:
                # Create a copy to avoid modifying original
                enhanced_card = card.copy()
                
                card_name = card.get('name')
                if card_name:
                    # Get image information
                    image_info = self.image_manager.get_card_image_info(card_name, image_size)
                    
                    if image_info:
                        # Add image URLs to card data
                        enhanced_card['image_info'] = {
                            'image_url': self._get_best_image_url(image_info, prefer_storage),
                            'image_status': image_info.status,
                            'available_sources': self._get_available_sources(image_info),
                            'file_size': image_info.file_size
                        }
                        
                        # Also add direct image_url field for convenience
                        enhanced_card['image_url'] = enhanced_card['image_info']['image_url']
                    else:
                        # No image available
                        enhanced_card['image_info'] = {
                            'image_url': None,
                            'image_status': 'not_available',
                            'available_sources': [],
                            'file_size': None
                        }
                        enhanced_card['image_url'] = None
                
                enhanced_cards.append(enhanced_card)
                
            except Exception as e:
                logger.error(f"Error enhancing card {card.get('name', 'unknown')} with image: {e}")
                # Add card without image info if enhancement fails
                enhanced_card = card.copy()
                enhanced_card['image_url'] = None
                enhanced_card['image_info'] = {'image_status': 'error', 'error': str(e)}
                enhanced_cards.append(enhanced_card)
        
        return enhanced_cards
    
    def _get_best_image_url(self, image_info, prefer_storage: bool = True) -> Optional[str]:
        """Get the best available image URL based on preferences"""
        if prefer_storage and image_info.storage_url:
            return image_info.storage_url
        elif image_info.local_path and os.path.exists(image_info.local_path):
            # Convert local path to API endpoint URL
            return f"/api/images/serve/{image_info.card_id}?size={image_info.size}"
        elif image_info.original_url:
            return image_info.original_url
        else:
            return None
    
    def _get_available_sources(self, image_info) -> List[str]:
        """Get list of available image sources"""
        sources = []
        if image_info.storage_url:
            sources.append('storage')
        if image_info.local_path and os.path.exists(image_info.local_path):
            sources.append('local')
        if image_info.original_url:
            sources.append('original')
        return sources
    
    # Enhanced search methods that include images
    def retrieve_cards_by_text_with_images(self, query_text: str, top_k: int = 10,
                                          image_size: str = 'normal',
                                          prefer_storage: bool = True) -> List[Dict[str, Any]]:
        """Retrieve cards by text similarity including image URLs"""
        try:
            cards = self.rag.retrieve_cards_by_text(query_text, top_k)
            return self.enhance_cards_with_images(cards, image_size, prefer_storage)
        except Exception as e:
            logger.error(f"Error in retrieve_cards_by_text_with_images: {e}")
            return []
    
    def search_cards_by_keyword_with_images(self, keyword: str, top_k: int = 10,
                                           image_size: str = 'normal',
                                           prefer_storage: bool = True) -> List[Dict[str, Any]]:
        """Search cards by keyword including image URLs"""
        try:
            cards = self.rag.search_cards_by_keyword(keyword, top_k)
            return self.enhance_cards_with_images(cards, image_size, prefer_storage)
        except Exception as e:
            logger.error(f"Error in search_cards_by_keyword_with_images: {e}")
            return []
    
    def retrieve_cards_by_price_with_images(self, query_card: str, top_k: int = 10,
                                           image_size: str = 'normal',
                                           prefer_storage: bool = True) -> List[Dict[str, Any]]:
        """Retrieve cards by price similarity including image URLs"""
        try:
            cards = self.rag.retrieve_cards_by_price(query_card, top_k)
            return self.enhance_cards_with_images(cards, image_size, prefer_storage)
        except Exception as e:
            logger.error(f"Error in retrieve_cards_by_price_with_images: {e}")
            return []
    
    def generate_deck_recommendation_with_images(self, strategy: str, 
                                               commander: Optional[str] = None,
                                               budget: Optional[float] = None, 
                                               colors: Optional[List[str]] = None,
                                               image_size: str = 'normal',
                                               prefer_storage: bool = True) -> Dict[str, Any]:
        """Generate deck recommendation including image URLs"""
        try:
            deck_rec = self.rag.generate_deck_recommendation(strategy, commander, budget, colors)
            
            if 'cards' in deck_rec and isinstance(deck_rec['cards'], list):
                deck_rec['cards'] = self.enhance_cards_with_images(
                    deck_rec['cards'], image_size, prefer_storage
                )
            
            return deck_rec
        except Exception as e:
            logger.error(f"Error in generate_deck_recommendation_with_images: {e}")
            return {"error": str(e)}
    
    def apply_search_filters(self, cards: List[Dict], filters: Dict) -> List[Dict]:
        """Apply search filters to card results"""
        if not filters:
            return cards
        
        filtered = []
        
        for card in cards:
            try:
                # Color filter
                colors = filters.get('colors', [])
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
                
                # Type filter
                card_type = filters.get('card_type') or filters.get('type')
                if card_type:
                    type_line = card.get('type_line', '')
                    if card_type not in type_line:
                        continue
                
                # Price filter
                max_price = filters.get('max_price')
                if max_price:
                    price = card.get('prices_usd', 0) or 0
                    try:
                        if float(price) > max_price:
                            continue
                    except (ValueError, TypeError):
                        pass  # Include cards with unknown prices
                
                filtered.append(card)
                
            except Exception as e:
                logger.error(f"Error applying filters to card {card.get('name', 'unknown')}: {e}")
                # Include card if filter application fails
                filtered.append(card)
        
        return filtered
    
    def enhanced_search(self, query: str, filters: Dict = None, top_k: int = 20,
                       image_size: str = 'normal', include_images: bool = True) -> Dict[str, Any]:
        """Enhanced search with filtering and image support"""
        try:
            # Perform search with images if requested
            if include_images:
                matching_cards = self.retrieve_cards_by_text_with_images(query, top_k, image_size)
            else:
                matching_cards = self.rag.retrieve_cards_by_text(query, top_k)
            
            # Apply filters if provided
            if filters:
                filtered_cards = self.apply_search_filters(matching_cards, filters)
            else:
                filtered_cards = matching_cards
            
            return {
                "success": True,
                "query": query,
                "total_found": len(matching_cards),
                "returned": len(filtered_cards),
                "cards": filtered_cards,
                "search_metadata": {
                    "image_size": image_size,
                    "includes_images": include_images,
                    "filters_applied": list(filters.keys()) if filters else [],
                    "avg_relevance": sum(card.get('similarity', 0) for card in filtered_cards) / len(filtered_cards) if filtered_cards else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced_search: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def get_card_with_image(self, card_name: str, image_size: str = 'normal') -> Optional[Dict]:
        """Get a single card with image information"""
        try:
            if not self.rag.db.is_connected:
                return None
            
            # Get card data
            response = self.rag.db.client.table("mtg_cards").select(
                "id, name, oracle_text, type_line, mana_cost, colors, color_identity, prices_usd"
            ).eq("name", card_name).limit(1).execute()
            
            if not response.data:
                return None
            
            card = response.data[0]
            
            # Enhance with image
            enhanced_cards = self.enhance_cards_with_images([card], image_size)
            return enhanced_cards[0] if enhanced_cards else card
            
        except Exception as e:
            logger.error(f"Error getting card with image: {e}")
            return None