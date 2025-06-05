# enhanced_rag_endpoints.py - FIXED VERSION
from flask import request, jsonify
import logging
import json

# Import the Flask app
from src.api_server import app

logger = logging.getLogger("EnhancedRAGEndpoints")

# Initialize the enhanced RAG system with better error handling
enhanced_rag = None
try:
    from src.rag_system import MTGRetrievalSystem
    try:
        # Try to import the enhanced system
        from src.enhanced_rag import EnhancedMTGRAGSystem
        base_rag = MTGRetrievalSystem(embedding_model_name="all-MiniLM-L6-v2")
        enhanced_rag = EnhancedMTGRAGSystem(base_rag)
        logger.info("Enhanced RAG system initialized successfully")
        print("✓ Enhanced RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Error creating enhanced RAG system: {e}")
        print(f"✗ Error creating enhanced RAG system: {e}")
        enhanced_rag = None
except Exception as e:
    logger.error(f"Error importing RAG dependencies: {e}")
    print(f"✗ Error importing RAG dependencies: {e}")
    enhanced_rag = None

@app.route('/api/rag/enhanced-search', methods=['POST'])
def enhanced_card_search():
    """Enhanced card search with multiple strategies and filtering"""
    print("Enhanced search endpoint called")  # Debug
    
    if not enhanced_rag:
        return jsonify({
            'success': False,
            'error': 'Enhanced RAG system not available'
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Query parameter is required'
            }), 400
        
        query = data['query']
        filters = data.get('filters', {})
        top_k = data.get('top_k', 20)
        
        # Validate top_k
        top_k = min(max(1, int(top_k)), 50)
        
        # Perform enhanced search
        result = enhanced_rag.enhanced_card_search(query, filters, top_k)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in enhanced card search: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rag/card-synergies', methods=['POST'])
def find_card_synergies():
    """Find cards that synergize with given seed cards"""
    print("Card synergies endpoint called")  # Debug
    
    if not enhanced_rag:
        return jsonify({
            'success': False,
            'error': 'Enhanced RAG system not available'
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'seed_cards' not in data:
            return jsonify({
                'success': False,
                'error': 'seed_cards parameter is required'
            }), 400
        
        seed_cards = data['seed_cards']
        if not isinstance(seed_cards, list) or not seed_cards:
            return jsonify({
                'success': False,
                'error': 'seed_cards must be a non-empty list'
            }), 400
        
        top_k = data.get('top_k', 15)
        top_k = min(max(1, int(top_k)), 30)
        
        # Find synergies
        result = enhanced_rag.find_card_synergies(seed_cards, top_k)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error finding card synergies: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rag/mechanics-analysis', methods=['GET'])
def analyze_card_mechanics():
    """Analyze a card's mechanics and find related cards"""
    print("Mechanics analysis endpoint called")  # Debug
    
    if not enhanced_rag:
        return jsonify({
            'success': False,
            'error': 'Enhanced RAG system not available'
        }), 500
    
    try:
        card_name = request.args.get('card')
        if not card_name:
            return jsonify({
                'success': False,
                'error': 'card parameter is required'
            }), 400
        
        # Analyze card mechanics
        result = enhanced_rag.analyze_card_mechanics(card_name)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing card mechanics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rag/combo-finder', methods=['POST'])
def find_card_combos():
    """Find potential combos and interactions between cards"""
    print("Combo finder endpoint called")  # Debug
    
    if not enhanced_rag:
        return jsonify({
            'success': False,
            'error': 'Enhanced RAG system not available'
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'cards' not in data:
            return jsonify({
                'success': False,
                'error': 'cards parameter is required'
            }), 400
        
        cards = data['cards']
        if not isinstance(cards, list) or len(cards) < 2:
            return jsonify({
                'success': False,
                'error': 'At least 2 cards are required for combo finding'
            }), 400
        
        # Find synergies between the cards
        synergy_result = enhanced_rag.find_card_synergies(cards, 20)
        
        if not synergy_result['success']:
            return jsonify(synergy_result)
        
        # Analyze each card for combo potential
        combo_analysis = {}
        for card_name in cards:
            analysis = enhanced_rag.analyze_card_mechanics(card_name)
            if analysis['success']:
                combo_analysis[card_name] = analysis
        
        # Simple combo detection
        potential_combos = []
        for card_name, analysis in combo_analysis.items():
            interactions = analysis.get('interactions', [])
            for interaction in interactions:
                if interaction['type'] in ['activated_ability', 'etb', 'sacrifice']:
                    potential_combos.append({
                        'cards': [card_name],
                        'type': interaction['type'],
                        'description': f"{card_name} has {interaction['type']} that could combo"
                    })
        
        # Enabler cards
        enabler_cards = [
            {'name': 'Mana Vault', 'type': 'mana_acceleration', 'reason': 'Helps cast expensive spells'},
            {'name': 'Counterspell', 'type': 'protection', 'reason': 'Protects combo pieces'},
            {'name': 'Demonic Tutor', 'type': 'tutor', 'reason': 'Finds combo pieces'}
        ]
        
        result = {
            'success': True,
            'input_cards': cards,
            'combo_analysis': combo_analysis,
            'synergistic_cards': synergy_result.get('synergistic_cards', []),
            'potential_combos': potential_combos,
            'enabler_cards': enabler_cards if data.get('include_enablers', True) else []
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error finding card combos: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rag/enhanced-status', methods=['GET'])
def enhanced_rag_status():
    """Check enhanced RAG system status"""
    return jsonify({
        'enhanced_rag_available': enhanced_rag is not None,
        'endpoints': [
            '/api/rag/enhanced-search',
            '/api/rag/card-synergies', 
            '/api/rag/mechanics-analysis',
            '/api/rag/universal-search',
            '/api/rag/combo-finder'
        ]
    })

# Print confirmation that endpoints are being registered
print("✓ Enhanced RAG endpoints registered:")
print("  - POST /api/rag/enhanced-search")
print("  - POST /api/rag/card-synergies")
print("  - GET  /api/rag/mechanics-analysis")
print("  - POST /api/rag/universal-search")
print("  - POST /api/rag/combo-finder")
print("  - GET  /api/rag/enhanced-status")