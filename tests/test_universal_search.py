import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.enhanced_universal_search import EnhancedUniversalSearchHandler

class DummyDB:
    is_connected = False
    client = None

class DummyRAG:
    def __init__(self):
        self.db = DummyDB()
        self.embedding_model = None
        self.embedding_available = False
        self.text_index = None
    def search_cards_by_keyword(self, keyword, top_k=3):
        return []
    def retrieve_cards_by_text(self, text, top_k=20):
        return []

class DummyRAGStrategy(DummyRAG):
    def generate_deck_recommendation(self, strategy, commander=None, budget=None, colors=None):
        return {'cards': [{'name': 'Rampant Growth'}], 'strategy': strategy, 'colors': colors}

class TestColorExtraction(unittest.TestCase):
    def setUp(self):
        self.handler = EnhancedUniversalSearchHandler(DummyRAG())

    def test_color_abbreviations(self):
        constraints = self.handler.extract_constraints("Looking for a UB artifact")
        self.assertIn('U', constraints['colors'])
        self.assertIn('B', constraints['colors'])

    def test_multi_color_abbrev(self):
        constraints = self.handler.extract_constraints("Need a WUG ramp deck")
        self.assertEqual(set(constraints['colors']), {'W','U','G'})
        self.assertIn('ramp', constraints['strategies'])

class TestStrategySearch(unittest.TestCase):
    def setUp(self):
        self.handler = EnhancedUniversalSearchHandler(DummyRAGStrategy())

    def test_strategy_search_process(self):
        result = self.handler.process_universal_query('Need a WUG ramp deck')
        self.assertTrue(result.get('success'))
        self.assertEqual(result.get('query_type'), 'strategy_search')
        self.assertIn('deck', result)
        self.assertEqual(result['deck']['strategy'], 'ramp')

if __name__ == '__main__':
    unittest.main()
