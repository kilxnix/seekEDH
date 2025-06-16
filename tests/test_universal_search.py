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


class TestScryfallFallback(unittest.TestCase):
    def setUp(self):
        self.handler = EnhancedUniversalSearchHandler(DummyRAG())

    def _mock_response(self):
        return {
            "data": [
                {
                    "id": "1",
                    "name": "Fake Card",
                    "type_line": "Creature",
                    "oracle_text": "",
                    "color_identity": ["G"],
                    "prices": {"usd": "0.10"}
                }
            ]
        }

    def test_fallback_general_search(self):
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._mock_response()

        with patch('src.enhanced_universal_search.requests.get', return_value=mock_resp):
            result = self.handler._fallback_general_search('nonsense query')
        self.assertTrue(result['success'])
        self.assertGreater(len(result['cards']), 0)

if __name__ == '__main__':
    unittest.main()
