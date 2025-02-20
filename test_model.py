import unittest
import pandas as pd
from model import collaborative_filtering, content_based_filtering, hybrid_recommendation

# Load test data
users = pd.read_csv('users.csv')
products = pd.read_csv('products.csv')
purchases = pd.read_csv('purchases.csv')
browsing_history = pd.read_csv('browsing_history.csv')

class TestRecommendations(unittest.TestCase):
    def test_collaborative_filtering(self):
        recs = collaborative_filtering(1, purchases, products)
        self.assertFalse(recs.empty, "Collaborative filtering should return recommendations")
        self.assertTrue(all(p not in [1, 2, 3] for p in recs['product_id']), "Should exclude user purchases")

    def test_content_based_filtering(self):
        recs = content_based_filtering(1, purchases, browsing_history, products)
        self.assertFalse(recs.empty, "Content-based filtering should return recommendations")
        self.assertTrue(all(p not in [4, 5] for p in recs['product_id']), "Should exclude browsed items")

    def test_hybrid_recommendation(self):
        recs = hybrid_recommendation(1, purchases, browsing_history, products)
        self.assertFalse(recs.empty, "Hybrid should return recommendations")
        self.assertTrue(any(recs['source'] == 'Collaborative Filtering'), "Should include collaborative recs")
        self.assertTrue(any(recs['source'] == 'Content-Based Filtering'), "Should include content-based recs")
        self.assertTrue(all(p not in [1, 2, 3, 4, 5] for p in recs['product_id']), "Should exclude interacted items")

if __name__ == '__main__':
    unittest.main()