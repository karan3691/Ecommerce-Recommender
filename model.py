import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def collaborative_filtering(user_id, purchases, products):
    logger.debug(f"Collaborative Filtering for user_id: {user_id}")
    user_purchases = purchases[purchases['user_id'] == user_id]['product_id'].unique()
    logger.debug(f"User purchases: {user_purchases}")
    other_users = purchases[purchases['product_id'].isin(user_purchases) & 
                            (purchases['user_id'] != user_id)]['user_id'].unique()
    other_purchases = purchases[purchases['user_id'].isin(other_users)]

    product_counts = other_purchases['product_id'].value_counts()
    recommendations = products[products['product_id'].isin(product_counts.index) & 
                               ~products['product_id'].isin(user_purchases)].copy()
    
    # Enhanced scoring: frequency * rating, normalized
    recommendations['score'] = (recommendations['product_id'].map(product_counts).fillna(0) * 
                                recommendations['rating']) / product_counts.max()
    recommendations['source'] = 'Collaborative Filtering'
    logger.debug(f"Collaborative recommendations:\n{recommendations[['product_id', 'score', 'source']]}")
    return recommendations.sort_values(by='score', ascending=False)

def content_based_filtering(user_id, purchases, browsing_history, products):
    logger.debug(f"Content-Based Filtering for user_id: {user_id}")
    user_history = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()
    logger.debug(f"User browsing history: {user_history}")
    user_products = products[products['product_id'].isin(user_history)]

    if not user_products.empty and 'category' in products.columns:
        recommendations = products[products['category'].isin(user_products['category']) & 
                                   ~products['product_id'].isin(user_history)].copy()
        # Enhanced scoring: average rating of browsed products in category
        avg_rating = user_products['rating'].mean()
        recommendations['score'] = recommendations['rating'] / 5.0 * avg_rating
    else:
        recommendations = pd.DataFrame(columns=['product_id', 'product_name', 'price', 'rating', 'score', 'source'])
        logger.debug("No content-based recommendations.")
    
    recommendations['source'] = 'Content-Based Filtering'
    logger.debug(f"Content-based recommendations:\n{recommendations[['product_id', 'score', 'source']]}")
    return recommendations

def hybrid_recommendation(user_id, purchases, browsing_history, products):
    logger.debug(f"Hybrid Recommendation for user_id: {user_id}")
    
    user_purchases = purchases[purchases['user_id'] == user_id]['product_id'].unique()
    user_browsed = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()
    user_history = set(user_purchases).union(user_browsed)
    logger.debug(f"User history (purchases + browsed): {user_history}")

    collab_recs = collaborative_filtering(user_id, purchases, products)
    content_recs = content_based_filtering(user_id, purchases, browsing_history, products)

    all_recommendations = pd.concat([collab_recs, content_recs], ignore_index=True)
    logger.debug(f"Combined recommendations:\n{all_recommendations[['product_id', 'score', 'source']]}")

    if all_recommendations.empty:
        logger.debug("No recommendations; adding popular products.")
        popular_products = purchases['product_id'].value_counts().head(3).index
        all_recommendations = products[products['product_id'].isin(popular_products) & 
                                      ~products['product_id'].isin(user_history)].copy()
        all_recommendations['score'] = 0.5
        all_recommendations['source'] = 'Popular Products'

    final_recommendations = all_recommendations.sort_values(by='score', ascending=False) \
                                               .drop_duplicates(subset=['product_id'], keep='first')
    logger.debug(f"Final hybrid recommendations:\n{final_recommendations[['product_id', 'score', 'source']]}")
    
    return final_recommendations