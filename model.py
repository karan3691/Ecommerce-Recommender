import pandas as pd

def collaborative_filtering(user_id, purchases, browsing_history, products):
    user_purchases = purchases[purchases['user_id'] == user_id]['product_id'].unique()
    other_users = purchases[purchases['product_id'].isin(user_purchases)]['user_id'].unique()
    other_purchases = purchases[purchases['user_id'].isin(other_users)]['product_id'].unique()

    # Recommendations that exclude the user's own purchases
    recommendations = products[~products['product_id'].isin(user_purchases) & 
                                products['product_id'].isin(other_purchases)]
    recommendations['source'] = 'Collaborative Filtering'
    return recommendations

def content_based_filtering(user_id, purchases, browsing_history, products):
    user_history = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()
    user_products = products[products['product_id'].isin(user_history)]

    if not user_products.empty:
        recommendations = products[products['category'].isin(user_products['category']) &
                                    ~products['product_id'].isin(user_history)]
    else:
        recommendations = pd.DataFrame()  # Return empty if no products in user history

    recommendations['source'] = 'Content-Based Filtering'
    return recommendations

def hybrid_recommendation(user_id, purchases, browsing_history, products):
    collaborative_recommendations = collaborative_filtering(user_id, purchases, browsing_history, products)
    content_based_recommendations = content_based_filtering(user_id, purchases, browsing_history, products)
    
    # Combine collaborative and content-based recommendations
    hybrid_recommendations = pd.concat([collaborative_recommendations, content_based_recommendations]).drop_duplicates('prduct_id')

    # Unseen products (products user hasn't interacted with)
    user_history = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()
    unseen_products = products[~products['product_id'].isin(user_history)]
    
    # Add random unseen products to diversify recommendations
    if len(unseen_products) > 5:
        random_recommendations = unseen_products.sample(n=5)
    else:
        random_recommendations = unseen_products  # Get all if less than 5
    random_recommendations['source'] = 'Random Unseen Products'
    
    # Add popular products (most purchased)
    popular_products = purchases['product_id'].value_counts().index[:5]
    popular_recommendations = products[products['product_id'].isin(popular_products)]
    popular_recommendations['source'] = 'Popular Products'

    print(all_recommendations[all_recommendations.duplicated(subset=['product_id'])])

    
    # Combine all recommendations and drop duplicates based on Product ID
    all_recommendations = pd.concat([hybrid_recommendations, random_recommendations, popular_recommendations]).drop_duplicates(subset=['product_id'])


    return all_recommendations
