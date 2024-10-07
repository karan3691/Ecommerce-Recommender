from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def collaborative_filtering(user_id, purchases, browsing_history, products):
    user_purchases = purchases[purchases['user_id'] == user_id]
    user_browsing = browsing_history[browsing_history['user_id'] == user_id]

    if user_purchases.empty and user_browsing.empty:
        return pd.DataFrame()  # If no history, return empty DataFrame

    # Preparing for feature matrix
    purchased_product_ids = user_purchases['product_id'].unique()
    browsed_product_ids = user_browsing['product_id'].unique()
    product_features = products.set_index('product_id')[['price', 'rating']]

    # Creating feature vector based on purchases and browsing
    user_feature_vector = pd.DataFrame(0, index=[0], columns=product_features.index)
    user_feature_vector.loc[0, purchased_product_ids] = 1
    user_feature_vector.loc[0, browsed_product_ids] = 1

    similarity = cosine_similarity(user_feature_vector, product_features.T)

    # For recommendations based on the cosine similarity scores
    recommended_indices = similarity.argsort()[0][-5:][::-1]
    recommended_products = product_features.index[recommended_indices].tolist()

    recommendations_df = products[products['product_id'].isin(recommended_products)].copy()
    recommendations_df['source'] = recommendations_df['product_id'].apply(
        lambda x: 'Purchased' if x in purchased_product_ids else 'Browsed' if x in browsed_product_ids else 'N/A'
    )

    return recommendations_df

def content_based_filtering(user_id, purchases, browsing_history, products):
    user_purchases = purchases[purchases['user_id'] == user_id]
    user_browsing = browsing_history[browsing_history['user_id'] == user_id]

    purchased_product_ids = user_purchases['product_id'].unique()
    browsed_product_ids = user_browsing['product_id'].unique()

    # Combine the purchased and browsed product IDs for exclusion
    excluded_products = set(purchased_product_ids).union(set(browsed_product_ids))

    # Calculate total purchases for each product
    product_ratings = purchases.groupby('product_id')['quantity'].sum().reset_index()
    product_ratings.columns = ['product_id', 'total_purchases']
    top_products = product_ratings.sort_values(by='total_purchases', ascending=False)

    # Exclude products that the user has already purchased
    recommendations = top_products[~top_products['product_id'].isin(purchased_product_ids)]

    recommendations_df = pd.DataFrame()

    if not recommendations.empty:
        recommended_products = recommendations['product_id'].head(5).tolist()
        recommendations_df = products[products['product_id'].isin(recommended_products)].copy()
    else:
        # If there are no new products to recommend, include all products with proper sources
        recommendations_df = products.copy()

    # Update the source column based on purchases and browsing history
    recommendations_df['source'] = recommendations_df['product_id'].apply(
        lambda x: 'Purchased' if x in purchased_product_ids else 'Browsed' if x in browsed_product_ids else 'N/A'
    )

    # Debugging output
    print("Purchased Product IDs:", purchased_product_ids)
    print("Browsed Product IDs:", browsed_product_ids)
    print("Excluded Products:", excluded_products)
    print("Recommendations DataFrame:\n", recommendations_df)

    return recommendations_df

