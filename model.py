from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def collaborative_filtering(user_id, purchases, browsing_history, products):
    user_purchases = purchases[purchases['user_id'] == user_id]
    user_browsing = browsing_history[browsing_history['user_id'] == user_id]

    if user_purchases.empty and user_browsing.empty:
        return pd.DataFrame()  #if no history, it will return empty df

    #preparing for feature matrix
    purchased_product_ids = user_purchases['product_id'].unique()
    browsed_product_ids = user_browsing['product_id'].unique()
    product_features = products.set_index('product_id')[['price', 'rating']]

    #creating feature vector based on purchases and browsing
    user_feature_vector = pd.DataFrame(0, index=[0], columns=product_features.index)
    user_feature_vector.loc[0, purchased_product_ids] = 1
    user_feature_vector.loc[0, browsed_product_ids] = 1

    similarity = cosine_similarity(user_feature_vector, product_features.T)

    # for recommendations based on the cosine similarity scores
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

    # average
    product_ratings = purchases.groupby('product_id')['quantity'].sum().reset_index()
    product_ratings.columns = ['product_id', 'total_purchases']
    top_products = product_ratings.sort_values(by='total_purchases', ascending=False)

    # excluding those products the user has already purchased or browsed
    recommendations = top_products[~top_products['product_id'].isin(purchased_product_ids)]
    recommendations = recommendations[~recommendations['product_id'].isin(browsed_product_ids)]
    recommended_products = recommendations['product_id'].head(5).tolist()  # Get top 5 recommendations

    # df for recommendations
    recommendations_df = products[products['product_id'].isin(recommended_products)].copy()
    recommendations_df['source'] = recommendations_df['product_id'].apply(
        lambda x: 'Purchased' if x in purchased_product_ids else 'Browsed' if x in browsed_product_ids else 'N/A'
    )

    return recommendations_df