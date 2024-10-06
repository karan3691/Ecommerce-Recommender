from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def collaborative_filtering(user_id, purchases, products):
    # Get the user's purchase history
    user_purchases = purchases[purchases['user_id'] == user_id]

    if user_purchases.empty:
        return pd.DataFrame()  # No purchases, return empty DataFrame

    # Prepare the product feature matrix
    purchased_product_ids = user_purchases['product_id'].unique()
    product_features = products.set_index('product_id')[['price', 'rating']]

    # Create a user feature vector based on purchases (e.g., one-hot encoded)
    user_feature_vector = pd.DataFrame(0, index=[0], columns=product_features.index)
    user_feature_vector.loc[0, purchased_product_ids] = 1

    # Calculate cosine similarity
    similarity = cosine_similarity(user_feature_vector, product_features.T)

    # Get recommendations based on the similarity scores
    recommended_indices = similarity.argsort()[0][-5:][::-1]
    recommended_products = product_features.index[recommended_indices].tolist()

    return products[products['product_id'].isin(recommended_products)]

def content_based_filtering(user_id, purchases, products):
    # Example implementation: recommend products based on average ratings or other features
    user_purchases = purchases[purchases['user_id'] == user_id]
    purchased_product_ids = user_purchases['product_id'].unique()

    # Compute average ratings for all products
    product_ratings = purchases.groupby('product_id')['quantity'].sum().reset_index()
    product_ratings.columns = ['product_id', 'total_purchases']
    top_products = product_ratings.sort_values(by='total_purchases', ascending=False)

    # Exclude products the user has already purchased
    recommendations = top_products[~top_products['product_id'].isin(purchased_product_ids)]

    return products[products['product_id'].isin(recommendations['product_id'].head(5))]

