from flask import Flask, render_template, request
import pandas as pd
from model import collaborative_filtering, content_based_filtering

app = Flask(__name__)

users = pd.read_csv('users.csv') 
products = pd.read_csv('products.csv') 
purchases = pd.read_csv('purchases.csv')
browsing_history = pd.read_csv('browsing_history.csv')

@app.route('/')
def index():
    return render_template('index.html', products=products.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_id = int(request.form['user_id'])
    algorithm = request.form['algorithm']

    # get purchased and browsed product IDs
    purchased_product_ids = purchases[purchases['user_id'] == user_id]['product_id'].unique()
    browsed_product_ids = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()

    # products the user has already interacted with
    interacted_products = products[products['product_id'].isin(purchased_product_ids) | products['product_id'].isin(browsed_product_ids)]

    # penerate recommendations based on selected algorithm
    if algorithm == 'collaborative':
        recommendations = collaborative_filtering(user_id, purchases, browsing_history, products)
    elif algorithm == 'content-based':
        recommendations = content_based_filtering(user_id, purchases, browsing_history, products)
    elif algorithm == 'hybrid':
        collaborative_recommendations = collaborative_filtering(user_id, purchases, browsing_history, products)
        content_based_recommendations = content_based_filtering(user_id, purchases, browsing_history, products)
        recommendations = pd.concat([collaborative_recommendations, content_based_recommendations]).drop_duplicates()

    # it will exclude the already interacted products from recommendations
    recommended_products = recommendations[~recommendations['product_id'].isin(purchased_product_ids) & ~recommendations['product_id'].isin(browsed_product_ids)]

    # prepare data for rendering
    interacted_products = interacted_products[['product_id', 'product_name', 'price', 'rating']].copy()
    interacted_products['source'] = interacted_products['product_id'].apply(
        lambda x: 'Purchased' if x in purchased_product_ids else 'Browsed'
    )

    recommended_products = recommended_products[['product_id', 'product_name', 'price', 'rating']].copy()
    recommended_products['source'] = 'Recommended'

    return render_template('recommendations.html', interacted_products=interacted_products.to_dict(orient='records'), recommended_products=recommended_products.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
