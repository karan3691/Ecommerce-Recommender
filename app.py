from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import torch
from model import collaborative_filtering, content_based_filtering, hybrid_recommendation, MultiModalModel
import logging

app = Flask(__name__)
app.secret_key = 'your-secret-key'
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load data
users = pd.read_csv('users_expanded.csv')
products = pd.read_csv('products_expanded.csv')
product_images = pd.read_csv('product_images_expanded.csv')
# Initialize multi-modal model
num_users = users['user_id'].nunique()
num_products = products['product_id'].nunique()
model = MultiModalModel(num_users, num_products)
purchases = pd.read_csv('purchases_expanded.csv')
browsing_history = pd.read_csv('browsing_history_expanded.csv')

@app.route('/')
def index():
    return render_template('index.html', products=products.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        user_id = int(request.form['user_id'])
        algorithm = request.form['algorithm']
        logger.debug(f"Processing request for user_id: {user_id}, algorithm: {algorithm}")

        if user_id not in users['user_id'].values:
            flash('User ID not found!')
            return redirect(url_for('index'))

        # Get interacted products
        purchased_product_ids = purchases[purchases['user_id'] == user_id]['product_id'].unique()
        browsed_product_ids = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()
        interacted_products = products[products['product_id'].isin(purchased_product_ids) | 
                                      products['product_id'].isin(browsed_product_ids)].copy()
        interacted_products['source'] = interacted_products['product_id'].apply(
            lambda x: 'Purchased' if x in purchased_product_ids else 'Browsed'
        )
        logger.debug(f"Interacted products: {interacted_products['product_id'].tolist()}")

        # Generate recommendations
        if algorithm == 'collaborative':
            recommendations = collaborative_filtering(user_id, purchases, products)
        elif algorithm == 'content-based':
            recommendations = content_based_filtering(user_id, purchases, browsing_history, products)
        elif algorithm == 'hybrid':
            recommendations = hybrid_recommendation(user_id, purchases, browsing_history, products)
        elif algorithm == 'multi-modal':
            # Prepare multi-modal inputs
            # Adjust product IDs to be 0-indexed for the embedding layer
            product_ids = torch.LongTensor(products['product_id'].values) - 1
            texts = products['description'].tolist()
            
            # Generate recommendations using all available modalities
            with torch.no_grad():
                outputs = model(
                    torch.LongTensor([user_id - 1]),
                    product_ids,
                    texts,
                    edge_index=None,
                    product_images_df=product_images
                )
            
            # Calculate recommendation scores
            scores = outputs.mean(dim=1).cpu().numpy()
            
            # Create recommendations dataframe
            recommendations = products.copy()
            recommendations['score'] = scores
            recommendations['source'] = 'Multi-Modal'
        else:
            flash('Invalid algorithm selected!')
            return redirect(url_for('index'))

        # Filter out interacted products
        recommended_products = recommendations[~recommendations['product_id'].isin(purchased_product_ids) & 
                                               ~recommendations['product_id'].isin(browsed_product_ids)].copy()
        logger.debug(f"Filtered recommendations:\n{recommended_products[['product_id', 'score', 'source']]}")

        if recommended_products.empty:
            flash('No recommendations available for this user.')

        return render_template('recommendations.html', 
                             interacted_products=interacted_products.to_dict(orient='records'),
                             recommended_products=recommended_products.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Use gunicorn in production, debug mode locally
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)