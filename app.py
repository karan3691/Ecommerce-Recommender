from flask import Flask, render_template, request
import pandas as pd
from model import collaborative_filtering, content_based_filtering

app = Flask(__name__)

# Load your data
users = pd.read_csv('users.csv')  # Ensure this file exists
products = pd.read_csv('products.csv')  # Ensure this file exists
purchases = pd.read_csv('purchases.csv')  # Ensure this file exists

@app.route('/')
def index():
    return render_template('index.html', products=products.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_id = int(request.form['user_id'])
    method = request.form['method']  # Assume you have a dropdown for method selection

    if method == 'collaborative':
        recommendations = collaborative_filtering(user_id, purchases, products)
    elif method == 'content-based':
        recommendations = content_based_filtering(user_id, purchases, products)

    if recommendations.empty:
        message = "No recommendations available for this user."
        return render_template('recommendations.html', message=message)

    # Format the recommendations to include product name, price, and rating
    formatted_recommendations = []
    for _, row in recommendations.iterrows():
        formatted_recommendations.append({
            'product_id': row['product_id'],
            'product_name': row['product_name'],
            'price': row['price'],
            'rating': row['rating']
        })

    return render_template('recommendations.html', recommendations=formatted_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
