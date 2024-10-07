from flask import Flask, render_template, request
import pandas as pd
from model import collaborative_filtering, content_based_filtering

app = Flask(__name__)

# Load your data
users = pd.read_csv('users.csv')  # Ensure this file exists
products = pd.read_csv('products.csv')  # Ensure this file exists
purchases = pd.read_csv('purchases.csv')  # Ensure this file exists
browsing_history = pd.read_csv('browsing_history.csv')  # Ensure this file exists

# Display the contents
print("Users:")
print(users)
print("\nProducts:")
print(products)
print("\nPurchases:")
print(purchases)
print("\nBrowsing History:")
print(browsing_history)

@app.route('/')
def index():
    return render_template('index.html', products=products.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_id = int(request.form['user_id'])
    algorithm = request.form['algorithm']

    if algorithm == 'collaborative':
        recommendations = collaborative_filtering(user_id, purchases, browsing_history, products)
    elif algorithm == 'content-based':
        recommendations = content_based_filtering(user_id, purchases, browsing_history, products)
    elif algorithm == 'hybrid':
        collaborative_recommendations = collaborative_filtering(user_id, purchases, browsing_history, products)
        content_based_recommendations = content_based_filtering(user_id, purchases, browsing_history, products)
        recommendations = pd.concat([collaborative_recommendations, content_based_recommendations]).drop_duplicates()

    if recommendations.empty:
        message = "No recommendations available for this user."
        return render_template('recommendations.html', message=message)

    # Format recommendations for the template
    formatted_recommendations = []
    for _, row in recommendations.iterrows():
        formatted_recommendations.append({
            'product_id': row['product_id'],
            'product_name': row['product_name'],
            'price': row['price'],
            'rating': row['rating'],
            'source': row['source']  # Ensure the source is set correctly in both methods
        })

    return render_template('recommendations.html', recommendations=formatted_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
