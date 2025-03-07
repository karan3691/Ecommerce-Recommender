import pandas as pd
import torch
from model import MultiModalModel

# Load data
users = pd.read_csv("users_expanded.csv")
products = pd.read_csv("products_expanded.csv")
product_images = pd.read_csv("product_images_expanded.csv")

# Initialize model
num_users = users["user_id"].nunique()
num_products = products["product_id"].nunique()
model = MultiModalModel(num_users, num_products)

# Prepare inputs
user_id = 1
# Adjust product IDs to be 0-indexed for the embedding layer
product_ids = torch.LongTensor(products["product_id"].values) - 1
texts = products["description"].tolist()

# Run model inference
with torch.no_grad():
    outputs = model(torch.LongTensor([user_id - 1]), product_ids, texts, edge_index=None, product_images_df=product_images)

# Print results
print("Outputs shape:", outputs.shape)
print("Mean dim=0:", outputs.mean(dim=0).shape)
print("Mean dim=1:", outputs.mean(dim=1).shape)
print("Products shape:", products.shape)