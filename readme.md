# E-commerce Product Recommendation System

## Project Overview
This project implements a comprehensive product recommendation system for an e-commerce platform. The system utilizes multiple recommendation techniques including collaborative filtering, content-based filtering, hybrid approach, and an advanced multi-modal deep learning model to provide highly personalized product recommendations to users based on their purchase and browsing history.

## Features
- **Collaborative Filtering**: Recommends products based on the purchase behavior of similar users.
- **Content-Based Filtering**: Recommends products based on the attributes of the products the user has previously interacted with.
- **Hybrid Approach**: Combines collaborative and content-based filtering for more robust recommendations.
- **Multi-Modal Deep Learning**: Leverages product images, text descriptions, and user-item interactions using a sophisticated neural network architecture.
- **User-Friendly Interface**: A web application built with Flask for easy interaction and visualization of recommendations.

## Technologies Used
- **Python**: For backend logic and data processing.
- **PyTorch**: For building and training the multi-modal neural network model.
- **Flask**: Web framework for creating the web application.
- **Pandas**: For data manipulation and analysis.
- **Sentence Transformers**: For natural language processing of product descriptions.
- **PyTorch Geometric**: For graph-based neural network components.
- **ResNet50**: For image feature extraction.
- **HTML/CSS**: For frontend design.

## System Architecture
The recommendation system consists of four main components:

1. **Collaborative Filtering Module**: Analyzes user purchase patterns to find similar users and recommend products they've purchased.

2. **Content-Based Filtering Module**: Analyzes product attributes (especially categories) and user browsing history to recommend similar products.

3. **Hybrid Recommendation Module**: Combines results from both collaborative and content-based approaches, with fallback to popular products when necessary.

4. **Multi-Modal Deep Learning Model**: A sophisticated neural network that processes:
   - User embeddings
   - Product embeddings
   - Image features (using ResNet50 with fashion-specific optimizations)
   - Text features (using Sentence Transformers)
   - Graph-based relationships (using Graph Convolutional Networks)

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- Flask
- Pandas
- Sentence Transformers
- PyTorch Geometric
- Torchvision
- PIL

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd er
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup
1. Download the DeepFashion dataset from [official website](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

2. Create a `datasets` directory in the project root:
   ```bash
   mkdir datasets
   ```

3. Extract the downloaded dataset into the `datasets` directory. The directory structure should look like:
   ```
   datasets/
     ├── shape_ann/
     ├── test_images/
     └── train_images/
   ```

4. Place the following data files in the project root:
   - users_expanded.csv
   - products_expanded.csv
   - product_images_expanded.csv
   - purchases_expanded.csv
   - browsing_history_expanded.csv

**Note**: The dataset files are not included in the repository due to their large size. Please download them separately.

## Running The Application
1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to http://127.0.0.1:5000.

## Data Files
1. `users_expanded.csv`: Contains user information.
2. `products_expanded.csv`: Contains product details such as name, category, description, price, and rating.
3. `product_images_expanded.csv`: Contains mappings between products and their image files.
4. `purchases_expanded.csv`: Contains user purchase history.
5. `browsing_history_expanded.csv`: Contains user browsing history.

## Dataset
The system uses fashion product images from the DeepFashion dataset, which includes various clothing items with multiple view angles (front, side, back, full). The multi-modal model is specifically optimized for fashion recommendations with view-aware processing.

## Usage
1. Enter a user ID and select a recommendation method (collaborative, content-based, hybrid, or multi-modal).
2. Click on "Get Recommendations" to view the recommended products.
3. The system will display both products the user has interacted with and new recommendations.

## Multi-Modal Model Details
The multi-modal model (`MultiModalModel` class) integrates multiple data sources:

- **Collaborative Filtering Component**: User and product embeddings.
- **Image Processing Component**: Fashion-optimized ResNet50 with view-type awareness (front, side, back, full views).
- **Text Processing Component**: Sentence transformer to encode product descriptions.
- **Graph Component**: Graph convolutional network for modeling product relationships.
- **Fusion Layer**: Combines all modalities for final recommendation scores.

## Screenshots
![Homepage](images/homepage.png)
![Recommendations Page](images/recommendations.png)

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## Future Enhancements

- **More Data**: Plans to expand the dataset to enhance the model's recommendation capabilities.
  
- **Clean UI**: Improving the user interface (UI) to enhance user experience and engagement.

- **Feedback Mechanism**: Implement a way for users to provide feedback on the recommendations, which can help refine the system.

- **Explainability**: Work on explaining the reasoning behind recommendations, which can enhance user trust and satisfaction.

- **Model Optimization**: Fine-tune the multi-modal model for better performance and faster inference.

- **Additional Modalities**: Incorporate more data sources such as user demographics and seasonal trends.



   
