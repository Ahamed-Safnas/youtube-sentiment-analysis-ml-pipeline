import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
from flask import Flask, jsonify,request,send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle


app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print  (f"Error in preprocessing comment: {e}")
        return comment

# Function to load the model and vectorizer from MLflow model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    """Load the trained model and vectorizer from MLflow."""

    mlflow.set_tracking_uri("http://ec2-3-235-226-137.compute-1.amazonaws.com:5000/")   
    client = MlflowClient()
    # model_uri = f"models:/{model_name}/{model_version}"
    
    # Load the model
    # model = mlflow.sklearn.load_model(model_uri)

    # Get the model version details to extract the correct artifact URI
    model_version_details = client.get_model_version(model_name, model_version)
    source = model_version_details.source  # This is the raw S3 URI
    
    print(f"Loading model from: {source}")  # Debug: verify the path
    model = mlflow.sklearn.load_model(source)
    
    # Load the vectorizer
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    
    return model, vectorizer
# here im getting model from mlflow and vectorizer from local storage, you can also store vectorizer in mlflow and load it from there if you want to keep everything in one place    

model, vectorizer = load_model_and_vectorizer('my_model', 1, './tfidf_vectorizer.pkl')

# def load_model(model_path, vectorizer_path):
#     """Load the trained model and vectorizer from local storage."""
#     try:
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
#         with open(vectorizer_path, 'rb') as file:
#             vectorizer = pickle.load(file)
#         return model, vectorizer
#     except Exception as e:
#         print(f"Error loading model or vectorizer: {e}")
#         return None, None



@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API!" 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comment')
    # print("comments received for prediction:", comments)
    # print("type of comments received:", type(comments))
    if not comments:
        return jsonify({'error': 'No comment provided'}), 400
    
    try:
        # Preprocess the comment
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Vectorize the comment
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert the sparse matrix to a dense format before making predictions
        dense_comments = transformed_comments.toarray()
        # Make predictions
        predictions = model.predict(dense_comments).tolist()

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

    response = [{"comment": comment, "sentiment": prediction} for comment, prediction in zip(comments, predictions)]
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)