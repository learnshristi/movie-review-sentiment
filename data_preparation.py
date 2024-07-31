# data_preparation.py
import pandas as pd

# Load the dataset
data = pd.read_csv('movie_reviews.csv')  # Ensure this file contains 'review' and 'sentiment' columns

# Preview the dataset
print(data.head())


