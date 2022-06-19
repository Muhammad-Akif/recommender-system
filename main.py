import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD


amazon_ratings = pd.read_csv('./ratings_Notes.csv/ratings_Notes.csv')
amazon_ratings = amazon_ratings.dropna()
# print("Head *************")
# print(amazon_ratings.head())
# print(amazon_ratings.shape)
# print()

# data = '[{"_id":"610f098b17fca751ec76b521","userId":"610ee3692408732bec945b9f","noteId":"610ee84e53ede227f8877d56","rating":5}]';
# userNoteId = "0205616461"
# amazon_ratings = pd.read_json(data);

popular_products = pd.DataFrame(amazon_ratings.groupby('noteId')['rating'].count())
most_popular = popular_products.sort_values('rating', ascending=False)
print("Recommendation System part 1, on basis of rating printing popularity.")
print(most_popular.head(10).to_json())
print()

# Recommendation system part 2 (Collaborative filtering)
amazon_ratings1 = amazon_ratings.head(10000)
ratings_utility_matrix = amazon_ratings1.pivot_table(values='rating', index='userId', columns='noteId', fill_value=0)
# print("user to product ratings")
# print(ratings_utility_matrix.head())
# print(ratings_utility_matrix.shape)
# print()

X = ratings_utility_matrix.T
# print("Products to User ratings")
# print(X.head())
# print(X.shape)
# print()

X1 = X

SVD = TruncatedSVD(n_components=10)
print("*/*/*/*/*/*/*/*/-*/-*/-*/-*/-*/*")
print(X)
decomposed_matrix = SVD.fit_transform(X)
# print("Decomposing the matrix")
# print(decomposed_matrix.shape)
# print()

correlation_matrix = np.corrcoef(decomposed_matrix)
# print("Correlation matrix")
# print(correlation_matrix.shape)
# print()

i = "0205616461" # X.index[0] #noteId
product_names = list(X.index)
product_ID = product_names.index(i)

correlation_product_ID = correlation_matrix[product_ID]
print("Correlation ")
print(correlation_product_ID.shape)
print()

Recommend = list(X.index[correlation_product_ID > 0.90])
# Removes the item already bought by the customer
Recommend.remove(i)
print("Recommending top 10")
print(Recommend[0:9])
print()
