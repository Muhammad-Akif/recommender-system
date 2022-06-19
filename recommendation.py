from flask import Flask, request
from flask_restful import Api, Resource

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


app = Flask(__name__)
api = Api(app)


class BasicRecommendation(Resource):
    def post(self):
        global value_data
        test_data = request.json
        for key, value in test_data.items():
            value_data = value
        data = pd.DataFrame.from_dict(value_data)
        popular_products = pd.DataFrame(data.groupby('noteId')['rating'].count())
        most_popular = popular_products.sort_values('rating', ascending=False)
        return most_popular.to_json()


class AdvanceRecommendation(Resource):
    def post(self):
        global value_data
        test_data = request.json['data']
        downloaded_data = request.json['downloaded']
        data = pd.DataFrame.from_dict(test_data)
        data = data.head(10000)
        ratings_utility_matrix = data.pivot_table(values='rating', index='userId', columns='noteId', fill_value=0)
        X = ratings_utility_matrix.T
        X1 = X
        SVD = TruncatedSVD(n_components=1)
        decomposed_matrix = SVD.fit_transform(X)
        correlation_matrix = np.corrcoef(decomposed_matrix)
        i = downloaded_data['noteId']  # X.index[0] #noteId
        product_names = list(X.index)
        product_ID = product_names.index(i)
        correlation_product_ID = correlation_matrix[product_ID]
        Recommend = list(X.index[correlation_product_ID > 0.90])
        # Removes the item already bought by the customer
        try:
            Recommend.remove(i)
        except:
            print("err occured.")
        return Recommend[0:9]


api.add_resource(BasicRecommendation, "/basicrecommendation")
api.add_resource(AdvanceRecommendation, "/advancerecommendation")

if __name__ == "__main__":
    app.run(debug=True)
