from flask import Blueprint
from dbconnect import load_from_db
from dbconnect import read_data_from_db
import json
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

place_api = Blueprint('place_api', __name__)


@place_api.route('/place/<int:user_id>')
def recommend_place(user_id):
    sql = 'SELECT user_id, place_id, rating FROM rating_place'
    ds = read_data_from_db(sql)

    if len(ds) > 0:
        reader = Reader()
        data = Dataset.load_from_df(ds[['user_id', 'place_id', 'rating']], reader=reader)
        alg = SVD()
        alg.fit(data.build_full_trainset())

        iids = ds['place_id'].unique()
        rated_iids = ds.loc[ds['user_id'] == user_id, 'place_id']
        iids_to_pred = np.setdiff1d(iids, rated_iids)
        testset = [[user_id, iid, 4.] for iid in iids_to_pred]
        predictions = alg.test(testset)
        predictions.sort(key=lambda x: x.est, reverse=True)
        list_of_ids = []
        for i in range(20 if len(predictions) >= 20 else len(predictions)):
            list_of_ids.append(int(predictions[i].iid))
        return json.dumps(list_of_ids), 200
    return "", 400

# def content_based_recommend(name):
#     sql = 'SELECT id, name, about FROM place'
#     ds = read_data_from_db(sql)
#     tfidf = TfidfVectorizer(stop_words='english')
#     ds['about'] = ds['about'].fillna('')
#     tfidf_matrix = tfidf.fit_transform(ds['about'])
#     cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#     indices = pd.Series(ds.index, index=ds['name']).drop_duplicates()
#
#     idx=indices[name]
#     sim_scores = list(enumerate(cosine_sim(idx)))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:11]
#     place_indices = [i[0] for i in sim_scores]
#     list_places = ds['name'].iloc[place_indices]
#     for i in range(10):
#         return list_places[i]
#     return ds['name'].iloc[place_indices]