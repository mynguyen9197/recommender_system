from flask import Blueprint
from dbconnect import load_from_db
from dbconnect import read_data_from_db
import json
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import SVD

place_api = Blueprint('place_api', __name__)


@place_api.route('/place/<int:user_id>')
def recommend_place(user_id):
    sql = 'SELECT user_id, place_id, rating FROM rating_place'
    ds = read_data_from_db(sql)

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
    for i in range(20):
        print(predictions[i])
    return ""