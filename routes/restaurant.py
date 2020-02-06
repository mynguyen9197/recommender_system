from flask import Blueprint
from dbconnect import load_from_db
from dbconnect import read_data_from_db
import json
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import SVD

restaurant_api = Blueprint('restaurant_api', __name__)


@restaurant_api.route('/restaurant/<int:user_id>')
def recommend_restaurant(user_id):
    sql = 'SELECT user_id, res_id, rating FROM rating_restaurant'
    ds = read_data_from_db(sql)

    reader = Reader()
    data = Dataset.load_from_df(ds[['user_id', 'res_id', 'rating']], reader=reader)
    alg = SVD()
    alg.fit(data.build_full_trainset())

    iids = ds['res_id'].unique()
    rated_iids = ds.loc[ds['user_id'] == user_id, 'res_id']
    iids_to_pred = np.setdiff1d(iids, rated_iids)
    testset = [[user_id, iid, 4.] for iid in iids_to_pred]
    predictions = alg.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    list_of_ids = []
    for i in range(20):
        list_of_ids.append(int(predictions[i].iid))
    return json.dumps(list_of_ids)