from flask import Blueprint
from dbconnect import load_from_db
from dbconnect import read_data_from_db
from recs.content_based import get_item_profile
from recs.content_based import _concatenate_cats_of_item
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
        for i in range(50 if len(predictions) >= 50 else len(predictions)):
            list_of_ids.append(int(predictions[i].iid))
        return json.dumps(list_of_ids), 200
    return "", 400
    

@place_api.route('/place/detail/<int:place_id>')
def recommend_similar_place(place_id):
    sql = 'SELECT id, place_id, activity_id FROM activity_place where stt=1;'
    ds = read_data_from_db(sql)
    df_cat_per_item = ds.groupby('place_id')['activity_id'].agg(_concatenate_cats_of_item)
    df_cat_per_item.name = 'item_cats'
    df_cat_per_item = df_cat_per_item.reset_index()
    df_cat_per_item[~df_cat_per_item.item_cats.isnull()].reset_index(drop=True)

    tour_profile = get_item_profile(df_cat_per_item)
    simi_items = tour_profile.iloc[place_id-1].sort_values(ascending=False)[:20]
    simi_items = [int(x+1) for x in simi_items.index.values]
    return json.dumps(simi_items)