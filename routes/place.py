from flask import Blueprint
from dbconnect import load_from_db
from dbconnect import read_data_from_db
from recs.content_based import get_item_profile
from recs.content_based import _concatenate_cats_of_item
from recs.content_based import get_user_profile
import json
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Response

place_api = Blueprint('place_api', __name__)


@place_api.route('/place/<int:user_id>')
def recommend_place(user_id):
    sql = 'SELECT user_id, place_id, rating FROM rating_place'
    ds = read_data_from_db(sql, None)

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
        similar_places = get_list_db_objects_from_ids(tuple(list_of_ids))
        return Response(similar_places.to_json(orient="records"), mimetype='application/json')
    return "", 400
    

@place_api.route('/place/detail/<int:place_id>')
def recommend_similar_place(place_id):
    df_cat_per_item = get_cat_per_item()

    place_profile = get_item_profile(df_cat_per_item)
    simi_items = place_profile.iloc[place_id-1].sort_values(ascending=False)[:20]
    simi_items = tuple(int(x+1) for x in simi_items.index.values)
    similar_places = get_list_db_objects_from_ids(simi_items)
    return Response(similar_places.to_json(orient="records"), mimetype='application/json')


@place_api.route('/place/similarity/<int:user_id>')
def recommend_similar_place_user_viewed(user_id):
    sql = "SELECT count(*) as times, place_id FROM place_user_log where user_id=%(user_id)s and place_id!='' group by place_id;"
    params = {"user_id" : int(user_id)}
    ds = read_data_from_db(sql, params)

    df_cat_per_item = get_cat_per_item()
    user_data_with_cat_of_items = df_cat_per_item.reset_index().merge(ds, on='place_id')
    recommendations = get_user_profile(user_data_with_cat_of_items, df_cat_per_item)
    print(df_cat_per_item['item_cats'][recommendations])
    simi_items = tuple(int(x+1) for x in recommendations)
    similar_places = get_list_db_objects_from_ids(simi_items)
    return Response(similar_places.to_json(orient="records"), mimetype='application/json')


def get_cat_per_item():
    sql = 'SELECT id, place_id, activity_id FROM activity_place where stt=1;'
    ds = read_data_from_db(sql, None)
    df_cat_per_item = ds.groupby('place_id')['activity_id'].agg(_concatenate_cats_of_item)
    df_cat_per_item.name = 'item_cats'
    df_cat_per_item = df_cat_per_item.reset_index()
    df_cat_per_item[~df_cat_per_item.item_cats.isnull()].reset_index(drop=True)
    return df_cat_per_item


def get_list_db_objects_from_ids(tuple_of_item):
    simi_items_as_string=','.join(map(str,tuple_of_item))
    get_simi_items_query = "SELECT * FROM place where id in %(simi_items)s ORDER BY FIND_IN_SET(id, %(ordered_list)s);"
    params = {"simi_items" : tuple_of_item, "ordered_list": simi_items_as_string}
    ds = read_data_from_db(get_simi_items_query, params)
    return ds