from flask import Blueprint
from dbconnect import load_from_db
from dbconnect import read_data_from_db
from recs.content_based import get_item_profile
import json
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import SVD
import pandas as pd

restaurant_api = Blueprint('restaurant_api', __name__)


@restaurant_api.route('/restaurant/<int:user_id>')
def recommend_restaurant(user_id):
    sql = 'SELECT user_id, res_id, rating FROM rating_restaurant'
    ds = read_data_from_db(sql, None)

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
    for i in range(50):
        list_of_ids.append(int(predictions[i].iid))
    return json.dumps(list_of_ids)


@restaurant_api.route('/restaurant/detail/<int:restaurant_id>')
def recommend_similar_restaurant(restaurant_id):
    cuisine_sql = 'SELECT id, cuisine_id, res_id FROM cuisine_restaurant where stt=1;'
    ds = read_data_from_db(cuisine_sql, None)
    df_cuisine_per_item = ds.groupby('res_id')['cuisine_id'].agg(_concatenate_cuisine_of_item)

    feature_sql = 'SELECT id, feature_id, res_id FROM feature_restaurant where stt=1;'
    ds = read_data_from_db(feature_sql, None)
    df_feature_per_item = ds.groupby('res_id')['feature_id'].agg(_concatenate_feature_of_item)

    meal_sql = 'SELECT id, meal_id, res_id FROM meal_restaurant where stt=1;'
    ds = read_data_from_db(meal_sql, None)
    df_meal_per_item = ds.groupby('res_id')['meal_id'].agg(_concatenate_meal_of_item)

    type_sql ='SELECT id, type_id, res_id FROM foodtype_restaurant where stt=1;'
    ds = read_data_from_db(type_sql, None)
    df_type_per_item = ds.groupby('res_id')['type_id'].agg(_concatenate_type_of_item)

    df_data = pd.merge(df_cuisine_per_item, df_feature_per_item, how='left', on='res_id')
    df_data = pd.merge(df_data, df_meal_per_item, how='left', on='res_id')
    df_data = pd.merge(df_data, df_type_per_item, how='left', on='res_id')
    serie_data = df_data[df_data.columns[0:]].apply(
            lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df_cat_per_item = pd.DataFrame({'res_id':serie_data.index, 'item_cats':serie_data.values})

    restaurant_profile = get_item_profile(df_cat_per_item)
    simi_items = restaurant_profile.iloc[restaurant_id-1].sort_values(ascending=False)[:20]
    print(simi_items)
    simi_items = [int(x+1) for x in simi_items.index.values]
    return json.dumps(simi_items)


def _concatenate_cuisine_of_item(cats):
    cats_as_str = ' '.join(set(map(lambda x: 'c_' + str(x), cats)))
    return cats_as_str


def _concatenate_feature_of_item(cats):
    cats_as_str = ' '.join(set(map(lambda x: 'f_' + str(x), cats)))
    return cats_as_str


def _concatenate_meal_of_item(cats):
    cats_as_str = ' '.join(set(map(lambda x: 'm_' + str(x), cats)))
    return cats_as_str


def _concatenate_type_of_item(cats):
    cats_as_str = ' '.join(set(map(lambda x: 't_' + str(x), cats)))
    return cats_as_str