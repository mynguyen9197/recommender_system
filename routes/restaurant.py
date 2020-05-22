from flask import Blueprint
from dbconnect import load_from_db
from dbconnect import read_data_from_db
from recs.content_based import get_item_profile
from recs.content_based import get_user_profile
from recs.content_based import get_liked_cats_at_the_first_time
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import SVD
import pandas as pd
from flask import Response

restaurant_api = Blueprint('restaurant_api', __name__)


@restaurant_api.route('/restaurant/collab/<int:user_id>')
def recommend_restaurant(user_id):
    try:
        sql = 'SELECT user_id, res_id, rating FROM rating_restaurant'
        ds = read_data_from_db(sql, None)

        if len(ds) > 0:
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
            similar_restaurants = get_list_db_objects_from_ids(tuple(list_of_ids))
            return Response(similar_restaurants.to_json(orient="records"), status=200, mimetype='application/json')
        return "not found", 400
    except Exception as e:
        print(str(e))
        return "", 500


@restaurant_api.route('/restaurant/detail/<int:restaurant_id>')
def recommend_similar_restaurant(restaurant_id):
    try:
        df_cat_per_item = get_cat_per_item()

        if len(df_cat_per_item) > 0:
            restaurant_profile = get_item_profile(df_cat_per_item)
            index = df_cat_per_item.index[df_cat_per_item['res_id'] == restaurant_id].tolist()[0]
            simi_items_index = restaurant_profile.iloc[index].sort_values(ascending=False)[:20].index.tolist()
            restaurant_ids = df_cat_per_item.loc[simi_items_index, 'res_id'].tolist()
            restaurant_ids.remove(restaurant_id)
            similar_restaurants = get_list_db_objects_from_ids(restaurant_ids)
            return Response(similar_restaurants.to_json(orient="records"), status=200, mimetype='application/json')
        return "not found", 404
    except Exception as e:
        print(str(e))
        return "", 500


@restaurant_api.route('/restaurant/similarity/<int:user_id>')
def recommend_similar_restaurant_user_viewed(user_id):
    try:
        sql = "SELECT count(*) as times, rest_id as res_id FROM restaurant_user_log where user_id=%(user_id)s and rest_id!='' group by rest_id;"
        params = {"user_id" : int(user_id)}
        ds = read_data_from_db(sql, params)
        sql2 = "SELECT event_type FROM restaurant_user_log where user_id=%(user_id)s and event_type!='VIEW_DETAIL';"
        ds2 = read_data_from_db(sql2, params)
        chosen_cats_as_string = ' '.join(set(map(lambda x: x.split(': ')[1], ds2['event_type'])))

        if (len(ds) == 0 and (not chosen_cats_as_string)):
            return "not found", 404
        else:
            df_cat_per_item = get_cat_per_item()
            if chosen_cats_as_string:
                get_liked_cats_at_the_first_time(chosen_cats_as_string, df_cat_per_item, ds)
            
            user_data_with_cat_of_items = df_cat_per_item.reset_index().merge(ds, on='res_id')
            recommendations = get_user_profile(user_data_with_cat_of_items, df_cat_per_item)
            rest_ids = df_cat_per_item.loc[recommendations, 'res_id'].tolist()
            print(df_cat_per_item['item_cats'][recommendations])
            simi_items = tuple(int(x) for x in rest_ids)
            similar_restaurants = get_list_db_objects_from_ids(simi_items)
            return Response(similar_restaurants.to_json(orient="records"), status=200, mimetype='application/json')
        
    except Exception as e:
        print(str(e))
        return "", 500


def get_cat_per_item():
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
    return df_cat_per_item


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


def get_list_db_objects_from_ids(tuple_of_item):
    simi_items_as_string=','.join(map(str,tuple_of_item))
    get_simi_items_query = "SELECT * FROM restaurant where id in %(simi_items)s ORDER BY FIND_IN_SET(id, %(ordered_list)s);"
    params = {"simi_items" : tuple_of_item, "ordered_list": simi_items_as_string}
    ds = read_data_from_db(get_simi_items_query, params)
    return ds