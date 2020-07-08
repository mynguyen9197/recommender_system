from flask import Blueprint
from dbconnect import load_from_db
from dbconnect import read_data_from_db
from recs.content_based import similar_to_item
from recs.content_based import _concatenate_cats_of_item
from recs.content_based import similar_to_user_profile
from recs.evaluate_prediction import evaluate_surprise_alg
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import SVD
import pandas as pd
from flask import Response

place_api = Blueprint('place_api', __name__)


@place_api.route('/place/collab/<int:user_id>')
def recommend_place(user_id):
    try:
        find_user_rating = 'SELECT * FROM rating_place where user_id=%(user_id)s;'
        params = {"user_id" : int(user_id)}
        user_rating = read_data_from_db(find_user_rating, params)
        sql = 'SELECT user_id, place_id, rating FROM rating_place'
        ds = read_data_from_db(sql, None)

        if len(ds) > 0 and len(user_rating) >0:
            reader = Reader()
            data = Dataset.load_from_df(ds[['user_id', 'place_id', 'rating']], reader=reader)
            alg = SVD()
            alg.fit(data.build_full_trainset())

            iids = ds['place_id'].unique()
            rated_iids = ds.loc[ds['user_id'] == user_id, 'place_id']
            iids_to_pred = np.setdiff1d(iids, rated_iids)
            testset = [[user_id, iid, 4.] for iid in iids_to_pred]
            predictions = alg.test(testset)
            evaluate_surprise_alg(predictions)
            predictions.sort(key=lambda x: x.est, reverse=True)
            list_of_ids = []
            for i in range(50 if len(predictions) >= 50 else len(predictions)):
                list_of_ids.append(int(predictions[i].iid))
                print(str(predictions[i].iid) + ' ' + str(predictions[i].est))
            similar_places = get_list_db_objects_from_ids(tuple(list_of_ids))
            return Response(similar_places.to_json(orient="records"), status=200, mimetype='application/json')
        return "not found", 404
    except Exception as e:
        print(str(e))
        return "", 500
    

@place_api.route('/place/detail/<int:place_id>')
def recommend_similar_place(place_id):
    try:
        df_cat_per_item = get_cat_per_item()
        if len(df_cat_per_item) > 0:
            place_profile = similar_to_item(df_cat_per_item)
            index = df_cat_per_item.index[df_cat_per_item['place_id'] == place_id].tolist()[0]
            simi_items_index = place_profile.iloc[index].sort_values(ascending=False)[:20].index.tolist()
            place_ids = df_cat_per_item.loc[simi_items_index, 'place_id'].tolist()
            if place_id in place_ids: place_ids.remove(place_id)
            similar_places = get_list_db_objects_from_ids(place_ids)
            return Response(similar_places.to_json(orient="records"), status=200, mimetype='application/json')
        return "not found", 404
    except Exception as e:
        print(str(e))
        return "", 500


@place_api.route('/place/similarity/<int:user_id>')
def recommend_similar_place_user_viewed(user_id):
    try:
        sql = "SELECT count(*) as times, place_id FROM place_user_log where user_id=%(user_id)s and place_id!='' group by place_id;"
        params = {"user_id" : int(user_id)}
        ds = read_data_from_db(sql, params)
        sql2 = "SELECT event_type FROM place_user_log where user_id=%(user_id)s and event_type!='VIEW_DETAIL';"
        ds2 = read_data_from_db(sql2, params)
        chosen_cats_as_string = ' '.join(set(map(lambda x: x.split(': ')[1], ds2['event_type'])))

        if (len(ds) == 0 and (not chosen_cats_as_string)):
            return "not found", 404
        else:
            df_cat_per_item = get_cat_per_item()
            if chosen_cats_as_string:
                get_liked_cats_at_the_first_time(chosen_cats_as_string, df_cat_per_item, ds)
            
            user_data_with_cat_of_items = df_cat_per_item.reset_index().merge(ds, on='place_id')
            print(user_data_with_cat_of_items)
            recommendations, simi_list = similar_to_user_profile(user_data_with_cat_of_items, df_cat_per_item)
            place_ids = df_cat_per_item.loc[recommendations, 'place_id'].tolist()
            recommended_items = df_cat_per_item.loc[recommendations]
            x = recommended_items.reset_index().join(simi_list)
            print(x)
            simi_items = tuple(int(x) for x in place_ids)
            similar_places = get_list_db_objects_from_ids(simi_items)
            return Response(similar_places.to_json(orient="records"), status=200, mimetype='application/json')
    except Exception as e:
        print(str(e))
        return "", 500


def get_cat_per_item ():
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
    params = {"simi_items": tuple_of_item, "ordered_list": simi_items_as_string}
    ds = read_data_from_db(get_simi_items_query, params)
    return ds


def get_liked_cats_at_the_first_time(chosen_cats_as_string, df_cat_per_item, df_times_per_item):
    item_id = 0 if df_cat_per_item.empty else df_cat_per_item['place_id'].max() + 1
    new_row_of_cat_per_item = [item_id, chosen_cats_as_string]
    new_row_of_times_per_item = [5, item_id]
    df_cat_per_item.loc[len(df_cat_per_item)] = new_row_of_cat_per_item
    df_times_per_item.loc[len(df_times_per_item)] = new_row_of_times_per_item