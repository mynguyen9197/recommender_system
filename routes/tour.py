from flask import Blueprint
from dbconnect import load_from_db, read_data_from_db
from recs.content_based import get_item_profile, _concatenate_cats_of_item, get_user_profile, get_liked_cats_at_the_first_time
from recs.evaluate_prediction import evaluate_surprise_alg
import numpy as np
from surprise import Dataset, accuracy, Reader, SVD, KNNBasic, KNNBaseline, SVDpp, NMF
from flask import Response
import pandas as pd

tour_api = Blueprint('tour_api', __name__)


@tour_api.route('/tour/collab/<int:user_id>')
def recommend_tour(user_id):
    try:
        sql = 'SELECT user_id, tour_id, rating FROM rating_tour'
        ds = read_data_from_db(sql, None)

        if len(ds) > 0:
            reader = Reader()
            data = Dataset.load_from_df(ds[['user_id', 'tour_id', 'rating']], reader=reader)
            alg = SVD()
            alg.fit(data.build_full_trainset())

            iids = ds['tour_id'].unique()
            rated_iids = ds.loc[ds['user_id'] == user_id, 'tour_id']
            iids_to_pred = np.setdiff1d(iids, rated_iids)
            testset = [[user_id, iid, 4.] for iid in iids_to_pred]
            predictions = alg.test(testset)
            evaluate_surprise_alg(predictions)
            predictions.sort(key=lambda x: x.est, reverse=True)
            list_of_ids = []
            for i in range(50):
                list_of_ids.append(int(predictions[i].iid))
            similar_tours = get_list_db_objects_from_ids(tuple(list_of_ids))
            return Response(similar_tours.to_json(orient="records"), status=200, mimetype='application/json')
        return "not found", 404
    except Exception as e:
        print(str(e))
        return "", 500


@tour_api.route('/tour/detail/<int:tour_id>')
def recommend_similar_tour(tour_id):
    try:
        df_cat_per_item = get_cat_per_item()

        if len(df_cat_per_item) > 0:
            tour_profile = get_item_profile(df_cat_per_item)
            index = df_cat_per_item.index[df_cat_per_item['tour_id'] == tour_id].tolist()[0]
            simi_items_index = tour_profile.iloc[index].sort_values(ascending=False)[:20].index.tolist()
            tour_ids = df_cat_per_item.loc[simi_items_index, 'tour_id'].tolist()
            if tour_id in tour_ids: tour_ids.remove(tour_id)
            similar_tours = get_list_db_objects_from_ids(tour_ids)
            return Response(similar_tours.to_json(orient="records"), status=200, mimetype='application/json')
        return "not found", 404
    except Exception as e:
        print(str(e))
        return "", 500


@tour_api.route('/tour/similarity/<int:user_id>')
def recommend_similar_tour_user_viewed(user_id):
    try:
        sql = "SELECT count(*) as times, tour_id FROM tour_user_log where user_id=%(user_id)s and tour_id!='' group by tour_id;"
        params = {"user_id" : int(user_id)}
        ds = read_data_from_db(sql, params)
        sql2 = "SELECT event_type FROM tour_user_log where user_id=%(user_id)s and event_type!='VIEW_DETAIL';"
        ds2 = read_data_from_db(sql2, params)
        chosen_cats_as_string = ' '.join(set(map(lambda x: x.split(': ')[1], ds2['event_type'])))

        if (len(ds) == 0 and (not chosen_cats_as_string)):
            return "not found", 404
        else:
            df_cat_per_item = get_cat_per_item()
            if chosen_cats_as_string:
                get_liked_cats_at_the_first_time(chosen_cats_as_string, df_cat_per_item, ds)
            
            user_data_with_cat_of_items = df_cat_per_item.reset_index().merge(ds, on='tour_id')
            recommendations = get_user_profile(user_data_with_cat_of_items, df_cat_per_item)
            tour_ids = df_cat_per_item.loc[recommendations, 'tour_id'].tolist()
            print(df_cat_per_item['item_cats'][recommendations])
            simi_items = tuple(int(x) for x in tour_ids)
            similar_tours = get_list_db_objects_from_ids(simi_items)
            return Response(similar_tours.to_json(orient="records"), status=200, mimetype='application/json')
    except Exception as e:
        print(str(e))
        return "", 500


def get_cat_per_item():
    sql = 'SELECT id, tour_id, activity_id FROM activity_tour where stt=1;'
    ds = read_data_from_db(sql, None)
    df_cat_per_item = ds.groupby('tour_id')['activity_id'].agg(_concatenate_cats_of_item)
    df_cat_per_item.name = 'item_cats'
    df_cat_per_item = df_cat_per_item.reset_index()
    df_cat_per_item[~df_cat_per_item.item_cats.isnull()].reset_index(drop=True)
    return df_cat_per_item


def get_list_db_objects_from_ids(tuple_of_item):
    simi_items_as_string=','.join(map(str,tuple_of_item))
    get_simi_items_query = "SELECT * FROM tour where id in %(simi_items)s ORDER BY FIND_IN_SET(id, %(ordered_list)s);"
    params = {"simi_items" : tuple_of_item, "ordered_list": simi_items_as_string}
    ds = read_data_from_db(get_simi_items_query, params)
    return ds