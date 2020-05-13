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
from sklearn.metrics.pairwise import cosine_similarity

tour_api = Blueprint('tour_api', __name__)


@tour_api.route('/tour/<int:user_id>')
def recommend_tour(user_id):
    sql = 'SELECT user_id, tour_id, rating FROM rating_tour'
    ds = read_data_from_db(sql, None)

    reader = Reader()
    data = Dataset.load_from_df(ds[['user_id', 'tour_id', 'rating']], reader=reader)
    alg = SVD()
    alg.fit(data.build_full_trainset())

    iids = ds['tour_id'].unique()
    rated_iids = ds.loc[ds['user_id'] == user_id, 'tour_id']
    iids_to_pred = np.setdiff1d(iids, rated_iids)
    print(iids_to_pred)
    testset = [[user_id, iid, 4.] for iid in iids_to_pred]
    predictions = alg.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    list_of_ids = []
    for i in range(50):
        list_of_ids.append(int(predictions[i].iid))
    return json.dumps(list_of_ids)


@tour_api.route('/tour/detail/<int:tour_id>')
def recommend_similar_tour(tour_id):
    df_cat_per_item = get_cat_per_item()

    tour_profile = get_item_profile(df_cat_per_item)
    simi_items = tour_profile.iloc[tour_id-1].sort_values(ascending=False)[:20]
    simi_items = [int(x+1) for x in simi_items.index.values]
    return json.dumps(simi_items)


@tour_api.route('/tour/similarity/<int:user_id>')
def recommend_similar_tour_user_viewed(user_id):
    sql = "SELECT count(*) as times, tour_id FROM tour_user_log where user_id=%(user_id)s and tour_id!='' group by tour_id;"
    params = {"user_id" : int(user_id)}
    ds = read_data_from_db(sql, params)

    df_cat_per_item = get_cat_per_item()
    user_data_with_cat_of_items = df_cat_per_item.reset_index().merge(ds, on='tour_id')
    max_times = user_data_with_cat_of_items['times'].max()
    user_data_with_cat_of_items['weight'] = user_data_with_cat_of_items['times']/max_times
    tf_idf = TfidfVectorizer()
    df_items_tf_idf_cats = tf_idf.fit_transform(df_cat_per_item.item_cats)
    user_profile = np.dot(df_items_tf_idf_cats[user_data_with_cat_of_items['index'].values].toarray().T, 
            user_data_with_cat_of_items['weight'].values)
    C = cosine_similarity(np.atleast_2d(user_profile), df_items_tf_idf_cats)
    R = np.argsort(C)[:, ::-1]
    recommendations = [i for i in R[0] if i not in user_data_with_cat_of_items['index'].values]
    print(df_cat_per_item['item_cats'][recommendations])
    return 'return'


def get_cat_per_item():
    sql = 'SELECT id, tour_id, activity_id FROM activity_tour where stt=1;'
    ds = read_data_from_db(sql, None)
    df_cat_per_item = ds.groupby('tour_id')['activity_id'].agg(_concatenate_cats_of_item)
    df_cat_per_item.name = 'item_cats'
    df_cat_per_item = df_cat_per_item.reset_index()
    df_cat_per_item[~df_cat_per_item.item_cats.isnull()].reset_index(drop=True)
    return df_cat_per_item