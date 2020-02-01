from flask import Flask
import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from sqlalchemy import create_engine
import pymysql

app = Flask(__name__)


def read_data_from_db(sql):
    db_connection_str = 'mysql+pymysql://app_root:mysql@12345678@db4free.net/hoian_travel'
    db_connection = create_engine(db_connection_str, pool_recycle=3600)

    df = pd.read_sql(sql, db_connection)
    pd.read_sql(sql, db_connection)
    pd.set_option('display.expand_frame_repr', False)

    return df


@app.route('/tour/<int:user_id>')
def recommend_tour(user_id):
    sql = 'SELECT user_id, tour_id, rating FROM rating_tour'
    ds = read_data_from_db(sql)

    reader = Reader()
    data = Dataset.load_from_df(ds[['user_id', 'tour_id', 'rating']], reader=reader)
    alg = SVD()
    alg.fit(data.build_full_trainset())

    iids = ds['tour_id'].unique()
    rated_iids = ds.loc[ds['user_id'] == user_id, 'tour_id']
    iids_to_pred = np.setdiff1d(iids, rated_iids)
    testset = [[user_id, iid, 4.] for iid in iids_to_pred]
    predictions = alg.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    for i in range(20):
        print(predictions[i])
    return ""


@app.route('/place/<int:user_id>')
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


@app.route('/restaurant/<int:user_id>')
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
    for i in range(20):
        print(predictions[i])
    return ""


if __name__=='__main__':
    app.run(debug=True, port=8080)