from flask import Blueprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def _concatenate_cats_of_item(cats):
    cats_as_str = ' '.join(set(map(str, cats)))
    return cats_as_str


def similar_to_item(df_item):
    tf_idf = TfidfVectorizer()
    df_items_tf_idf_cats = tf_idf.fit_transform(df_item.item_cats)
    cosine_sim = cosine_similarity(df_items_tf_idf_cats)
    df_tfidf_m2m = pd.DataFrame(cosine_sim)
    return df_tfidf_m2m
    

def similar_to_user_profile(user_data_with_cat_of_items, df_cat_per_item):
    max_times = user_data_with_cat_of_items['times'].max()
    user_data_with_cat_of_items['weight'] = user_data_with_cat_of_items['times']/max_times
    tf_idf = TfidfVectorizer()
    df_items_tf_idf_cats = tf_idf.fit_transform(df_cat_per_item.item_cats)
    user_profile = np.dot(df_items_tf_idf_cats[user_data_with_cat_of_items['index'].values].toarray().T, 
            user_data_with_cat_of_items['weight'].values)
    # print(df_items_tf_idf_cats[user_data_with_cat_of_items['index'].values].toarray().T.shape)
    # print(user_data_with_cat_of_items['weight'].values.shape)
    # print(user_profile.shape)
    # print(df_items_tf_idf_cats.shape)
    # print(np.atleast_2d(user_profile).shape)
    C = cosine_similarity(np.atleast_2d(user_profile), df_items_tf_idf_cats)
    # print(C.shape)
    R = np.argsort(C)[:, ::-1]
    recommendations = [i for i in R[0] if i not in user_data_with_cat_of_items['index'].values][:50]
    # print(C.loc[recommendations])
    return recommendations, pd.DataFrame(C[0][recommendations])