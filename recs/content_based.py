from flask import Blueprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def _concatenate_cats_of_item(cats):
    cats_as_str = ' '.join(set(map(str, cats)))
    return cats_as_str


def get_item_profile(df_item):
    tf_idf = TfidfVectorizer()
    df_items_tf_idf_cats = tf_idf.fit_transform(df_item.item_cats)
    cosine_sim = cosine_similarity(df_items_tf_idf_cats)
    df_tfidf_m2m = pd.DataFrame(cosine_sim)
    return df_tfidf_m2m
    
