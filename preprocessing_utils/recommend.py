import pandas as pd
import torch
import os
from tqdm.notebook import tqdm
import numpy as np
import re

from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.stats import pearsonr
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from math import pi, cos

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from math import pi, cos
import lightgbm as lgbm
import numpy as np
from numpy.random import RandomState

from scipy.stats import kurtosis


def weighted_mean(x):
    counts = x.value_counts()
    total = counts.sum()
    weighted_sum = sum(score * count for score, count in counts.items())
    return weighted_sum / total

# 前処理コード
def scaling(data):
    sc = StandardScaler()
    data_sc = np.log1p(data)
    data_sc = sc.fit_transform(data_sc)
    return data_sc

def all_similarity(input_vect):
    # Calculate cosine similarity
    cos_sim = cosine_similarity(input_vect)

    # Calculate Euclidean distance
    euclidean_dist = cdist(input_vect, input_vect, metric='euclidean')

    # Calculate Manhattan distance
    manhattan_dist = cdist(input_vect, input_vect, metric='cityblock')

    # Calculate Jaccard similarity
    jaccard_sim = 1 - pairwise_distances(input_vect, metric='jaccard')

    # Calculate Pearson correlation coefficient
    pearson_corr = np.corrcoef(input_vect, rowvar=True)

    all_dis = np.concatenate([cos_sim,euclidean_dist,manhattan_dist,jaccard_sim,pearson_corr],axis=1)

    return all_dis

def to_minutes(s):

    match = re.search(r'(\d+) hr.*? (\d+) min.', s)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    match = re.search(r'(\d+) min', s)
    if match:
        return int(match.group(1))
    return 1 # for Unknown

def merge_by_anime_id(left_df, right_df):
    return pd.merge(left_df, right_df, on="anime_id", how="left").drop(columns=["anime_id","user_id"])

def merge_by_anime_id_with(left_df, right_df):
    return pd.merge(left_df, right_df, on="anime_id", how="left")


def merge_by_user_id_with(left_df, right_df):
    return pd.merge(left_df, right_df, on="user_id", how="left")

# train_scores = train_df.groupby('anime_id')['score'].apply(weighted_mean)
def create_anime_numeric_feature(input_df: pd.DataFrame):
    """input_dfは train or test.csv のデータが入ってくることを想定しています."""
    
    use_columns = [
        "members", 
        "watching","completed","on_hold","dropped","plan_to_watch"
    ]
    return merge_by_anime_id(input_df, anime_df)[use_columns]
    # return pd.merge(input_df, anime_df, on="anime_id", how="left").drop(columns=["anime_id","user_id"])


def create_anime_type_one_hot_encoding(input_df):
    
    target_colname = "type"
    target_series = anime_df[target_colname]
    unique_values = target_series.unique()

    out_df = pd.DataFrame()
    for value in unique_values:
        is_value = target_series == value
        out_df[value] = is_value.astype(int)
        
    out_df["anime_id"] = anime_df["anime_id"]
    
    return merge_by_anime_id(input_df, out_df)
def type(input_df):
    # 単純にラベルエンコーディング
    encoder = LabelEncoder()
    type_encoder = anime_df[['anime_id']].copy()
    type_encoder["type_label"] = encoder.fit_transform(anime_df["type"])
    return merge_by_anime_id(input_df, type_encoder)

def create_anime_type_one_hot_encoding(input_df):
    
    target_colname = "type"
    target_series = anime_df[target_colname]
    unique_values = target_series.unique()

    out_df = pd.DataFrame()
    for value in unique_values:
        is_value = target_series == value
        out_df[value] = is_value.astype(int)
        
    out_df["anime_id"] = anime_df["anime_id"]
    
    return merge_by_anime_id(input_df, out_df)

def create_anime_type_count_encoding(input_df):
    count = anime_df["type"].map(anime_df["type"].value_counts())
    encoded_df = pd.DataFrame({
        "anime_id": anime_df["anime_id"],
        "tyoe_count": count
    })
    
    return merge_by_anime_id(input_df, encoded_df)

def create_licensors_count_encoding(input_df):
    count = anime_df["licensors"].map(anime_df["licensors"].value_counts())
    encoded_df = pd.DataFrame({
        "anime_id": anime_df["anime_id"],
        "licensors_count": count
    })
    
    return merge_by_anime_id(input_df, encoded_df)

def studios(input_df):
    # 単純にラベルエンコーディング
    encoder = LabelEncoder()
    studios_encode = anime_df[['anime_id']].copy()
    studios_encode["studios"] = encoder.fit_transform(anime_df["studios"])
    return merge_by_anime_id(input_df, studios_encode)

def create_studio_count_encoding(input_df):
    count = anime_df["studios"].map(anime_df["studios"].value_counts())
    encoded_df = pd.DataFrame({
        "anime_id": anime_df["anime_id"],
        "studios_count": count
    })
    
    return merge_by_anime_id(input_df, encoded_df)

def create_source_count_encoding(input_df):
    count = anime_df["source"].map(anime_df["source"].value_counts())
    encoded_df = pd.DataFrame({
        "anime_id": anime_df["anime_id"],
        "source_count": count
    })
    
    return merge_by_anime_id(input_df, encoded_df)

def source(input_df):
    # 単純にラベルエンコーディング
    encoder = LabelEncoder()
    source_encode = anime_df[['anime_id']].copy()
    source_encode["source"] = encoder.fit_transform(anime_df["source"])
    return merge_by_anime_id(input_df, source_encode)

def create_rating_count_encoding(input_df):
    count = anime_df["rating"].map(anime_df["rating"].value_counts())
    encoded_df = pd.DataFrame({
        "anime_id": anime_df["anime_id"],
        "rating_count": count
    })
    
    return merge_by_anime_id(input_df, encoded_df)

def create_genres_onehot_encoding(input_df):
    """Create 26-dim embedding"""
    chars = ['Comedy', 'Sci-Fi', 'Seinen', 'Slice of Life', 'Space',
       'Adventure', 'Mystery', 'Historical', 'Supernatural', 'Fantasy',
       'Ecchi', 'School', 'Harem', 'Romance', 'Shounen', 'Action',
       'Magic', 'Sports', 'Super Power', 'Drama', 'Thriller', 'Music',
       'Shoujo', 'Demons', 'Mecha', 'Game', 'Josei', 'Cars',
       'Psychological', 'Parody', 'Samurai', 'Military', 'Shoujo Ai',
       'Kids', 'Martial Arts', 'Horror', 'Dementia', 'Vampire',
       'Shounen Ai', 'Hentai', 'Yaoi', 'Police']
    genres = anime_df[['anime_id','genres']]
    genres.loc[:,chars] = 0
    genres['genres'] = genres['genres'].str.split(',')
    # genres[chars] = 0
    for i, row in genres.iterrows():
        for index in (s.strip() for s in row['genres']):
                genres.loc[i,index] = 1
    genres = genres.drop('genres',axis=1)
    return merge_by_anime_id(input_df, genres)


def create_producer_onehot_encoding(input_df):
    """Create 26-dim embedding"""
    producer = anime_df['producers'].str.split(',')
    all_producer = list(map(str.strip,producer.explode().unique()))
    producer = anime_df[['anime_id']]
    producer.loc[:,all_producer] = 0
    producer['producers'] = anime_df['producers'].str.split(',')
    # genres[chars] = 0
    for i, row in producer.iterrows():
        for index in (s.strip() for s in row['producers']):
                producer.loc[i,index] = 1
    
    return merge_by_anime_id(input_df, producer).drop('producers',axis=1)


def create_anime_type_one_hot_encoding(input_df):
    
    target_colname = "type"
    target_series = anime_df[target_colname]
    unique_values = target_series.unique()

    out_df = pd.DataFrame()
    for value in unique_values:
        is_value = target_series == value
        out_df[value] = is_value.astype(int)
        
    out_df["anime_id"] = anime_df["anime_id"]
    
    return merge_by_anime_id(input_df, out_df)

def create_duration2min(input_df):
    time_min = anime_df[['anime_id']]
    time_min['minutes'] = anime_df['duration'].apply(to_minutes)
    return merge_by_anime_id(input_df, time_min)
from datetime import datetime
def to_year(s):
    match = re.search(r'\d{4}', s)
    if match:
       return int(match.group())
    else:
        return 1000
    
def year_pre(input_df):
    year = anime_df[['anime_id']]
    encoder = LabelEncoder()
    year['year'] = encoder.fit_transform(anime_df['aired'].apply(to_year))
    
    return merge_by_anime_id(input_df, year)

def merge_embedding(input_df):
    embeds = np.load("train/anime_jpbert_embeddings.npy")
    ids = np.load("train/anime_jpbert_ids.npy")
    embeds_list = []
    for l in range(embeds.shape[0]):
        embeds_list.append(embeds[l,:])
    anime_embed = pd.DataFrame(data={"anime_id": ids, "embed" : embeds_list})
    embed_array = np.array(anime_embed["embed"].tolist())

    for i in range(768):
        anime_embed[f"embed_{i}"] = embed_array[:,i]

    anime_embed.drop("embed", axis=1, inplace=True)
    return merge_by_anime_id(input_df, anime_embed)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist, squareform
import pickle
from gensim.models import word2vec,Word2Vec


def create_tfidf_matrix(input_df,mode='anime'):
    text_columns = ['genres', 'japanese_name', 'type', 'episodes', 'aired', 'producers', 'licensors', 'studios', 'source', 'duration', 'rating']

    tf = TfidfVectorizer()
    try:
        tf = pickle.load(open("vectorizer.pickle", "rb"))
        print('loaded tf')
    except:
        tf.fit(anime_df[text_columns].astype(str).apply(lambda x: ';'.join(x), axis=1).tolist()) 
        pickle.dump(tf, open("vectorizer.pickle", "wb"))
    tfidf_matrix = tf.fit_transform(anime_df[text_columns].astype(str).apply(lambda x: ';'.join(x), axis=1).tolist())
    cosine_sim = all_similarity(tfidf_matrix.toarray()) 
    svd = TruncatedSVD(n_components=30,random_state=42)

    svd_arr = svd.fit_transform(cosine_sim)
    print('tfdf:',svd.explained_variance_ratio_.sum())
    
    col_df = pd.DataFrame(
    svd_arr,
    index=anime_df['anime_id'],
    columns=[f"svd_{mode}_{ix}" for ix in range(30)],
    ).reindex()
    return merge_by_anime_id_with(input_df, col_df)

def create_tf_user_anime_vector(input_df):
    df = pd.concat([train_df,test_df])
    matrix = create_tfidf_matrix(df,mode='user_mean').drop(['anime_id','score'],axis=1)
    user_matrix_mean = matrix.groupby('user_id').mean()
    user_matrix_std = matrix.groupby('user_id').std()
    user_matrix_skew = matrix.groupby('user_id').skew()
    user_matrix_var = matrix.groupby('user_id').var()
    
    
    svd = TruncatedSVD(n_components=20,random_state=42)
    mean_cos = svd.fit_transform(all_similarity(user_matrix_mean.values))
    print('mean_user:',svd.explained_variance_ratio_.sum())
    
    
    user_mean_cos_matrix = pd.DataFrame(mean_cos,
                          index=user_matrix_mean.index.values.tolist(),
                          columns=[f"w2v_mean_user_cos_{i}" for i in range(mean_cos.shape[1])]
                          )
    
    
    user_matrix = pd.concat([user_matrix_mean,user_matrix_std,user_matrix_skew,user_matrix_var,user_mean_cos_matrix],axis=1).reindex()
    user_matrix.index.name='user_id'
    return merge_by_user_id_with(input_df, user_matrix).drop('user_id',axis=1)

def create_llma_anime_vector(input_df,embed_source,mode='anime'):
    embeddings = np.load(f"./train/{embed_source}.npy")
    df_anime_preprocessd = anime_df[['anime_id']].copy()
    svd = TruncatedSVD(n_components=20,random_state=42)
    svd_arr = svd.fit_transform(all_similarity(embeddings))
    
    embeddings_columns = [f"{embed_source}_{mode}_{i}" for i in range(svd_arr.shape[1])]
    embeddings_df = pd.DataFrame(data=svd_arr, columns=embeddings_columns)
    df_anime_preprocessd = df_anime_preprocessd.join(embeddings_df)
    
    return merge_by_anime_id_with(input_df, df_anime_preprocessd)

def create_llma_anime_embed(input_df,embed_source,mode='anime'):
    embeddings = np.load(f"./train/{embed_source}.npy")
    df_anime_preprocessd = anime_df[['anime_id']].copy()
    svd = TruncatedSVD(n_components=20,random_state=42)
    svd_arr = svd.fit_transform(embeddings)
    
    embeddings_columns = [f"{embed_source}_{mode}_embed_{i}" for i in range(svd_arr.shape[1])]
    embeddings_df = pd.DataFrame(data=svd_arr, columns=embeddings_columns)
    df_anime_preprocessd = df_anime_preprocessd.join(embeddings_df)
    
    return merge_by_anime_id_with(input_df, df_anime_preprocessd)


def create_01EDA_user_vector(input_df):
    df = pd.concat([train_df,test_df])
    matrix = create_llma_anime_vector(df,embed_source='mBERT_embedding_01EDA',mode='user_mean').drop(['anime_id','score'],axis=1)
    user_matrix_mean = matrix.groupby('user_id').mean()
    user_matrix_std = matrix.groupby('user_id').std()
    user_matrix_skew = matrix.groupby('user_id').skew()
    user_matrix_var = matrix.groupby('user_id').var()    
    
    svd = TruncatedSVD(n_components=20,random_state=42)
    mean_cos = svd.fit_transform(all_similarity(user_matrix_mean.values))
    print('mean_user:',svd.explained_variance_ratio_.sum())
    
    
    user_mean_cos_matrix = pd.DataFrame(mean_cos,
                          index=user_matrix_mean.index.values.tolist(),
                          columns=[f"w2v_mean_user_cos_{i}" for i in range(mean_cos.shape[1])]
                          )
     
    user_matrix = pd.concat([user_matrix_mean,user_matrix_std,user_matrix_skew,user_matrix_var,user_mean_cos_matrix],axis=1).reindex()
    user_matrix.index.name='user_id'
    
    return merge_by_user_id_with(input_df, user_matrix).drop('user_id',axis=1)

def create_large_user_vector(input_df):
    df = pd.concat([train_df,test_df])
    matrix = create_llma_anime_vector(df,embed_source='mBERT_embedding_large',mode='user_mean').drop(['anime_id','score'],axis=1)
    user_matrix_mean = matrix.groupby('user_id').mean()
    user_matrix_std = matrix.groupby('user_id').std()
    user_matrix_skew = matrix.groupby('user_id').skew()
    user_matrix_var = matrix.groupby('user_id').var()
    # user_matrix_sum = matrix.groupby('user_id').sum()
    
    
    svd = TruncatedSVD(n_components=20,random_state=42)
    mean_cos = svd.fit_transform(all_similarity(user_matrix_mean.values))
    print('mean_user:',svd.explained_variance_ratio_.sum())
    
    # svd = TruncatedSVD(n_components=20,random_state=42)
    # sum_cos = svd.fit_transform(cosine_similarity(user_matrix_sum))
    # print('sum_user:',svd.explained_variance_ratio_.sum())
    
    user_mean_cos_matrix = pd.DataFrame(mean_cos,
                          index=user_matrix_mean.index.values.tolist(),
                          columns=[f"w2v_mean_user_cos_{i}" for i in range(mean_cos.shape[1])]
                          )
    
    # user_sum_cos_matrix = pd.DataFrame(sum_cos,
    #                       index=user_matrix_sum.index.values.tolist(),
    #                       columns=[f"w2v_sum_user_cos_{i}" for i in range(sum_cos.shape[1])]
    #                       )
    # user_embed_mean = create_llma_anime_embed(df,mode='user_mean',embed_source='mBERT_embedding_large').drop(['anime_id','score'],axis=1).groupby('user_id').mean()
    # user_embed_std = create_llma_anime_embed(df,mode='user_std',embed_source='mBERT_embedding_large').drop(['anime_id','score'],axis=1).groupby('user_id').std()
    # user_embed_skew = create_llma_anime_embed(df,mode='user_skew',embed_source='mBERT_embedding_large').drop(['anime_id','score'],axis=1).groupby('user_id').skew()
    # user_embed_var = create_llma_anime_embed(df,mode='user_var',embed_source='mBERT_embedding_large').drop(['anime_id','score'],axis=1).groupby('user_id').var()
    
    user_matrix = pd.concat([user_matrix_mean,user_matrix_std,user_matrix_skew,user_matrix_var,user_mean_cos_matrix],axis=1).reindex()
    user_matrix.index.name='user_id'
    
    return merge_by_user_id_with(input_df, user_matrix).drop('user_id',axis=1)

def create_small_user_vector(input_df):
    df = pd.concat([train_df,test_df])
    matrix = create_llma_anime_vector(df,embed_source='mBERT_embedding_small',mode='user_mean').drop(['anime_id','score'],axis=1)
    user_matrix_mean = matrix.groupby('user_id').mean()
    user_matrix_std = matrix.groupby('user_id').std()
    user_matrix_skew = matrix.groupby('user_id').skew()
    user_matrix_var = matrix.groupby('user_id').var()
    # user_matrix_sum = matrix.groupby('user_id').sum()
    
    
    svd = TruncatedSVD(n_components=20,random_state=42)
    mean_cos = svd.fit_transform(all_similarity(user_matrix_mean.values))
    print('mean_user:',svd.explained_variance_ratio_.sum())
    
    # svd = TruncatedSVD(n_components=20,random_state=42)
    # sum_cos = svd.fit_transform(cosine_similarity(user_matrix_sum))
    # print('sum_user:',svd.explained_variance_ratio_.sum())
    
    user_mean_cos_matrix = pd.DataFrame(mean_cos,
                          index=user_matrix_mean.index.values.tolist(),
                          columns=[f"w2v_mean_user_cos_{i}" for i in range(mean_cos.shape[1])]
                          )
    
    # user_sum_cos_matrix = pd.DataFrame(sum_cos,
    #                       index=user_matrix_sum.index.values.tolist(),
    #                       columns=[f"w2v_sum_user_cos_{i}" for i in range(sum_cos.shape[1])]
    #                       )
    # user_embed_mean = create_llma_anime_embed(df,mode='user_mean',embed_source='mBERT_embedding_small').drop(['anime_id','score'],axis=1).groupby('user_id').mean()
    # user_embed_std = create_llma_anime_embed(df,mode='user_std',embed_source='mBERT_embedding_small').drop(['anime_id','score'],axis=1).groupby('user_id').std()
    # user_embed_skew = create_llma_anime_embed(df,mode='user_skew',embed_source='mBERT_embedding_small').drop(['anime_id','score'],axis=1).groupby('user_id').skew()
    # user_embed_var = create_llma_anime_embed(df,mode='user_var',embed_source='mBERT_embedding_small').drop(['anime_id','score'],axis=1).groupby('user_id').var()
    
    user_matrix = pd.concat([user_matrix_mean,user_matrix_std,user_matrix_skew,user_matrix_var,user_mean_cos_matrix],axis=1).reindex()
    user_matrix.index.name='user_id'
   
    return merge_by_user_id_with(input_df, user_matrix).drop('user_id',axis=1)


def create_01EDA_anime_vector(input_df):
    return create_llma_anime_vector(input_df,embed_source='mBERT_embedding_01EDA').drop(columns=["anime_id","user_id"])
def create_large_anime_vector(input_df):
    return create_llma_anime_vector(input_df,embed_source='mBERT_embedding_large').drop(columns=["anime_id","user_id"])
def create_small_anime_vector(input_df):
    return create_llma_anime_vector(input_df,embed_source='mBERT_embedding_small').drop(columns=["anime_id","user_id"])

def create_01EDA_anime_embed(input_df):
    return create_llma_anime_embed(input_df,embed_source='mBERT_embedding_01EDA').drop(columns=["anime_id","user_id"])
def create_large_anime_embed(input_df):
    return create_llma_anime_embed(input_df,embed_source='mBERT_embedding_large').drop(columns=["anime_id","user_id"])
def create_small_anime_embed(input_df):
    return create_llma_anime_embed(input_df,embed_source='mBERT_embedding_small').drop(columns=["anime_id","user_id"])

def create_w2v_anime_vector(input_df,mode='anime'):
    text_columns = ['genres', 'japanese_name', 'type', 'episodes', 'aired', 'producers', 'licensors', 'studios', 'source']
    text_data = anime_df[text_columns].astype(str).apply(lambda x: ';;'.join(x), axis=1).tolist()

    text1 = [x.strip() for _ in text_data for x in re.split(';;',_) ] # テキストデータをコンマやセミコロンで分割して、空白を除去する
    text_data.append(text1)
    shuffled_sentence_list = [random.sample(sentence, len(sentence)) for sentence in text_data] 
    train_sentence_list = text_data + shuffled_sentence_list
    vector_size = 128
    w2v_params = {
        "vector_size": vector_size,  ## <= 変更点
        "seed": 42,
        "min_count": 1,
        "workers": 1
    }

    # word2vecのモデル学習
    try:
        model = Word2Vec.load('model/w2v.model')
    except:
        model = word2vec.Word2Vec(train_sentence_list, **w2v_params)
        model.save('model/w2v.model')
    vects = {}
    for i, row in anime_df.iterrows():
        vect = []
        for col in text_columns:
            vect.append(model.wv[row[col]])
        vects[row.anime_id] = np.concatenate(vect,axis=0)
    wv_matirx = pd.DataFrame(vects).T
    svd = TruncatedSVD(n_components=30,random_state=42)

    svd_cos = svd.fit_transform(all_similarity(wv_matirx.values))
    print('Word2Vec:',svd.explained_variance_ratio_.sum())
    
    wv_cos = pd.DataFrame(svd_cos,
                          index=wv_matirx.index.values.tolist(),
                          columns=[f"wv_{mode}_cos_vec_{i}" for i in range(svd_cos.shape[1])]
                          )
    wv_cos.index.name = 'anime_id'
    return merge_by_anime_id_with(input_df,wv_cos.reindex())

def create_w2v_anime_embed(input_df,mode='anime'):
    text_columns = ['genres', 'japanese_name', 'type', 'episodes', 'aired', 'producers', 'licensors', 'studios', 'source']
    text_data = anime_df[text_columns].astype(str).apply(lambda x: ';;'.join(x), axis=1).tolist()

    text1 = [x.strip() for _ in text_data for x in re.split(';;',_) ] # テキストデータをコンマやセミコロンで分割して、空白を除去する
    text_data.append(text1)
    shuffled_sentence_list = [random.sample(sentence, len(sentence)) for sentence in text_data] 
    train_sentence_list = text_data + shuffled_sentence_list
    vector_size = 128
    w2v_params = {
        "vector_size": vector_size,  ## <= 変更点
        "seed": 42,
        "min_count": 1,
        "workers": 1
    }

    # word2vecのモデル学習
    try:
        model = Word2Vec.load('model/w2v.model')
    except:
        model = word2vec.Word2Vec(train_sentence_list, **w2v_params)
        model.save('model/w2v.model')
    vects = {}
    for i, row in anime_df.iterrows():
        vect = []
        for col in text_columns:
            vect.append(model.wv[row[col]])
        vects[row.anime_id] = np.concatenate(vect,axis=0)
    wv_matirx = pd.DataFrame(vects).T
    svd = TruncatedSVD(n_components=20,random_state=42)

    svd_cos = svd.fit_transform(all_similarity(wv_matirx.values))
    print(svd.explained_variance_ratio_.sum())
    
    wv_cos = pd.DataFrame(svd_cos,
                          index=wv_matirx.index.values.tolist(),
                          columns=[f"wv_{mode}_embed_{i}" for i in range(svd_cos.shape[1])]
                          )
    wv_cos.index.name = 'anime_id'
    return merge_by_anime_id_with(input_df,wv_cos.reindex())

def create_w2v_user_vector(input_df):
    df = pd.concat([train_df,test_df])
    matrix = create_w2v_anime_vector(df,mode='user_mean').drop(['anime_id','score'],axis=1)
    user_matrix_mean = matrix.groupby('user_id').mean()
    user_matrix_std = matrix.groupby('user_id').std()
    user_matrix_skew = matrix.groupby('user_id').skew()
    user_matrix_var = matrix.groupby('user_id').var()
    # user_matrix_sum = matrix.groupby('user_id').sum()
    
    
    svd = TruncatedSVD(n_components=20,random_state=42)
    mean_cos = svd.fit_transform(all_similarity(user_matrix_mean.values))
    print('mean_user:',svd.explained_variance_ratio_.sum())
    
    # svd = TruncatedSVD(n_components=20,random_state=42)
    # sum_cos = svd.fit_transform(cosine_similarity(user_matrix_sum))
    # print('sum_user:',svd.explained_variance_ratio_.sum())
    
    user_mean_cos_matrix = pd.DataFrame(mean_cos,
                          index=user_matrix_mean.index.values.tolist(),
                          columns=[f"w2v_mean_user_cos_{i}" for i in range(mean_cos.shape[1])]
                          )
    
    # user_sum_cos_matrix = pd.DataFrame(sum_cos,
    #                       index=user_matrix_sum.index.values.tolist(),
    #                       columns=[f"w2v_sum_user_cos_{i}" for i in range(sum_cos.shape[1])]
    #                       )
    # user_embed_mean = create_w2v_anime_embed(df,mode='user_mean').drop(['anime_id','score'],axis=1).groupby('user_id').mean()
    # user_embed_std = create_w2v_anime_embed(df,mode='user_std').drop(['anime_id','score'],axis=1).groupby('user_id').std()
    # user_embed_skew = create_w2v_anime_embed(df,mode='user_skew').drop(['anime_id','score'],axis=1).groupby('user_id').skew()
    # user_embed_var = create_w2v_anime_embed(df,mode='user_var').drop(['anime_id','score'],axis=1).groupby('user_id').var()
    
    user_matrix = pd.concat([user_matrix_mean,user_matrix_std,user_matrix_skew,user_matrix_var,user_mean_cos_matrix],axis=1).reindex()
    user_matrix.index.name='user_id'
    return merge_by_user_id_with(input_df, user_matrix).drop('user_id',axis=1)

def create_feature(input_df):
    
    # functions に特徴量作成関数を配列で定義しました.
    # どの関数も同じ input / output のインターフェイスなので for で回せて嬉しいですね ;)
    functions = [
        create_tf_user_anime_vector,
        create_tfidf_matrix,
        create_w2v_anime_vector,
        # create_w2v_anime_embed,
        # num_user_vector,

        create_w2v_user_vector,
        create_01EDA_user_vector,
        create_large_user_vector,
        create_small_user_vector,

        create_anime_numeric_feature,
        create_anime_type_count_encoding,
        
        # create_anime_type_one_hot_encoding,
        create_genres_onehot_encoding,
        # create_producer_onehot_encoding,
        # create_licensors_count_encoding,
    
        create_studio_count_encoding,
        create_source_count_encoding,
        create_rating_count_encoding,
        create_duration2min,

        create_01EDA_anime_vector,
        create_large_anime_vector,
        create_small_anime_vector,
        # create_01EDA_anime_embed,
        # create_large_anime_embed,
        # create_small_anime_embed,
        year_pre,
    ]
    
    out_df = pd.DataFrame()
    for func in tqdm(functions):
        func_name = str(func.__name__)

        _df = func(input_df)
        out_df = pd.concat([out_df, _df], axis=1)
        
    return out_df