import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np


def impute_with_bin_mean(data, train_data, test_data, feature, bin_feature, bins=40, decimals=0, verbose=False):
    tr = train_data.copy()
    te = test_data.copy()
    database = pd.concat([tr, te])
    database["bin"], bins = pd.cut(database[bin_feature], bins=40, retbins=True)
    pb = 0
    for b in bins:
        bin_data = database[(database[bin_feature] >= pb) & (database[bin_feature] <= b)]
        bin_mean = bin_data[feature].mean()
        percentage = (bin_mean / bin_data[bin_feature]).mean()
        d = data[(data[bin_feature] >= pb) & (data[bin_feature] <= b)]
        ind = d[d[feature].isna()].index
        if pd.notna(bin_mean) and len(ind) > 0:
            bin_mean = round(bin_mean, decimals)
            data.at[ind, feature] = bin_mean
            if verbose:
                print(
                    "Set {} for {} rows with {} <= area_living <= {} to {}".format(feature, len(ind), pb, b, bin_mean))
        pb = b


def impute_nearest_neighbour_mean(data, database, feature, k, nn_features, decimals=0):
    kd_tree = KDTree(database[nn_features])
    d = data[data[feature].isna()]
    for ind, row in d.iterrows():
        neighbours = kd_tree.query([row[nn_features]], k=k, return_distance=False)[0]
        neighbour_mean = database.loc[neighbours][feature].mean()
        if not pd.isna(neighbour_mean):
            data.at[ind, feature] = round(neighbour_mean, decimals)
