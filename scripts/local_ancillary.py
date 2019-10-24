#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:28:11 2018

@author: Harshvardhan
"""
import os
import warnings

import numpy as np
import pandas as pd
import scipy as sp
from numba import jit, prange

from ancillary import encode_png, print_beta_images, print_pvals
from nipype_utils import nifti_to_data
from parsers import perform_encoding

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm

MASK = os.path.join('/computation', 'mask_4mm.nii')


def mean_and_len_y(y):
    """Caculate the mean and length of each y vector
    """
    meanY_vector = y.mean(axis=0)
    #    lenY_vector = y.count(axis=0)
    lenY_vector = np.count_nonzero(~np.isnan(y), axis=0)

    return meanY_vector, lenY_vector


def to_csv(df, path):
    b = pd.Series(df.dtypes, name='data_types')
    df = df.append(b)
    df.to_csv(path)


def from_csv(path):
    df = pd.read_csv(path, index_col=0)
    type_dict = df.iloc[-1].to_dict()
    df = df.drop(df.tail(1).index)
    df = df.astype(type_dict)
    return df


@jit(nopython=True)
def multiply(array_a, array_b):
    """Multiplies two matrices"""
    return array_a.T @ array_b


@jit(nopython=True)
def gather_local_stats(X, y):
    """Calculate local statistics"""
    size_y = y.shape[1]

    params = np.zeros((X.shape[1], size_y))
    sse = np.zeros(size_y)
    tvalues = np.zeros((X.shape[1], size_y))
    rsquared = np.zeros(size_y)

    for voxel in prange(size_y):
        curr_y = y[:, voxel]
        beta_vector = np.linalg.inv(X.T @ X) @ (X.T @ curr_y)
        params[:, voxel] = beta_vector

        curr_y_estimate = np.dot(beta_vector, X.T)

        SSE_global = np.linalg.norm(curr_y - curr_y_estimate)**2
        SST_global = np.sum(np.square(curr_y - np.mean(curr_y)))

        sse[voxel] = SSE_global
        r_squared_global = 1 - (SSE_global / SST_global)
        rsquared[voxel] = r_squared_global

        dof_global = len(curr_y) - len(beta_vector)

        MSE = SSE_global / dof_global
        var_covar_beta_global = MSE * np.linalg.inv(X.T @ X)
        se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
        ts_global = beta_vector / se_beta_global

        tvalues[:, voxel] = ts_global

    return (params, sse, tvalues, rsquared, dof_global)


def local_stats_to_dict_numba(args, X, y):
    """Wrap local statistics into a dictionary to be sent to the remote"""
    X_labels = list(X.columns)

    X1 = X.values.astype('float64')

    params, sse, tvalues, rsquared, dof_global = gather_local_stats(X1, y)

    pvalues = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)

    beta_vector = params.T.tolist()

    print_pvals(args, pvalues.T, tvalues.T, X_labels)
    print_beta_images(args, beta_vector, X_labels)

    local_stats_list = encode_png(args)

    return beta_vector, local_stats_list


def ignore_nans(X, y):
    """Removing rows containing NaN's in X and y"""

    if type(X) is pd.DataFrame:
        X_ = X.values.astype('float64')
    else:
        X_ = X

    if type(y) is pd.Series:
        y_ = y.values.astype('float64')
    else:
        y_ = y

    finite_x_idx = np.isfinite(X_).all(axis=1)
    finite_y_idx = np.isfinite(y_)

    finite_idx = finite_y_idx & finite_x_idx

    y_ = y_[finite_idx]
    X_ = X_[finite_idx, :]

    return X_, y_


def local_stats_to_dict_fsl(X, y):
    """Calculate local statistics"""
    y_labels = list(y.columns)

    biased_X = sm.add_constant(X)
    X_labels = list(biased_X.columns)

    local_params = []
    local_sse = []
    local_pvalues = []
    local_tvalues = []
    local_rsquared = []
    meanY_vector, lenY_vector = [], []

    for column in y.columns:
        curr_y = y[column]

        X_, y_ = ignore_nans(biased_X, curr_y)
        meanY_vector.append(np.mean(y_))
        lenY_vector.append(len(y_))

        # Printing local stats as well
        model = sm.OLS(y_, X_).fit()
        local_params.append(model.params)
        local_sse.append(model.ssr)
        local_pvalues.append(model.pvalues)
        local_tvalues.append(model.tvalues)
        local_rsquared.append(model.rsquared)

    keys = [
        "Coefficient", "Sum Square of Errors", "t Stat", "P-value",
        "R Squared", "covariate_labels"
    ]
    local_stats_list = []

    for index, _ in enumerate(y_labels):
        values = [
            local_params[index].tolist(), local_sse[index],
            local_tvalues[index].tolist(), local_pvalues[index].tolist(),
            local_rsquared[index], X_labels
        ]
        local_stats_dict = {key: value for key, value in zip(keys, values)}
        local_stats_list.append(local_stats_dict)

        beta_vector = [l.tolist() for l in local_params]

    return beta_vector, local_stats_list, meanY_vector, lenY_vector


def add_site_covariates(args, X):
    """Add site specific columns to the covariate matrix"""
    biased_X = sm.add_constant(X)
    site_covar_list = args["input"]["site_covar_list"]

    site_matrix = np.zeros((np.array(X).shape[0], len(site_covar_list)),
                           dtype=int)
    site_df = pd.DataFrame(site_matrix, columns=site_covar_list)

    select_cols = [
        col for col in site_df.columns
        if args["state"]["clientId"] in col[len('site_'):]
    ]

    site_df[select_cols] = 1

    biased_X.reset_index(drop=True, inplace=True)
    site_df.reset_index(drop=True, inplace=True)

    augmented_X = pd.concat([biased_X, site_df], axis=1)

    return augmented_X


def vbm_parser(args, X):
    """Parse the nifti (.nii) specific inputspec.json and return the
    covariate matrix (X) as well the dependent matrix (y) as dataframes
    """
    y_info = nifti_to_data(args, X)
    encoded_covar_info = perform_encoding(args, X)

    return (encoded_covar_info, y_info)


@jit(nopython=True)
def stats_calculation(X, y, avg_beta_vec, mean_y_global):
    """Calculate SSE and SST."""
    size_y = y.shape[1]
    sse_local = np.zeros(size_y)
    sst_local = np.zeros(size_y)

    for voxel in range(y.shape[1]):
        y1 = y[:, voxel]
        beta = avg_beta_vec[voxel]
        mean_y = mean_y_global[voxel]

        y1_estimate = np.dot(beta, X.T)
        sse_local[voxel] = np.linalg.norm(y1 - y1_estimate)**2
        sst_local[voxel] = np.sum(np.square(y1 - mean_y))

    return sse_local, sst_local
