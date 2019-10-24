#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation
"""
import os
import sys
import warnings

import numpy as np
import ujson as json

from local_ancillary import (from_csv, local_stats_to_dict_numba,
                             mean_and_len_y, multiply, stats_calculation,
                             to_csv, vbm_parser)
from nipype_utils import average_nifti
from parsers import parse_for_categorical
from rw_utils import write_file
from utils import list_recursive

warnings.simplefilter("ignore")


def local_0(args):
    """ The first function in the local computation chain
    """
    input_ = args["input"]
    state_ = args["state"]
    cache_dir = state_["cacheDirectory"]

    threshold = input_["threshold"]
    voxel_size = input_["voxel_size"]
    lamb = args['input']['lambda']

    # Gather categorical info
    categorical_dict = parse_for_categorical(args)
    categorical_dict_count = {
        key: len(val)
        for key, val in categorical_dict.items()
    }

    covar_x = average_nifti(args)

    write_file(args, args, 'cache', 'args_file')

    to_csv(covar_x, os.path.join(cache_dir, 'X_df'))

    output_dict = {
        "categorical_dict": categorical_dict,
        "categorical_dict_count": categorical_dict_count,
        "threshold": threshold,
        "voxel_size": voxel_size,
        "avg_nifti": "avg_nifti.nii",
        "computation_phase": "local_0"
    }
    cache_dict = {
        "covariates": 'X_df',
        "voxel_size": voxel_size,
        "lambda": lamb
    }

    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output_dict)


def local_1(args):
    """Computes local beta vector

    Args:
        args (dictionary) : {"input": {
                                "covariates": ,
                                 "data": ,
                                 "lambda": ,
                                },
                            "cache": {}
                            }

    Returns:
        computation_output(json) : {"output": {
                                        "beta_vector_local": ,
                                        "mean_y_local": ,
                                        "count_local": ,
                                        "computation_phase":
                                        },
                                    "cache": {
                                        "covariates": ,
                                        "dependents": ,
                                        "lambda":
                                        }
                                    }

    Comments:
        Step 1 : Generate the local beta_vector
        Step 2 : Compute mean_y_local and length of target values

    """
    cache_ = args["cache"]
    state_ = args["state"]
    #    output_dir = state_["transferDirectory"]
    cache_dir = state_["cacheDirectory"]

    X = from_csv(os.path.join(cache_dir, cache_['covariates']))
    regularizer_l2 = cache_['lambda']

    # Local Statistics (with global_drop_col)
    encoded_X1, y = vbm_parser(args, X)
    meanY_vector, lenY_vector = mean_and_len_y(y)
    beta_vector1, local_stats_list = local_stats_to_dict_numba(
        args, encoded_X1, y)
    X_labels = list(encoded_X1.columns)

    np.save(os.path.join(cache_dir, 'X.npy'), encoded_X1)
    np.save(os.path.join(cache_dir, 'y.npy'), y)

    output_dict = {
        "beta_vector_local": beta_vector1,
        "mean_y_local": meanY_vector.tolist(),
        "count_local": lenY_vector.tolist(),
        "local_stats_list": local_stats_list,
        "X_labels": X_labels,
        "lambda": regularizer_l2
    }
    cache_dict = {"covariates": "X.npy", "dependents": "y.npy"}

    write_file(args, output_dict, 'output', 'local_output')

    computation_output_dict = {
        "output": {
            "computation_phase": "local_1"
        },
        "cache": cache_dict,
    }

    return json.dumps(computation_output_dict)


def local_2(args):
    """Computes the SSE_local, SST_local and varX_matrix_local
    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }
    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }
    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local
    """
    input_ = args["input"]
    cache_ = args["cache"]
    state_ = args["state"]
    cache_dir = state_["cacheDirectory"]

    biased_X = np.load(os.path.join(cache_dir, cache_["covariates"]))
    y = np.load(os.path.join(cache_dir, cache_["dependents"]))

    #    avg_beta_vector = input_["avg_beta_vector"]
    #    mean_y_global = input_["mean_y_global"]

    avg_beta_vector = np.load(
        os.path.join(args["state"]["baseDirectory"],
                     input_["avg_beta_vector"]))

    mean_y_global = np.load(
        os.path.join(args["state"]["baseDirectory"], input_["mean_y_global"]))

    varX_matrix_local = multiply(biased_X, biased_X)

    sse_local, sst_local = stats_calculation(biased_X, y, avg_beta_vector,
                                             mean_y_global)

    output_dict = {
        "SSE_local": sse_local.tolist(),
        "SST_local": sst_local.tolist(),
        "varX_matrix_local": varX_matrix_local.tolist()
    }

    write_file(args, output_dict, 'output', 'local_output')

    output_dict = {"computation_phase": 'local_2'}
    cache_dict = {}
    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output_dict)


if __name__ == '__main__':

    PARAM_DICT = json.loads(sys.stdin.read())
    PHASE_KEY = list(list_recursive(PARAM_DICT, 'computation_phase'))

    if not PHASE_KEY:
        sys.stdout.write(local_0(PARAM_DICT))
    elif "remote_0" in PHASE_KEY:
        sys.stdout.write(local_1(PARAM_DICT))
    elif "remote_1" in PHASE_KEY:
        sys.stdout.write(local_2(PARAM_DICT))
    else:
        raise ValueError("Error occurred at Local")
