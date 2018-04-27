# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:12:05 2018

@author: Harsh
"""
import nibabel as nib
from nilearn import plotting
from nilearn.input_data import NiftiMasker
from nistats.utils import z_score
from nistats.thresholding import map_threshold
import numpy as np
import os
import pandas as pd


def make_params_images(args, X, params):
    mask_file = os.path.join(args["state"]["baseDirectory"], 'mask_6mm.nii')
    mask = nib.load(mask_file).get_data()

    df = pd.DataFrame(params, columns=X.columns)

    for column in df.columns:
        image_string_sequence = ('params', column, args["state"]["clientId"])
        image_string = '_'.join(image_string_sequence) + '.nii'
        new_data = np.zeros(mask.shape)

        new_data[mask.get_data() > 0] = params[column]
        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        pdata = nib.load(os.path.join(curr_work_folder, file)).get_data()
        pvals = pdata[mask_array > 0]
        nifti_masker = NiftiMasker(
            smoothing_fwhm=5, memory='nilearn_cache', memory_level=1)
        nifti_masker.fit_transform(mask)
        z_map = niftimasker.inverse_transform(
            z_score(np.power(10, -np.abs(pvals))))
        threshold1 = fdr_threshold(np.power(10, -np.abs(pvals)), 0.05)
        thresholded_map2, threshold2 = map_threshold(
            z_map, threshold=.05, height_control='fdr')
        print(file, threshold1, threshold2)

        # According to internet
        plotting.plot_stat_map(
            thresholded_map2,
            threshold=threshold2,
            title='Thresholded z map, expected fdr = .05',
            output_file=os.path.join(curr_work_folder,
                                     os.path.splitext(file)[0] + '_01'))

        return 0