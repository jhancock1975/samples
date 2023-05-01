#!/usr/bin/env python3
import pprint
import socket
import pandas as pd
import sys
from our_util import next_seed, get_feature_selection_experiments, get_logger
import os
import numpy as np
from scipy.stats import randint, uniform, loguniform
import dill as pickle
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from copy import deepcopy
from category_encoders.cat_boost import CatBoostEncoder
import argparse
import logging
from our_util import get_logger
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    matthews_corrcoef
)
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import importlib
import gc
from imblearn.under_sampling import RandomUnderSampler
import math
from utils import (
    get_performance,
    get_best_threshold_by_gmean,
    get_best_threshold_by_fmeasure,
    get_best_threshold_by_mcc,
    get_best_threshold_by_precision,
)
import json


def np_encoder(object):
    """
    for pretty printing dictionaries
    https://stackoverflow.com/a/65151218/3220610
    """
    if isinstance(object, np.generic):
        return object.item()


# define here for reuse
threshold_names = ['f_measure', 'g_mean', 'mcc', 'class_prior', 'precision',
                   'default', 'f_measure_no_constraint', 'g_mean_no_constraint', 'mcc_no_constraint', 'precision_no_constraint']

# base directory for datasets
base_dir = '/home/groups/fau-bigdata-datasets/medicare/2019'
# base_dir = '/home/ubuntu/medicare-data/2019'
sample_dir = '/mnt/beegfs/home/jhancoc4/medicare-data/2019-samples'
# sample_dir = '/home/ubuntu/medicare-data/2019-samples'

# medicare features from
# https://github.com/jhancock1975/spring-2021-research/blob/main/param-tuning-expansion/sn-article.pdf
datasets_dict = {
    'part-b-aggregated': {
        'file_name': f'{base_dir}/medicare-partb-aggregated-2013-2019.csv.gz',
        'sample_file': f'{sample_dir}/medicare-partb-aggregated-2013-2019.csv',
        'target': 'exclusion',
        'features':  [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
            'Tot_Srvcs_mean',
            'Tot_Srvcs_median',
            'Tot_Srvcs_sum',
            'Tot_Srvcs_std',
            'Tot_Srvcs_min',
            'Tot_Srvcs_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benes_std',
            'Tot_Benes_min',
            'Tot_Benes_max',
            'Tot_Bene_Day_Srvcs_mean',
            'Tot_Bene_Day_Srvcs_median',
            'Tot_Bene_Day_Srvcs_sum',
            'Tot_Bene_Day_Srvcs_std',
            'Tot_Bene_Day_Srvcs_min',
            'Tot_Bene_Day_Srvcs_max',
            'Avg_Sbmtd_Chrg_mean',
            'Avg_Sbmtd_Chrg_median',
            'Avg_Sbmtd_Chrg_sum',
            'Avg_Sbmtd_Chrg_std',
            'Avg_Sbmtd_Chrg_min',
            'Avg_Sbmtd_Chrg_max',
            'Avg_Mdcr_Pymt_Amt_mean',
            'Avg_Mdcr_Pymt_Amt_median',
            'Avg_Mdcr_Pymt_Amt_sum',
            'Avg_Mdcr_Pymt_Amt_std',
            'Avg_Mdcr_Pymt_Amt_min',
            'Avg_Mdcr_Pymt_Amt_max',
        ],
        'cat_features': [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
        ]
    },
    'part-b': {
        'file_name': f'{base_dir}/medicare-partb-2013-2019.csv.gz',
        'sample_file': f'{sample_dir}/medicare-partb-2013-2019.csv',
        'target': 'exclusion',
        'features':  ['Rndrng_Prvdr_Crdntls',
                       'Rndrng_Prvdr_Gndr',
                       'Rndrng_Prvdr_Ent_Cd',
                       'Rndrng_Prvdr_Type',
                       'Rndrng_Prvdr_Mdcr_Prtcptg_Ind',
                       'HCPCS_Cd',
                       'HCPCS_Desc',
                       'HCPCS_Drug_Ind',
                       'Place_Of_Srvc',
                       'Tot_Benes',
                       'Tot_Srvcs',
                       'Tot_Bene_Day_Srvcs',
                       'Avg_Sbmtd_Chrg',
                       'Avg_Mdcr_Alowd_Amt',
                       'Avg_Mdcr_Pymt_Amt',
                       'Avg_Mdcr_Stdzd_Amt'],
        'cat_features': ['Rndrng_Prvdr_Crdntls',
                         'Rndrng_Prvdr_Type',
                         'Rndrng_Prvdr_Gndr',
                         'Rndrng_Prvdr_Ent_Cd',
                         'Rndrng_Prvdr_Mdcr_Prtcptg_Ind',
                         'HCPCS_Cd',
                         'HCPCS_Desc',
                         'HCPCS_Drug_Ind',
                         'Place_Of_Srvc'],
    },
    'part-b-aggregated-new': {
        'file_name': f'{base_dir}/medicare-partb-aggregated-new-features-2013-2019.csv.gz',
        'sample_file': f'{sample_dir}/medicare-partb-aggregated-new-features-2013-2019.csv',
        'target': 'exclusion',
        'features':  [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
            'Tot_Srvcs_mean',
            'Tot_Srvcs_median',
            'Tot_Srvcs_sum',
            'Tot_Srvcs_std',
            'Tot_Srvcs_min',
            'Tot_Srvcs_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benes_std',
            'Tot_Benes_min',
            'Tot_Benes_max',
            'Tot_Bene_Day_Srvcs_mean',
            'Tot_Bene_Day_Srvcs_median',
            'Tot_Bene_Day_Srvcs_sum',
            'Tot_Bene_Day_Srvcs_std',
            'Tot_Bene_Day_Srvcs_min',
            'Tot_Bene_Day_Srvcs_max',
            'Avg_Sbmtd_Chrg_mean',
            'Avg_Sbmtd_Chrg_median',
            'Avg_Sbmtd_Chrg_sum',
            'Avg_Sbmtd_Chrg_std',
            'Avg_Sbmtd_Chrg_min',
            'Avg_Sbmtd_Chrg_max',
            'Avg_Mdcr_Pymt_Amt_mean',
            'Avg_Mdcr_Pymt_Amt_median',
            'Avg_Mdcr_Pymt_Amt_sum',
            'Avg_Mdcr_Pymt_Amt_std',
            'Avg_Mdcr_Pymt_Amt_min',
            'Avg_Mdcr_Pymt_Amt_max',
            'Tot_HCPCS_Cds',
            'Tot_Benes',
            'Tot_Srvcs',
            'Tot_Sbmtd_Chrg',
            'Tot_Mdcr_Alowd_Amt',
            'Tot_Mdcr_Pymt_Amt',
            'Tot_Mdcr_Stdzd_Amt',
            'Drug_Tot_HCPCS_Cds',
            'Drug_Tot_Benes',
            'Drug_Tot_Srvcs',
            'Drug_Sbmtd_Chrg',
            'Drug_Mdcr_Alowd_Amt',
            'Drug_Mdcr_Pymt_Amt',
            'Drug_Mdcr_Stdzd_Amt',
            'Med_Tot_HCPCS_Cds',
            'Med_Tot_Benes',
            'Med_Tot_Srvcs',
            'Med_Sbmtd_Chrg',
            'Med_Mdcr_Alowd_Amt',
            'Med_Mdcr_Pymt_Amt',
            'Med_Mdcr_Stdzd_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre'
        ],
        'cat_features': [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
        ]
    },
    'part-b-aggregated-new-top-10-sfs-5': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Tot_Bene_Day_Srvcs_sum",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Benes_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Avg_Mdcr_Pymt_Amt_min",
            "Rndrng_Prvdr_Type",
        ],
        'cat_features': [
        ]
    },
     'part-b-aggregated-new-top-15-sfs-5': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Tot_Bene_Day_Srvcs_sum",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Benes_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Avg_Mdcr_Pymt_Amt_min",
            "Rndrng_Prvdr_Type",
            "Tot_Sbmtd_Chrg",
            "Med_Tot_Srvcs",
            "Avg_Sbmtd_Chrg_min",
            "Avg_Mdcr_Pymt_Amt_std",
            "Tot_HCPCS_Cds",
        ],
        'cat_features': [
        ]
    },
    'part-b-aggregated-new-top-20-sfs-5': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Tot_Bene_Day_Srvcs_sum",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Benes_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Avg_Mdcr_Pymt_Amt_min",
            "Rndrng_Prvdr_Type",
            "Tot_Sbmtd_Chrg",
            "Med_Tot_Srvcs",
            "Avg_Sbmtd_Chrg_min",
            "Avg_Mdcr_Pymt_Amt_std",
            "Tot_HCPCS_Cds",
            "Avg_Sbmtd_Chrg_max",
            "Tot_Mdcr_Stdzd_Amt",
            "Tot_Benes_max",
            "Avg_Mdcr_Pymt_Amt_median",
            "Tot_Benes_std",
        ],
        'cat_features': [
        ]
    },
    'part-b-aggregated-new-top-25-sfs-5': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Tot_Bene_Day_Srvcs_sum",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Benes_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Avg_Mdcr_Pymt_Amt_min",
            "Rndrng_Prvdr_Type",
            "Tot_Sbmtd_Chrg",
            "Med_Tot_Srvcs",
            "Avg_Sbmtd_Chrg_min",
            "Avg_Mdcr_Pymt_Amt_std",
            "Tot_HCPCS_Cds",
            "Avg_Sbmtd_Chrg_max",
            "Tot_Mdcr_Stdzd_Amt",
            "Tot_Benes_max",
            "Avg_Mdcr_Pymt_Amt_median",
            "Tot_Benes_std",
            "Tot_Srvcs_max",
            "Avg_Mdcr_Pymt_Amt_mean",
            "Rndrng_Prvdr_Gndr",
            "Med_Tot_HCPCS_Cds",
            "Tot_Bene_Day_Srvcs_max",

        ],
        'cat_features': [
        ]
    },
    'part-b-aggregated-new-top-30-sfs-5': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Tot_Bene_Day_Srvcs_sum",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Benes_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Avg_Mdcr_Pymt_Amt_min",
            "Rndrng_Prvdr_Type",
            "Tot_Sbmtd_Chrg",
            "Med_Tot_Srvcs",
            "Avg_Sbmtd_Chrg_min",
            "Avg_Mdcr_Pymt_Amt_std",
            "Tot_HCPCS_Cds",
            "Avg_Sbmtd_Chrg_max",
            "Tot_Mdcr_Stdzd_Amt",
            "Tot_Benes_max",
            "Avg_Mdcr_Pymt_Amt_median",
            "Tot_Benes_std",
            "Tot_Srvcs_max",
            "Avg_Mdcr_Pymt_Amt_mean",
            "Rndrng_Prvdr_Gndr",
            "Med_Tot_HCPCS_Cds",
            "Tot_Bene_Day_Srvcs_max",
            "Tot_Mdcr_Alowd_Amt",
            "Bene_CC_IHD_Pct",
            "Tot_Benes_mean",
            "Avg_Sbmtd_Chrg_std",
            "Tot_Bene_Day_Srvcs_min",
        ],
        'cat_features': [
        ]
    },
        'part-b-aggregated-new-pre_encoded': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
            'Tot_Srvcs_mean',
            'Tot_Srvcs_median',
            'Tot_Srvcs_sum',
            'Tot_Srvcs_std',
            'Tot_Srvcs_min',
            'Tot_Srvcs_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benes_std',
            'Tot_Benes_min',
            'Tot_Benes_max',
            'Tot_Bene_Day_Srvcs_mean',
            'Tot_Bene_Day_Srvcs_median',
            'Tot_Bene_Day_Srvcs_sum',
            'Tot_Bene_Day_Srvcs_std',
            'Tot_Bene_Day_Srvcs_min',
            'Tot_Bene_Day_Srvcs_max',
            'Avg_Sbmtd_Chrg_mean',
            'Avg_Sbmtd_Chrg_median',
            'Avg_Sbmtd_Chrg_sum',
            'Avg_Sbmtd_Chrg_std',
            'Avg_Sbmtd_Chrg_min',
            'Avg_Sbmtd_Chrg_max',
            'Avg_Mdcr_Pymt_Amt_mean',
            'Avg_Mdcr_Pymt_Amt_median',
            'Avg_Mdcr_Pymt_Amt_sum',
            'Avg_Mdcr_Pymt_Amt_std',
            'Avg_Mdcr_Pymt_Amt_min',
            'Avg_Mdcr_Pymt_Amt_max',
            'Tot_HCPCS_Cds',
            'Tot_Benes',
            'Tot_Srvcs',
            'Tot_Sbmtd_Chrg',
            'Tot_Mdcr_Alowd_Amt',
            'Tot_Mdcr_Pymt_Amt',
            'Tot_Mdcr_Stdzd_Amt',
            'Drug_Tot_HCPCS_Cds',
            'Drug_Tot_Benes',
            'Drug_Tot_Srvcs',
            'Drug_Sbmtd_Chrg',
            'Drug_Mdcr_Alowd_Amt',
            'Drug_Mdcr_Pymt_Amt',
            'Drug_Mdcr_Stdzd_Amt',
            'Med_Tot_HCPCS_Cds',
            'Med_Tot_Benes',
            'Med_Tot_Srvcs',
            'Med_Sbmtd_Chrg',
            'Med_Mdcr_Alowd_Amt',
            'Med_Mdcr_Pymt_Amt',
            'Med_Mdcr_Stdzd_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre'
        ],
        'cat_features': [
        ]
    },
        'part-b-aggregated-new-top-10-sfs-6': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Tot_Bene_Day_Srvcs_sum",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Benes_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Avg_Mdcr_Pymt_Amt_min",
            "Rndrng_Prvdr_Type",
        ],
        'cat_features': [
        ]
    },
     'part-b-aggregated-new-top-15-sfs-6': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Bene_Day_Srvcs_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Benes_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Rndrng_Prvdr_Type",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_min",
	    "Tot_Sbmtd_Chrg",
            "Avg_Mdcr_Pymt_Amt_std",
            "Avg_Sbmtd_Chrg_max",
            "Tot_HCPCS_Cds",
            "Avg_Sbmtd_Chrg_min",

        ],
        'cat_features': [
        ]
    },
    'part-b-aggregated-new-top-20-sfs-6': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Bene_Day_Srvcs_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Benes_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Rndrng_Prvdr_Type",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_min",
	    "Tot_Sbmtd_Chrg",
            "Avg_Mdcr_Pymt_Amt_std",
            "Avg_Sbmtd_Chrg_max",
            "Tot_HCPCS_Cds",
            "Avg_Sbmtd_Chrg_min",	    
            "Med_Tot_Srvcs",
            "Rndrng_Prvdr_Gndr",
            "Avg_Mdcr_Pymt_Amt_mean",
            "Tot_Benes_max",
            "Tot_Srvcs_max",
        ],
        'cat_features': [
        ]
    },
    'part-b-aggregated-new-top-25-sfs-6': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Bene_Day_Srvcs_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Benes_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Rndrng_Prvdr_Type",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_min",
	    "Tot_Sbmtd_Chrg",
            "Avg_Mdcr_Pymt_Amt_std",
            "Avg_Sbmtd_Chrg_max",
            "Tot_HCPCS_Cds",
            "Avg_Sbmtd_Chrg_min",	    
            "Med_Tot_Srvcs",
            "Rndrng_Prvdr_Gndr",
            "Avg_Mdcr_Pymt_Amt_mean",
            "Tot_Benes_max",
            "Tot_Srvcs_max",	    
            "Tot_Benes_std",
            "Avg_Mdcr_Pymt_Amt_median",
            "Med_Tot_HCPCS_Cds",
            "Bene_Avg_Age",
            "Bene_Avg_Risk_Scre",
        ],
        'cat_features': [
        ]
    },
    'part-b-aggregated-new-top-30-sfs-6': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
             "Tot_Srvcs",
            "Avg_Mdcr_Pymt_Amt_sum",
            "Avg_Mdcr_Pymt_Amt_max",
            "Tot_Bene_Day_Srvcs_sum",
            "Avg_Sbmtd_Chrg_sum",
            "Tot_Benes_sum",
            "Tot_Mdcr_Pymt_Amt",
            "Rndrng_Prvdr_Type",
            "Tot_Srvcs_sum",
            "Avg_Mdcr_Pymt_Amt_min",
	    "Tot_Sbmtd_Chrg",
            "Avg_Mdcr_Pymt_Amt_std",
            "Avg_Sbmtd_Chrg_max",
            "Tot_HCPCS_Cds",
            "Avg_Sbmtd_Chrg_min",	    
            "Med_Tot_Srvcs",
            "Rndrng_Prvdr_Gndr",
            "Avg_Mdcr_Pymt_Amt_mean",
            "Tot_Benes_max",
            "Tot_Srvcs_max",	    
            "Tot_Benes_std",
            "Avg_Mdcr_Pymt_Amt_median",
            "Med_Tot_HCPCS_Cds",
            "Bene_Avg_Age",
            "Bene_Avg_Risk_Scre",	    
            "Avg_Sbmtd_Chrg_std",
            "Tot_Mdcr_Stdzd_Amt",
            "Tot_Bene_Day_Srvcs_max",
            "Place_Of_Srvc",
            "Med_Sbmtd_Chrg",
        ],
        'cat_features': [
        ]
    },
            'part-b-aggregated-new-top-10-tbfs': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Avg_Mdcr_Pymt_Amt_sum",
            "Rndrng_Prvdr_Type",
            "Avg_Sbmtd_Chrg_sum",
            "Bene_Avg_Age",
            "Bene_CC_Dprssn_Pct",
            "Bene_CC_Sz_Pct",
            "Bene_CC_Dbts_Pct",
            "Bene_CC_RAOA_Pct",
            "Bene_CC_Asthma_Pct",
            "Bene_CC_Alzhmr_Pct",
        ],
        'cat_features': [
        ]
    },
     'part-b-aggregated-new-top-15-tbfs': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Avg_Mdcr_Pymt_Amt_sum",
            "Rndrng_Prvdr_Type",
            "Avg_Sbmtd_Chrg_sum",
            "Bene_Avg_Age",
            "Bene_CC_Dprssn_Pct",
            "Bene_CC_Sz_Pct",
            "Bene_CC_Dbts_Pct",
            "Bene_CC_RAOA_Pct",
            "Bene_CC_Asthma_Pct",
            "Bene_CC_Alzhmr_Pct",	    
            "Bene_CC_Hyplpdma_Pct",
            "Bene_CC_CKD_Pct",
            "Bene_CC_COPD_Pct",
            "Tot_Srvcs_sum",
            "Bene_CC_AF_Pct",
        ],
        'cat_features': [
        ]
    },
    'part-b-aggregated-new-top-20-tbfs': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Avg_Mdcr_Pymt_Amt_sum",
            "Rndrng_Prvdr_Type",
            "Avg_Sbmtd_Chrg_sum",
            "Bene_Avg_Age",
            "Bene_CC_Dprssn_Pct",
            "Bene_CC_Sz_Pct",
            "Bene_CC_Dbts_Pct",
            "Bene_CC_RAOA_Pct",
            "Bene_CC_Asthma_Pct",
            "Bene_CC_Alzhmr_Pct",	    
            "Bene_CC_Hyplpdma_Pct",
            "Bene_CC_CKD_Pct",
            "Bene_CC_COPD_Pct",
            "Tot_Srvcs_sum",
            "Bene_CC_AF_Pct",	    
            "Bene_CC_Cncr_Pct",
            "Bene_CC_Opo_Pct",
            "Med_Mdcr_Alowd_Amt",
            "Med_Mdcr_Pymt_Amt",
            "Place_Of_Srvc",
        ],
        'cat_features': [
        ]
    },
    'part-b-aggregated-new-top-25-tbfs': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Avg_Mdcr_Pymt_Amt_sum",
            "Rndrng_Prvdr_Type",
            "Avg_Sbmtd_Chrg_sum",
            "Bene_Avg_Age",
            "Bene_CC_Dprssn_Pct",
            "Bene_CC_Sz_Pct",
            "Bene_CC_Dbts_Pct",
            "Bene_CC_RAOA_Pct",
            "Bene_CC_Asthma_Pct",
            "Bene_CC_Alzhmr_Pct",	    
            "Bene_CC_Hyplpdma_Pct",
            "Bene_CC_CKD_Pct",
            "Bene_CC_COPD_Pct",
            "Tot_Srvcs_sum",
            "Bene_CC_AF_Pct",	    
            "Bene_CC_Cncr_Pct",
            "Bene_CC_Opo_Pct",
            "Med_Mdcr_Alowd_Amt",
            "Med_Mdcr_Pymt_Amt",
            "Place_Of_Srvc",	    
            "Drug_Tot_HCPCS_Cds",
            "Tot_Mdcr_Alowd_Amt",
            "Tot_Bene_Day_Srvcs_sum",
            "Rndrng_Prvdr_Gndr",
            "Med_Mdcr_Stdzd_Amt",

        ],
        'cat_features': [
        ]
    },
    'part-b-aggregated-new-top-30-tbfs': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partb-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Avg_Mdcr_Pymt_Amt_sum",
            "Rndrng_Prvdr_Type",
            "Avg_Sbmtd_Chrg_sum",
            "Bene_Avg_Age",
            "Bene_CC_Dprssn_Pct",
            "Bene_CC_Sz_Pct",
            "Bene_CC_Dbts_Pct",
            "Bene_CC_RAOA_Pct",
            "Bene_CC_Asthma_Pct",
            "Bene_CC_Alzhmr_Pct",	    
            "Bene_CC_Hyplpdma_Pct",
            "Bene_CC_CKD_Pct",
            "Bene_CC_COPD_Pct",
            "Tot_Srvcs_sum",
            "Bene_CC_AF_Pct",	    
            "Bene_CC_Cncr_Pct",
            "Bene_CC_Opo_Pct",
            "Med_Mdcr_Alowd_Amt",
            "Med_Mdcr_Pymt_Amt",
            "Place_Of_Srvc",	    
            "Drug_Tot_HCPCS_Cds",
            "Tot_Mdcr_Alowd_Amt",
            "Tot_Bene_Day_Srvcs_sum",
            "Rndrng_Prvdr_Gndr",
            "Med_Mdcr_Stdzd_Amt",	    
            "Tot_Srvcs_max",
            "Bene_Avg_Risk_Scre",
            "Bene_CC_Strok_Pct",
            "Bene_CC_IHD_Pct",
            "Avg_Sbmtd_Chrg_mean",
        ],
        'cat_features': [
        ]
    },
    'part-b-cleaned': {
        'file_name': f'{base_dir}/cleaned/partb-aggregated-new-features-cleaned.csv.gz',
        'sample_file': f'{sample_dir}/partb-aggregated-new-features-cleaned.csv',
        'target': 'exclusion',
        'features': [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
            'Tot_Srvcs_mean',
            'Tot_Srvcs_median',
            'Tot_Srvcs_sum',
            'Tot_Srvcs_std',
            'Tot_Srvcs_min',
            'Tot_Srvcs_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benes_std',
            'Tot_Benes_min',
            'Tot_Benes_max',
            'Tot_Bene_Day_Srvcs_mean',
            'Tot_Bene_Day_Srvcs_median',
            'Tot_Bene_Day_Srvcs_sum',
            'Tot_Bene_Day_Srvcs_std',
            'Tot_Bene_Day_Srvcs_min',
            'Tot_Bene_Day_Srvcs_max',
            'Avg_Sbmtd_Chrg_mean',
            'Avg_Sbmtd_Chrg_median',
            'Avg_Sbmtd_Chrg_sum',
            'Avg_Sbmtd_Chrg_std',
            'Avg_Sbmtd_Chrg_min',
            'Avg_Sbmtd_Chrg_max',
            'Avg_Mdcr_Pymt_Amt_mean',
            'Avg_Mdcr_Pymt_Amt_median',
            'Avg_Mdcr_Pymt_Amt_sum',
            'Avg_Mdcr_Pymt_Amtt_std',
            'Avg_Mdcr_Pymt_Amt_min',
            'Avg_Mdcr_Pymt_Amt_max',
            'Tot_HCPCS_Cds',
            'Tot_Benes',
            'Tot_Srvcs',
            'Tot_Sbmtd_Chrg',
            'Tot_Mdcr_Alowd_Amt',
            'Tot_Mdcr_Pymt_Amt',
            'Tot_Mdcr_Stdzd_Amt',
            'Drug_Tot_HCPCS_Cds',
            'Drug_Tot_Benes',
            'Drug_Tot_Srvcs',
            'Drug_Sbmtd_Chrg',
            'Drug_Mdcr_Alowd_Amt',
            'Drug_Mdcr_Pymt_Amt',
            'Drug_Mdcr_Stdzd_Amt',
            'Med_Tot_HCPCS_Cds',
            'Med_Tot_Benes',
            'Med_Tot_Srvcs',
            'Med_Sbmtd_Chrg',
            'Med_Mdcr_Alowd_Amt',
            'Med_Mdcr_Pymt_Amt',
            'Med_Mdcr_Stdzd_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre'],
        'cat_features': [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
        ]
    },
    'part-b-noisy': {
        'file_name': f'{base_dir}/cleaned/partb-aggregated-new-features-noise.csv.gz',
        'sample_file': f'{sample_dir}/partb-aggregated-new-features-noise.csv',
        'target': 'exclusion',
        'features': [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
            'Tot_Srvcs_mean',
            'Tot_Srvcs_median',
            'Tot_Srvcs_sum',
            'Tot_Srvcs_std',
            'Tot_Srvcs_min',
            'Tot_Srvcs_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benes_std',
            'Tot_Benes_min',
            'Tot_Benes_max',
            'Tot_Bene_Day_Srvcs_mean',
            'Tot_Bene_Day_Srvcs_median',
            'Tot_Bene_Day_Srvcs_sum',
            'Tot_Bene_Day_Srvcs_std',
            'Tot_Bene_Day_Srvcs_min',
            'Tot_Bene_Day_Srvcs_max',
            'Avg_Sbmtd_Chrg_mean',
            'Avg_Sbmtd_Chrg_median',
            'Avg_Sbmtd_Chrg_sum',
            'Avg_Sbmtd_Chrg_std',
            'Avg_Sbmtd_Chrg_min',
            'Avg_Sbmtd_Chrg_max',
            'Avg_Mdcr_Pymt_Amt_mean',
            'Avg_Mdcr_Pymt_Amt_median',
            'Avg_Mdcr_Pymt_Amt_sum',
            'Avg_Mdcr_Pymt_Amtt_std',
            'Avg_Mdcr_Pymt_Amt_min',
            'Avg_Mdcr_Pymt_Amt_max',
            'Tot_HCPCS_Cds',
            'Tot_Benes',
            'Tot_Srvcs',
            'Tot_Sbmtd_Chrg',
            'Tot_Mdcr_Alowd_Amt',
            'Tot_Mdcr_Pymt_Amt',
            'Tot_Mdcr_Stdzd_Amt',
            'Drug_Tot_HCPCS_Cds',
            'Drug_Tot_Benes',
            'Drug_Tot_Srvcs',
            'Drug_Sbmtd_Chrg',
            'Drug_Mdcr_Alowd_Amt',
            'Drug_Mdcr_Pymt_Amt',
            'Drug_Mdcr_Stdzd_Amt',
            'Med_Tot_HCPCS_Cds',
            'Med_Tot_Benes',
            'Med_Tot_Srvcs',
            'Med_Sbmtd_Chrg',
            'Med_Mdcr_Alowd_Amt',
            'Med_Mdcr_Pymt_Amt',
            'Med_Mdcr_Stdzd_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre'],
        'cat_features': [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
        ]
    },
    'part-b-noisy-clean-inverted': {
        'file_name': f'/mnt/beegfs/home/jhancoc4/medicare-data/2019/partb-aggregated-new-features-noise_clean_inverted.csv.gz',
        'sample_file': f'{sample_dir}/partb-aggregated-new-features-noise_clean_inverted.csv',
        'target': 'exclusion',
        'features': [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
            'Tot_Srvcs_mean',
            'Tot_Srvcs_median',
            'Tot_Srvcs_sum',
            'Tot_Srvcs_std',
            'Tot_Srvcs_min',
            'Tot_Srvcs_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benes_std',
            'Tot_Benes_min',
            'Tot_Benes_max',
            'Tot_Bene_Day_Srvcs_mean',
            'Tot_Bene_Day_Srvcs_median',
            'Tot_Bene_Day_Srvcs_sum',
            'Tot_Bene_Day_Srvcs_std',
            'Tot_Bene_Day_Srvcs_min',
            'Tot_Bene_Day_Srvcs_max',
            'Avg_Sbmtd_Chrg_mean',
            'Avg_Sbmtd_Chrg_median',
            'Avg_Sbmtd_Chrg_sum',
            'Avg_Sbmtd_Chrg_std',
            'Avg_Sbmtd_Chrg_min',
            'Avg_Sbmtd_Chrg_max',
            'Avg_Mdcr_Pymt_Amt_mean',
            'Avg_Mdcr_Pymt_Amt_median',
            'Avg_Mdcr_Pymt_Amt_sum',
            'Avg_Mdcr_Pymt_Amtt_std',
            'Avg_Mdcr_Pymt_Amt_min',
            'Avg_Mdcr_Pymt_Amt_max',
            'Tot_HCPCS_Cds',
            'Tot_Benes',
            'Tot_Srvcs',
            'Tot_Sbmtd_Chrg',
            'Tot_Mdcr_Alowd_Amt',
            'Tot_Mdcr_Pymt_Amt',
            'Tot_Mdcr_Stdzd_Amt',
            'Drug_Tot_HCPCS_Cds',
            'Drug_Tot_Benes',
            'Drug_Tot_Srvcs',
            'Drug_Sbmtd_Chrg',
            'Drug_Mdcr_Alowd_Amt',
            'Drug_Mdcr_Pymt_Amt',
            'Drug_Mdcr_Stdzd_Amt',
            'Med_Tot_HCPCS_Cds',
            'Med_Tot_Benes',
            'Med_Tot_Srvcs',
            'Med_Sbmtd_Chrg',
            'Med_Mdcr_Alowd_Amt',
            'Med_Mdcr_Pymt_Amt',
            'Med_Mdcr_Stdzd_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre'],
        'cat_features': [
            'Rndrng_Prvdr_Type',
            'Place_Of_Srvc',
            'Rndrng_Prvdr_Gndr',
        ]
    },
    'part-d-aggregated': {
        'file_name': f'{base_dir}/medicare-partd-aggregated-2013-2019.csv.gz',
        'sample_file': f'{sample_dir}/medicare-partd-aggregated-2013-2019.csv',
        'target': 'exclusion',
        'features': [
            'Prscrbr_Type',
            'Tot_Clms_mean',
            'Tot_Clms_median',
            'Tot_Clms_sum',
            'Tot_Clms_std',
            'Tot_Clms_min',
            'Tot_Clms_max',
            'Tot_30day_Fills_mean',
            'Tot_30day_Fills_median',
            'Tot_30day_Fills_sum',
            'Tot_30day_Fills_std',
            'Tot_30day_Fills_min',
            'Tot_30day_Fills_max',
            'Tot_Day_Suply_mean',
            'Tot_Day_Suply_median',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply_std',
            'Tot_Day_Suply_min',
            'Tot_Day_Suply_max',
            'Tot_Drug_Cst_mean',
            'Tot_Drug_Cst_median',
            'Tot_Drug_Cst_sum',
            'Tot_Drug_Cst_std',
            'Tot_Drug_Cst_min',
            'Tot_Drug_Cst_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benest_std',
            'Tot_Benes_min',
            'Tot_Benest_max',
        ],
        'cat_features': ['Prscrbr_Type']
    },
    'part-d': {
        'file_name': f'{base_dir}/medicare-partd-2013-2019.csv.gz',
        'sample_file': f'{sample_dir}/medicare-partd-2013-2019.csv',
        'target': 'exclusion',
        'features':  [
            'Prscrbr_Type',
            'Prscrbr_Type_Src',
            'Brnd_Name',
            'Gnrc_Name',
            'Tot_Clms',
            'Tot_30day_Fills',
            'Tot_Day_Suply',
            'Tot_Drug_Cst',
            'Tot_Benes',
            # missing values 'GE65_Sprsn_Flag',
            # missing values 'GE65_Tot_Clms',
            # missing values 'GE65_Tot_30day_Fills',
            # missing values 'GE65_Tot_Drug_Cst',
            # missing values 'GE65_Tot_Day_Suply',
            # missing values 'GE65_Bene_Sprsn_Flag',
            # missing values 'GE65_Tot_Benes',
        ],
        'cat_features': [
            'Prscrbr_Type',
            'Prscrbr_Type_Src',
            'Brnd_Name',
            'Gnrc_Name',
            # missing values 'GE65_Sprsn_Flag',
            # missing values 'GE65_Bene_Sprsn_Flag'
        ]
    },
    'part-d-aggregated-new': {
        'file_name': f'{base_dir}/medicare-partd-aggregated-new-features-2013-2019.csv.gz',
        'sample_file': f'{sample_dir}/medicare-partd-aggregated-new-features-2013-2019.csv',
        'target': 'exclusion',
        'features':  [
            'Prscrbr_Type',
            'Tot_Clms_mean',
            'Tot_Clms_median',
            'Tot_Clms_sum',
            'Tot_Clms_std',
            'Tot_Clms_min',
            'Tot_Clms_max',
            'Tot_30day_Fills_mean',
            'Tot_30day_Fills_median',
            'Tot_30day_Fills_sum',
            'Tot_30day_Fills_std',
            'Tot_30day_Fills_min',
            'Tot_30day_Fills_max',
            'Tot_Day_Suply_mean',
            'Tot_Day_Suply_median',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply_std',
            'Tot_Day_Suply_min',
            'Tot_Day_Suply_max',
            'Tot_Drug_Cst_mean',
            'Tot_Drug_Cst_median',
            'Tot_Drug_Cst_sum',
            'Tot_Drug_Cst_std',
            'Tot_Drug_Cst_min',
            'Tot_Drug_Cst_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benest_std',
            'Tot_Benes_min',
            'Tot_Benest_max',
            'Tot_Clms',
            'Tot_30day_Fills',
            'Tot_Drug_Cst',
            'Tot_Day_Suply',
            'Tot_Benes',
            'GE65_Tot_Clms',
            'GE65_Tot_30day_Fills',
            'GE65_Tot_Drug_Cst',
            'GE65_Tot_Day_Suply',
            'GE65_Tot_Benes',
            'Brnd_Tot_Clms',
            'Brnd_Tot_Drug_Cst',
            'Gnrc_Tot_Clms',
            'Gnrc_Tot_Drug_Cst',
            'Othr_Tot_Clms',
            'Othr_Tot_Drug_Cst',
            'MAPD_Tot_Clms',
            'MAPD_Tot_Drug_Cst',
            'PDP_Tot_Clms',
            'PDP_Tot_Drug_Cst',
            'LIS_Tot_Clms',
            'LIS_Drug_Cst',
            'NonLIS_Tot_Clms',
            'NonLIS_Drug_Cst',
            'Opioid_Tot_Clms',
            'Opioid_Tot_Drug_Cst',
            'Opioid_Tot_Suply',
            'Opioid_Tot_Benes',
            'Opioid_Prscrbr_Rate',
            'Opioid_LA_Tot_Clms',
            'Opioid_LA_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Opioid_LA_Prscrbr_Rate',
            'Antbtc_Tot_Clms',
            'Antbtc_Tot_Drug_Cst',
            'Antbtc_Tot_Benes',
            'Antpsyct_GE65_Tot_Clms',
            'Antpsyct_GE65_Tot_Drug_Cst',
            'Antpsyct_GE65_Bene_Suprsn_Flag',
            'Antpsyct_GE65_Tot_Benes',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_Avg_Risk_Scre',
        ],
        'cat_features': [
            'Prscrbr_Type',
        ]
    },
        'part-d-aggregated-new-top-10': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Drug_Cst_sum',
            'Bene_Avg_Age',            
            'Opioid_Tot_Drug_Cst',            
            'Opioid_LA_Tot_Drug_Cst',            
            'Opioid_Tot_Clms',
            'Opioid_Tot_Suply',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-top-15': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Drug_Cst_sum',
            'Bene_Avg_Age',            
            'Opioid_Tot_Drug_Cst',            
            'Opioid_LA_Tot_Drug_Cst',            
            'Opioid_Tot_Clms',
            'Opioid_Tot_Suply',
            'Opioid_LA_Tot_Clms',
            'Opioid_Tot_Benes',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',            
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-top-20': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Drug_Cst_sum',
            'Bene_Avg_Age',            
            'Opioid_Tot_Drug_Cst',            
            'Opioid_LA_Tot_Drug_Cst',            
            'Opioid_Tot_Clms',
            'Opioid_Tot_Suply',
            'Opioid_LA_Tot_Clms',
            'Opioid_Tot_Benes',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Bene_Age_LT_65_Cnt',
            'Prscrbr_Type',
            'LIS_Drug_Cst',
            'Opioid_LA_Prscrbr_Rate',
            'PDP_Tot_Clms',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-top-25': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Drug_Cst_sum',
            'Bene_Avg_Age',            
            'Opioid_Tot_Drug_Cst',            
            'Opioid_LA_Tot_Drug_Cst',            
            'Opioid_Tot_Clms',
            'Opioid_Tot_Suply',
            'Opioid_LA_Tot_Clms',
            'Opioid_Tot_Benes',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Bene_Age_LT_65_Cnt',
            'Prscrbr_Type',
            'LIS_Drug_Cst',
            'Opioid_LA_Prscrbr_Rate',
            'PDP_Tot_Clms',
            'PDP_Tot_Drug_Cst',
            'Antpsyct_GE65_Tot_Benes',
            'Tot_Clms',
            'LIS_Tot_Clms',
            'Tot_Day_Suply',
            
        ],
        'cat_features': [
        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-top-30': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Drug_Cst_sum',
            'Bene_Avg_Age',            
            'Opioid_Tot_Drug_Cst',            
            'Opioid_LA_Tot_Drug_Cst',            
            'Opioid_Tot_Clms',
            'Opioid_Tot_Suply',
            'Opioid_LA_Tot_Clms',
            'Opioid_Tot_Benes',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Bene_Age_LT_65_Cnt',
            'Prscrbr_Type',
            'LIS_Drug_Cst',
            'Opioid_LA_Prscrbr_Rate',
            'PDP_Tot_Clms',
            'PDP_Tot_Drug_Cst',
            'Antpsyct_GE65_Tot_Benes',
            'Tot_Clms',
            'LIS_Tot_Clms',
            'Tot_Day_Suply',
            'Antbtc_Tot_Benes',
            'Gnrc_Tot_Clms',
            'Bene_Dual_Cnt',
            'Tot_Day_Suply_max',
            'Tot_Drug_Cst',            
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-cufs-top-10': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Opioid_LA_Tot_Suply',
            'Tot_Day_Suply_sum',
            'Opioid_LA_Tot_Benes',
            'Opioid_LA_Prscrbr_Rate',
            'Tot_30day_Fills_sum',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Drug_Cst',
            'Opioid_Tot_Benes',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-cufs-top-15': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Opioid_LA_Tot_Suply',
            'Tot_Day_Suply_sum',
            'Opioid_LA_Tot_Benes',
            'Opioid_LA_Prscrbr_Rate',
            'Tot_30day_Fills_sum',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Drug_Cst',
            'Opioid_Tot_Benes',
            'Opioid_Tot_Clms',
            'Tot_Clms_sum',
            'LIS_Drug_Cst',
            'Opioid_Tot_Suply',
            'Bene_Age_LT_65_Cnt',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-cufs-top-20': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Opioid_LA_Tot_Suply',
            'Tot_Day_Suply_sum',
            'Opioid_LA_Tot_Benes',
            'Opioid_LA_Prscrbr_Rate',
            'Tot_30day_Fills_sum',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Drug_Cst',
            'Opioid_Tot_Benes',
            'Opioid_Tot_Clms',
            'Tot_Clms_sum',
            'LIS_Drug_Cst',
            'Opioid_Tot_Suply',
            'Bene_Age_LT_65_Cnt',
            'Tot_Benes_sum',
            'Opioid_Tot_Drug_Cst',
            'Antpsyct_GE65_Tot_Benes',
            'Bene_Avg_Age',
            'Bene_Dual_Cnt',
        ],
        'cat_features': [
        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-cufs-top-25': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Opioid_LA_Tot_Suply',
            'Tot_Day_Suply_sum',
            'Opioid_LA_Tot_Benes',
            'Opioid_LA_Prscrbr_Rate',
            'Tot_30day_Fills_sum',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Drug_Cst',
            'Opioid_Tot_Benes',
            'Opioid_Tot_Clms',
            'Tot_Clms_sum',
            'LIS_Drug_Cst',
            'Opioid_Tot_Suply',
            'Bene_Age_LT_65_Cnt',
            'Tot_Benes_sum',
            'Opioid_Tot_Drug_Cst',
            'Antpsyct_GE65_Tot_Benes',
            'Bene_Avg_Age',
            'Bene_Dual_Cnt',
            'PDP_Tot_Drug_Cst',
            'Tot_Day_Suply',
            'Antpsyct_GE65_Tot_Drug_Cst',
            'Antpsyct_GE65_Tot_Clms',
            'LIS_Tot_Clms',
   
        ],
        'cat_features': [
        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-cufs-top-30': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Opioid_LA_Tot_Suply',
            'Tot_Day_Suply_sum',
            'Opioid_LA_Tot_Benes',
            'Opioid_LA_Prscrbr_Rate',
            'Tot_30day_Fills_sum',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Drug_Cst',
            'Opioid_Tot_Benes',
            'Opioid_Tot_Clms',
            'Tot_Clms_sum',
            'LIS_Drug_Cst',
            'Opioid_Tot_Suply',
            'Bene_Age_LT_65_Cnt',
            'Tot_Benes_sum',
            'Opioid_Tot_Drug_Cst',
            'Antpsyct_GE65_Tot_Benes',
            'Bene_Avg_Age',
            'Bene_Dual_Cnt',
            'PDP_Tot_Drug_Cst',
            'Tot_Day_Suply',
            'Antpsyct_GE65_Tot_Drug_Cst',
            'Antpsyct_GE65_Tot_Clms',
            'LIS_Tot_Clms',
            'Opioid_Prscrbr_Rate',
            'Brnd_Tot_Clms',
            'Tot_30day_Fills',
            'PDP_Tot_Clms',
            'Tot_Drug_Cst',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-sfs-top-7a': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Clms',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply',
            'Tot_30day_Fills',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-sfs-top-7b': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Clms',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply',
            'Gnrc_Tot_Clms'
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-sfs-top-8': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Clms',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply',
            'Tot_30day_Fills',
            'Gnrc_Tot_Clms',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-sfs-top-9': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Clms',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply',
            'Tot_30day_Fills',
            'Gnrc_Tot_Clms',
            'GE65_Tot_Clms'
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-sfs-top-10': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Clms',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply',
            'Tot_30day_Fills',
            'Gnrc_Tot_Clms',
            'GE65_Tot_Clms',
            'Tot_Drug_Cst_std',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-sfs-top-15': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Clms',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply',
            'Tot_30day_Fills',
            'Gnrc_Tot_Clms',
            'GE65_Tot_Clms',
            'Tot_Drug_Cst_std',
            'Tot_Clms_max',
            'Tot_30day_Fills_max',
            'Tot_Drug_Cst',
            'Tot_Clms_mean',
            'Tot_Drug_Cst_sum',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-sfs-top-20': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Clms',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply',
            'Tot_30day_Fills',
            'Gnrc_Tot_Clms',
            'GE65_Tot_Clms',
            'Tot_Drug_Cst_std',
            'Tot_Clms_max',
            'Tot_30day_Fills_max',
            'Tot_Drug_Cst',
            'Tot_Clms_mean',
            'Tot_Drug_Cst_sum',
            'Tot_Clms_std',
            'GE65_Tot_30day_Fills',
            'Gnrc_Tot_Drug_Cst',
            'Prscrbr_Type',
            'Tot_Day_Suply_std',            
        ],
        'cat_features': [
        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-sfs-top-25': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Clms',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply',
            'Tot_30day_Fills',
            'Gnrc_Tot_Clms',
            'GE65_Tot_Clms',
            'Tot_Drug_Cst_std',
            'Tot_Clms_max',
            'Tot_30day_Fills_max',
            'Tot_Drug_Cst',
            'Tot_Clms_mean',
            'Tot_Drug_Cst_sum',
            'Tot_Clms_std',
            'GE65_Tot_30day_Fills',
            'Gnrc_Tot_Drug_Cst',
            'Prscrbr_Type',
            'Tot_Day_Suply_std',
            'Tot_Benest_max',
            'Tot_Benes',
            'Tot_Drug_Cst_max',
            'Tot_30day_Fills_std',
            'Tot_Drug_Cst_min',
        ],
        'cat_features': [
        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-sfs-new-top-30': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Clms',
            'Tot_30day_Fills_sum',
            'Tot_Benes_sum',
            'Tot_Clms_sum',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply',
            'Tot_30day_Fills',
            'Gnrc_Tot_Clms',
            'GE65_Tot_Clms',
            'Tot_Drug_Cst_std',
            'Tot_Clms_max',
            'Tot_30day_Fills_max',
            'Tot_Drug_Cst',
            'Tot_Clms_mean',
            'Tot_Drug_Cst_sum',
            'Tot_Clms_std',
            'GE65_Tot_30day_Fills',
            'Gnrc_Tot_Drug_Cst',
            'Prscrbr_Type',
            'Tot_Day_Suply_std',
            'Tot_Benest_max',
            'Tot_Benes',
            'Tot_Drug_Cst_max',
            'Tot_30day_Fills_std',
            'Tot_Drug_Cst_min',
            'Bene_Male_Cnt',
            'GE65_Tot_Day_Suply',
            'Tot_Day_Suply_max',
            'Bene_Avg_Age',
            'Tot_Drug_Cst_median',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-combined-top-10': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Clms_sum',
            'Tot_Benes_sum',
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Opioid_Tot_Benes'
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-combined-top-15': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Clms_sum',
            'Tot_Benes_sum',
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Opioid_Tot_Benes',
            'Opioid_Tot_Clms',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Opioid_Tot_Suply',
            'PDP_Tot_Clms',
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-combined-top-20': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Clms_sum',
            'Tot_Benes_sum',
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Opioid_Tot_Benes',
            'Opioid_Tot_Clms',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Opioid_Tot_Suply',
            'PDP_Tot_Clms',
            'Opioid_LA_Tot_Drug_Cst',
            'Prscrbr_Type',
            'Tot_Day_Suply',
            'Opioid_LA_Prscrbr_Rate',
            'Tot_Clms',
        ],
        'cat_features': [
        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-new-combined-top-25': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Clms_sum',
            'Tot_Benes_sum',
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Opioid_Tot_Benes',
            'Opioid_Tot_Clms',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Opioid_Tot_Suply',
            'PDP_Tot_Clms',
            'Opioid_LA_Tot_Drug_Cst',
            'Prscrbr_Type',
            'Tot_Day_Suply',
            'Opioid_LA_Prscrbr_Rate',
            'Tot_Clms',
            'LIS_Drug_Cst',
            'Opioid_Tot_Drug_Cst',
            'LIS_Tot_Clms',
            'PDP_Tot_Drug_Cst',
            'Gnrc_Tot_Clms',
        ],
        'cat_features': [
        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-aggregated-combined-new-top-30': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Tot_Day_Suply_sum',
            'Tot_30day_Fills_sum',
            'Tot_Clms_sum',
            'Tot_Benes_sum',
            'Tot_Drug_Cst_sum',
            'Opioid_LA_Tot_Clms',
            'Gnrc_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Opioid_Tot_Benes',
            'Opioid_Tot_Clms',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Opioid_Tot_Suply',
            'PDP_Tot_Clms',
            'Opioid_LA_Tot_Drug_Cst',
            'Prscrbr_Type',
            'Tot_Day_Suply',
            'Opioid_LA_Prscrbr_Rate',
            'Tot_Clms',
            'LIS_Drug_Cst',
            'Opioid_Tot_Drug_Cst',
            'LIS_Tot_Clms',
            'PDP_Tot_Drug_Cst',
            'Gnrc_Tot_Clms',
            'Tot_30day_Fills',
            'Tot_Clms_max',
            'Tot_Drug_Cst',
            'Bene_Dual_Cnt',
            'Antpsyct_GE65_Tot_Benes' 
        ],
        'cat_features': [

        ],
        'pos_label': 1,
        'neg_label': 0
    },
    'part-d-cleaned': {
        'file_name': f'{base_dir}/cleaned/partd-aggregated-new-features-cleaned.csv.gz',
        'sample_file': f'{sample_dir}/partd-aggregated-new-features-cleaned.csv',
        'target': 'exclusion',
        'features':  [
            'Prscrbr_Type',
            'Tot_Clms_mean',
            'Tot_Clms_median',
            'Tot_Clms_sum',
            'Tot_Clms_std',
            'Tot_Clms_min',
            'Tot_Clms_max',
            'Tot_30day_Fills_mean',
            'Tot_30day_Fills_median',
            'Tot_30day_Fills_sum',
            'Tot_30day_Fills_std',
            'Tot_30day_Fills_min',
            'Tot_30day_Fills_max',
            'Tot_Day_Suply_mean',
            'Tot_Day_Suply_median',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply_std',
            'Tot_Day_Suply_min',
            'Tot_Day_Suply_max',
            'Tot_Drug_Cst_mean',
            'Tot_Drug_Cst_median',
            'Tot_Drug_Cst_sum',
            'Tot_Drug_Cst_std',
            'Tot_Drug_Cst_min',
            'Tot_Drug_Cst_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benest_std',
            'Tot_Benes_min',
            'Tot_Benest_max',
            'Tot_Clms',
            'Tot_30day_Fills',
            'Tot_Drug_Cst',
            'Tot_Day_Suply',
            'Tot_Benes',
            'GE65_Tot_Clms',
            'GE65_Tot_30day_Fills',
            'GE65_Tot_Drug_Cst',
            'GE65_Tot_Day_Suply',
            'GE65_Tot_Benes',
            'Brnd_Tot_Clms',
            'Brnd_Tot_Drug_Cst',
            'Gnrc_Tot_Clms',
            'Gnrc_Tot_Drug_Cst',
            'Othr_Tot_Clms',
            'Othr_Tot_Drug_Cst',
            'MAPD_Tot_Clms',
            'MAPD_Tot_Drug_Cst',
            'PDP_Tot_Clms',
            'PDP_Tot_Drug_Cst',
            'LIS_Tot_Clms',
            'LIS_Drug_Cst',
            'NonLIS_Tot_Clms',
            'NonLIS_Drug_Cst',
            'Opioid_Tot_Clms',
            'Opioid_Tot_Drug_Cst',
            'Opioid_Tot_Suply',
            'Opioid_Tot_Benes',
            'Opioid_Prscrbr_Rate',
            'Opioid_LA_Tot_Clms',
            'Opioid_LA_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Opioid_LA_Prscrbr_Rate',
            'Antbtc_Tot_Clms',
            'Antbtc_Tot_Drug_Cst',
            'Antbtc_Tot_Benes',
            'Antpsyct_GE65_Tot_Clms',
            'Antpsyct_GE65_Tot_Drug_Cst',
            'Antpsyct_GE65_Bene_Suprsn_Flag',
            'Antpsyct_GE65_Tot_Benes',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_Avg_Risk_Scre',
        ],
        'cat_features':
        [
            'Prscrbr_Type',
        ]
    },
    'part-d-noisy': {
        'file_name': f'{base_dir}/cleaned/partd-aggregated-new-features-noise.csv.gz',
        'sample_file': f'{sample_dir}/partd-aggregated-new-features-noise.csv',
        'target': 'exclusion',
        'features':  [
            'Prscrbr_Type',
            'Tot_Clms_mean',
            'Tot_Clms_median',
            'Tot_Clms_sum',
            'Tot_Clms_std',
            'Tot_Clms_min',
            'Tot_Clms_max',
            'Tot_30day_Fills_mean',
            'Tot_30day_Fills_median',
            'Tot_30day_Fills_sum',
            'Tot_30day_Fills_std',
            'Tot_30day_Fills_min',
            'Tot_30day_Fills_max',
            'Tot_Day_Suply_mean',
            'Tot_Day_Suply_median',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply_std',
            'Tot_Day_Suply_min',
            'Tot_Day_Suply_max',
            'Tot_Drug_Cst_mean',
            'Tot_Drug_Cst_median',
            'Tot_Drug_Cst_sum',
            'Tot_Drug_Cst_std',
            'Tot_Drug_Cst_min',
            'Tot_Drug_Cst_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benest_std',
            'Tot_Benes_min',
            'Tot_Benest_max',
            'Tot_Clms',
            'Tot_30day_Fills',
            'Tot_Drug_Cst',
            'Tot_Day_Suply',
            'Tot_Benes',
            'GE65_Tot_Clms',
            'GE65_Tot_30day_Fills',
            'GE65_Tot_Drug_Cst',
            'GE65_Tot_Day_Suply',
            'GE65_Tot_Benes',
            'Brnd_Tot_Clms',
            'Brnd_Tot_Drug_Cst',
            'Gnrc_Tot_Clms',
            'Gnrc_Tot_Drug_Cst',
            'Othr_Tot_Clms',
            'Othr_Tot_Drug_Cst',
            'MAPD_Tot_Clms',
            'MAPD_Tot_Drug_Cst',
            'PDP_Tot_Clms',
            'PDP_Tot_Drug_Cst',
            'LIS_Tot_Clms',
            'LIS_Drug_Cst',
            'NonLIS_Tot_Clms',
            'NonLIS_Drug_Cst',
            'Opioid_Tot_Clms',
            'Opioid_Tot_Drug_Cst',
            'Opioid_Tot_Suply',
            'Opioid_Tot_Benes',
            'Opioid_Prscrbr_Rate',
            'Opioid_LA_Tot_Clms',
            'Opioid_LA_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Opioid_LA_Prscrbr_Rate',
            'Antbtc_Tot_Clms',
            'Antbtc_Tot_Drug_Cst',
            'Antbtc_Tot_Benes',
            'Antpsyct_GE65_Tot_Clms',
            'Antpsyct_GE65_Tot_Drug_Cst',
            'Antpsyct_GE65_Bene_Suprsn_Flag',
            'Antpsyct_GE65_Tot_Benes',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_Avg_Risk_Scre',
        ],
        'cat_features':
        [
            'Prscrbr_Type',
        ]
    },
    'part-d-noisy-clean-inverted': {
        'file_name': f'/mnt/beegfs/home/jhancoc4/medicare-data/2019/partd-aggregated-new-features-noise_clean_inverted.csv.gz',
        'sample_file': f'{sample_dir}/partd-aggregated-new-features-noise_clean_inverted.csv',
        'target': 'exclusion',
        'features':  [
            'Prscrbr_Type',
            'Tot_Clms_mean',
            'Tot_Clms_median',
            'Tot_Clms_sum',
            'Tot_Clms_std',
            'Tot_Clms_min',
            'Tot_Clms_max',
            'Tot_30day_Fills_mean',
            'Tot_30day_Fills_median',
            'Tot_30day_Fills_sum',
            'Tot_30day_Fills_std',
            'Tot_30day_Fills_min',
            'Tot_30day_Fills_max',
            'Tot_Day_Suply_mean',
            'Tot_Day_Suply_median',
            'Tot_Day_Suply_sum',
            'Tot_Day_Suply_std',
            'Tot_Day_Suply_min',
            'Tot_Day_Suply_max',
            'Tot_Drug_Cst_mean',
            'Tot_Drug_Cst_median',
            'Tot_Drug_Cst_sum',
            'Tot_Drug_Cst_std',
            'Tot_Drug_Cst_min',
            'Tot_Drug_Cst_max',
            'Tot_Benes_mean',
            'Tot_Benes_median',
            'Tot_Benes_sum',
            'Tot_Benest_std',
            'Tot_Benes_min',
            'Tot_Benest_max',
            'Tot_Clms',
            'Tot_30day_Fills',
            'Tot_Drug_Cst',
            'Tot_Day_Suply',
            'Tot_Benes',
            'GE65_Tot_Clms',
            'GE65_Tot_30day_Fills',
            'GE65_Tot_Drug_Cst',
            'GE65_Tot_Day_Suply',
            'GE65_Tot_Benes',
            'Brnd_Tot_Clms',
            'Brnd_Tot_Drug_Cst',
            'Gnrc_Tot_Clms',
            'Gnrc_Tot_Drug_Cst',
            'Othr_Tot_Clms',
            'Othr_Tot_Drug_Cst',
            'MAPD_Tot_Clms',
            'MAPD_Tot_Drug_Cst',
            'PDP_Tot_Clms',
            'PDP_Tot_Drug_Cst',
            'LIS_Tot_Clms',
            'LIS_Drug_Cst',
            'NonLIS_Tot_Clms',
            'NonLIS_Drug_Cst',
            'Opioid_Tot_Clms',
            'Opioid_Tot_Drug_Cst',
            'Opioid_Tot_Suply',
            'Opioid_Tot_Benes',
            'Opioid_Prscrbr_Rate',
            'Opioid_LA_Tot_Clms',
            'Opioid_LA_Tot_Drug_Cst',
            'Opioid_LA_Tot_Suply',
            'Opioid_LA_Tot_Benes',
            'Opioid_LA_Prscrbr_Rate',
            'Antbtc_Tot_Clms',
            'Antbtc_Tot_Drug_Cst',
            'Antbtc_Tot_Benes',
            'Antpsyct_GE65_Tot_Clms',
            'Antpsyct_GE65_Tot_Drug_Cst',
            'Antpsyct_GE65_Bene_Suprsn_Flag',
            'Antpsyct_GE65_Tot_Benes',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_Avg_Risk_Scre',
        ],
        'cat_features':
        [
            'Prscrbr_Type',
        ]
    },
    'dmepos-aggregated': {
        'file_name': f'{base_dir}/medicare-dmepos-aggregated-2013-2019.csv.gz',
        'sample_file': f'{sample_dir}/medicare-dmepos-aggregated-2013-2019.csv',
        'target': 'exclusion',
        'features':
        [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
            'Tot_Suplrs_mean',
            'Tot_Suplrs_median',
            'Tot_Suplrs_sum',
            'Tot_Suplrs_std',
            'Tot_Suplrs_min',
            'Tot_Suplrs_max',
            'Tot_Suplr_Benes_mean',
            'Tot_Suplr_Benes_median',
            'Tot_Suplr_Benes_sum',
            'Tot_Suplr_Benes_std',
            'Tot_Suplr_Benes_min',
            'Tot_Suplr_Benes_max',
            'Tot_Suplr_Clms_mean',
            'Tot_Suplr_Clms_median',
            'Tot_Suplr_Clms_sum',
            'Tot_Suplr_Clms_std',
            'Tot_Suplr_Clms_min',
            'Tot_Suplr_Clms_max',
            'Tot_Suplr_Srvcs_mean',
            'Tot_Suplr_Srvcs_median',
            'Tot_Suplr_Srvcs_sum',
            'Tot_Suplr_Srvcs_std',
            'Tot_Suplr_Srvcs_min',
            'Tot_Suplr_Srvcs_max',
            'Avg_Suplr_Sbmtd_Chrg_mean',
            'Avg_Suplr_Sbmtd_Chrg_median',
            'Avg_Suplr_Sbmtd_Chrg_sum',
            'Avg_Suplr_Sbmtd_Chrg_std',
            'Avg_Suplr_Sbmtd_Chrg_min',
            'Avg_Suplr_Sbmtd_Chrg_max',
            'Avg_Suplr_Mdcr_Pymt_Amt_mean',
            'Avg_Suplr_Mdcr_Pymt_Amt_median',
            'Avg_Suplr_Mdcr_Pymt_Amt_sum',
            'Avg_Suplr_Mdcr_Pymt_Amt_std',
            'Avg_Suplr_Mdcr_Pymt_Amt_min',
            'Avg_Suplr_Mdcr_Pymt_Amt_max',
        ],
        'cat_features': [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr'
        ]
    },
    'dmepos': {
        'file_name': f'{base_dir}/medicare-dmepos-2013-2019.csv.gz',
        'sample_file': f'{sample_dir}/medicare-dmepos-2013-2019.csv',
        'target': 'exclusion',
        'features':  [
            'Rfrg_Prvdr_Crdntls',
            'Rfrg_Prvdr_Gndr',
            'Rfrg_Prvdr_Ent_Cd',
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Type_Flag',
            'BETOS_Lvl',
            'BETOS_Cd',
            'BETOS_Desc',
            'HCPCS_Cd',
            'HCPCS_Desc',
            'Suplr_Rentl_Ind',
            'Tot_Suplrs',
            'Tot_Suplr_Benes',
            'Tot_Suplr_Clms',
            'Tot_Suplr_Srvcs',
            'Avg_Suplr_Sbmtd_Chrg',
            'Avg_Suplr_Mdcr_Alowd_Amt',
            'Avg_Suplr_Mdcr_Pymt_Amt',
            # missing values 'Rfrg_Prvdr_RUCA_CAT',
            # missing values 'Rfrg_Prvdr_Type_cd',
        ],
        'cat_features': [
            'Rfrg_Prvdr_Crdntls',
            'Rfrg_Prvdr_Gndr',
            'Rfrg_Prvdr_Ent_Cd',
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Type_Flag',
            'BETOS_Lvl',
            'BETOS_Cd',
            'BETOS_Desc',
            'HCPCS_Cd',
            'HCPCS_Desc',
            'Suplr_Rentl_Ind',
        ]
    },
    'dmepos-aggregated-new': {
        'file_name': f'/mnt/beegfs/home/jhancoc4/medicare-data/2019/corrected/medicare-dmepos-aggregated-new-features-2013-2019_corrected.csv.gz',
        'sample_file': f'{sample_dir}/medicare-dmepos-aggregated-new-features-2013-2019_corrected.csv',
        'target': 'exclusion',
        'features':  [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
            'Tot_Suplrs_mean',
            'Tot_Suplrs_median',
            'Tot_Suplrs_sum',
            'Tot_Suplrs_std',
            'Tot_Suplrs_min',
            'Tot_Suplrs_max',
            'Tot_Suplr_Benes_mean',
            'Tot_Suplr_Benes_median',
            'Tot_Suplr_Benes_sum',
            'Tot_Suplr_Benes_std',
            'Tot_Suplr_Benes_min',
            'Tot_Suplr_Benes_max',
            'Tot_Suplr_Clms_mean',
            'Tot_Suplr_Clms_median',
            'Tot_Suplr_Clms_sum',
            'Tot_Suplr_Clms_std',
            'Tot_Suplr_Clms_min',
            'Tot_Suplr_Clms_max',
            'Tot_Suplr_Srvcs_mean',
            'Tot_Suplr_Srvcs_median',
            'Tot_Suplr_Srvcs_sum',
            'Tot_Suplr_Srvcs_std',
            'Tot_Suplr_Srvcs_min',
            'Tot_Suplr_Srvcs_max',
            'Avg_Suplr_Sbmtd_Chrg_mean',
            'Avg_Suplr_Sbmtd_Chrg_median',
            'Avg_Suplr_Sbmtd_Chrg_sum',
            'Avg_Suplr_Sbmtd_Chrg_std',
            'Avg_Suplr_Sbmtd_Chrg_min',
            'Avg_Suplr_Sbmtd_Chrg_max',
            'Avg_Suplr_Mdcr_Pymt_Amt_mean',
            'Avg_Suplr_Mdcr_Pymt_Amt_median',
            'Avg_Suplr_Mdcr_Pymt_Amt_sum',
            'Avg_Suplr_Mdcr_Pymt_Amt_std',
            'Avg_Suplr_Mdcr_Pymt_Amt_min',
            'Avg_Suplr_Mdcr_Pymt_Amt_max',
            'Tot_Suplrs',
            'Tot_Suplr_HCPCS_Cds',
            'Tot_Suplr_Benes',
            'Tot_Suplr_Clms',
            'Tot_Suplr_Srvcs',
            'Suplr_Sbmtd_Chrgs',
            'Suplr_Mdcr_Alowd_Amt',
            'Suplr_Mdcr_Pymt_Amt',
            'DME_Tot_Suplrs',
            'DME_Tot_Suplr_HCPCS_Cds',
            'DME_Tot_Suplr_Benes',
            'DME_Tot_Suplr_Clms',
            'DME_Tot_Suplr_Srvcs',
            'DME_Suplr_Sbmtd_Chrgs',
            'DME_Suplr_Mdcr_Alowd_Amt',
            'DME_Suplr_Mdcr_Pymt_Amt',
            'POS_Tot_Suplrs',
            'POS_Tot_Suplr_HCPCS_Cds',
            'POS_Tot_Suplr_Benes',
            'POS_Tot_Suplr_Clms',
            'POS_Tot_Suplr_Srvcs',
            'POS_Suplr_Sbmtd_Chrgs',
            'POS_Suplr_Mdcr_Alowd_Amt',
            'POS_Suplr_Mdcr_Pymt_Amt',
            'Drug_Tot_Suplrs',
            'Drug_Tot_Suplr_HCPCS_Cds',
            'Drug_Tot_Suplr_Benes',
            'Drug_Tot_Suplr_Clms',
            'Drug_Tot_Suplr_Srvcs',
            'Drug_Suplr_Sbmtd_Chrgs',
            'Drug_Suplr_Mdcr_Alowd_Amt',
            'Drug_Suplr_Mdcr_Pymt_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre',
        ],
        'cat_features': [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
        ]
    },
    'dmepos-aggregated-new-pre-encoded': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
            'Tot_Suplrs_mean',
            'Tot_Suplrs_median',
            'Tot_Suplrs_sum',
            'Tot_Suplrs_std',
            'Tot_Suplrs_min',
            'Tot_Suplrs_max',
            'Tot_Suplr_Benes_mean',
            'Tot_Suplr_Benes_median',
            'Tot_Suplr_Benes_sum',
            'Tot_Suplr_Benes_std',
            'Tot_Suplr_Benes_min',
            'Tot_Suplr_Benes_max',
            'Tot_Suplr_Clms_mean',
            'Tot_Suplr_Clms_median',
            'Tot_Suplr_Clms_sum',
            'Tot_Suplr_Clms_std',
            'Tot_Suplr_Clms_min',
            'Tot_Suplr_Clms_max',
            'Tot_Suplr_Srvcs_mean',
            'Tot_Suplr_Srvcs_median',
            'Tot_Suplr_Srvcs_sum',
            'Tot_Suplr_Srvcs_std',
            'Tot_Suplr_Srvcs_min',
            'Tot_Suplr_Srvcs_max',
            'Avg_Suplr_Sbmtd_Chrg_mean',
            'Avg_Suplr_Sbmtd_Chrg_median',
            'Avg_Suplr_Sbmtd_Chrg_sum',
            'Avg_Suplr_Sbmtd_Chrg_std',
            'Avg_Suplr_Sbmtd_Chrg_min',
            'Avg_Suplr_Sbmtd_Chrg_max',
            'Avg_Suplr_Mdcr_Pymt_Amt_mean',
            'Avg_Suplr_Mdcr_Pymt_Amt_median',
            'Avg_Suplr_Mdcr_Pymt_Amt_sum',
            'Avg_Suplr_Mdcr_Pymt_Amt_std',
            'Avg_Suplr_Mdcr_Pymt_Amt_min',
            'Avg_Suplr_Mdcr_Pymt_Amt_max',
            'Tot_Suplrs',
            'Tot_Suplr_HCPCS_Cds',
            'Tot_Suplr_Benes',
            'Tot_Suplr_Clms',
            'Tot_Suplr_Srvcs',
            'Suplr_Sbmtd_Chrgs',
            'Suplr_Mdcr_Alowd_Amt',
            'Suplr_Mdcr_Pymt_Amt',
            'DME_Tot_Suplrs',
            'DME_Tot_Suplr_HCPCS_Cds',
            'DME_Tot_Suplr_Benes',
            'DME_Tot_Suplr_Clms',
            'DME_Tot_Suplr_Srvcs',
            'DME_Suplr_Sbmtd_Chrgs',
            'DME_Suplr_Mdcr_Alowd_Amt',
            'DME_Suplr_Mdcr_Pymt_Amt',
            'POS_Tot_Suplrs',
            'POS_Tot_Suplr_HCPCS_Cds',
            'POS_Tot_Suplr_Benes',
            'POS_Tot_Suplr_Clms',
            'POS_Tot_Suplr_Srvcs',
            'POS_Suplr_Sbmtd_Chrgs',
            'POS_Suplr_Mdcr_Alowd_Amt',
            'POS_Suplr_Mdcr_Pymt_Amt',
            'Drug_Tot_Suplrs',
            'Drug_Tot_Suplr_HCPCS_Cds',
            'Drug_Tot_Suplr_Benes',
            'Drug_Tot_Suplr_Clms',
            'Drug_Tot_Suplr_Srvcs',
            'Drug_Suplr_Sbmtd_Chrgs',
            'Drug_Suplr_Mdcr_Alowd_Amt',
            'Drug_Suplr_Mdcr_Pymt_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre',
        ],
        'cat_features': [
        ]
    },
    'dmepos-aggregated-new-sfs-5-top-10': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",	    
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Clms_max",
            "Avg_Suplr_Sbmtd_Chrg_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_max",
            "Tot_Suplr_Srvcs_sum",
        ],
        'cat_features': [
        ]
    },
    'dmepos-aggregated-new-sfs-5-top-15': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",	    
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Clms_max",
            "Avg_Suplr_Sbmtd_Chrg_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_max",
            "Tot_Suplr_Srvcs_sum",
            "Rfrg_Prvdr_Type",
            "Avg_Suplr_Mdcr_Pymt_Amt_std",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Benes",
            "Tot_Suplr_Srvcs_max",
        ],
        'cat_features': [
        ]
    },
        'dmepos-aggregated-new-sfs-5-top-20': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",	    
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Clms_max",
            "Avg_Suplr_Sbmtd_Chrg_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_max",
            "Tot_Suplr_Srvcs_sum",
            "Rfrg_Prvdr_Type",
            "Avg_Suplr_Mdcr_Pymt_Amt_std",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Benes",
            "Tot_Suplr_Srvcs_max",
            "Suplr_Mdcr_Alowd_Amt",
            "Tot_Suplr_Clms_std",
            "Suplr_Mdcr_Pymt_Amt",
            "Avg_Suplr_Sbmtd_Chrg_std",
            "Avg_Suplr_Mdcr_Pymt_Amt_min",
        ],
        'cat_features': [
        ]
        },
    'dmepos-aggregated-new-sfs-5-top-25': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",	    
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Clms_max",
            "Avg_Suplr_Sbmtd_Chrg_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_max",
            "Tot_Suplr_Srvcs_sum",
            "Rfrg_Prvdr_Type",
            "Avg_Suplr_Mdcr_Pymt_Amt_std",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Benes",
            "Tot_Suplr_Srvcs_max",
            "Suplr_Mdcr_Alowd_Amt",
            "Tot_Suplr_Clms_std",
            "Suplr_Mdcr_Pymt_Amt",
            "Avg_Suplr_Sbmtd_Chrg_std",
            "Avg_Suplr_Mdcr_Pymt_Amt_min",
            "Avg_Suplr_Sbmtd_Chrg_min",
            "DME_Suplr_Sbmtd_Chrgs",
            "Tot_Suplr_Benes_max",
            "Tot_Suplr_Srvcs_std",
            "Tot_Suplr_Clms_mean",            
            
        ],
        'cat_features': [
        ]
        },
    'dmepos-aggregated-new-sfs-5-top-30': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",	    
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Clms_max",
            "Avg_Suplr_Sbmtd_Chrg_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_max",
            "Tot_Suplr_Srvcs_sum",
            "Rfrg_Prvdr_Type",
            "Avg_Suplr_Mdcr_Pymt_Amt_std",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Benes",
            "Tot_Suplr_Srvcs_max",
            "Suplr_Mdcr_Alowd_Amt",
            "Tot_Suplr_Clms_std",
            "Suplr_Mdcr_Pymt_Amt",
            "Avg_Suplr_Sbmtd_Chrg_std",
            "Avg_Suplr_Mdcr_Pymt_Amt_min",
            "Avg_Suplr_Sbmtd_Chrg_min",
            "DME_Suplr_Sbmtd_Chrgs",
            "Tot_Suplr_Benes_max",
            "Tot_Suplr_Srvcs_std",
            "Tot_Suplr_Clms_mean",
            "Avg_Suplr_Mdcr_Pymt_Amt_median",
            "Avg_Suplr_Mdcr_Pymt_Amt_mean",
            "Tot_Suplr_Srvcs",
            "DME_Tot_Suplr_Clms",
            "Avg_Suplr_Sbmtd_Chrg_mean",            
        ],
        'cat_features': [
        ]
        },
        'dmepos-aggregated-new-sfs-6-top-10': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
            "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Srvcs_sum",
            "Tot_Suplr_Clms_max",
            "Rfrg_Prvdr_Type",
            "Avg_Suplr_Sbmtd_Chrg_max",

        ],
        'cat_features': [
        ]
        },
    'dmepos-aggregated-new-sfs-6-top-15': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
        "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Srvcs_sum",
            "Tot_Suplr_Clms_max",
            "Rfrg_Prvdr_Type",
            "Avg_Suplr_Sbmtd_Chrg_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Srvcs_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_std",
            "Suplr_Mdcr_Alowd_Amt",            
        ],
        'cat_features': [
        ]
    },
    'dmepos-aggregated-new-sfs-6-top-20': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
        "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Srvcs_sum",
            "Tot_Suplr_Clms_max",
            "Rfrg_Prvdr_Type",
            "Avg_Suplr_Sbmtd_Chrg_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Srvcs_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_std",
            "Suplr_Mdcr_Alowd_Amt",
            "Tot_Suplr_Clms_std",
            "Avg_Suplr_Sbmtd_Chrg_min",
            "Tot_Suplr_Srvcs",
            "Suplr_Mdcr_Pymt_Amt",
            "Tot_Suplr_Benes",
        ],
        'cat_features': [
        ]
    },
        'dmepos-aggregated-new-sfs-6-top-25': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
        "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Srvcs_sum",
            "Tot_Suplr_Clms_max",
            "Rfrg_Prvdr_Type",
            "Avg_Suplr_Sbmtd_Chrg_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Srvcs_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_std",
            "Suplr_Mdcr_Alowd_Amt",
            "Tot_Suplr_Clms_std",
            "Avg_Suplr_Sbmtd_Chrg_min",
            "Tot_Suplr_Srvcs",
            "Suplr_Mdcr_Pymt_Amt",
            "Tot_Suplr_Benes",
            "Avg_Suplr_Sbmtd_Chrg_std",
            "DME_Suplr_Sbmtd_Chrgs",
            "Tot_Suplr_Clms_mean",
            "Tot_Suplr_Benes_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_median",
        ],
        'cat_features': [
        ]
    },

    'dmepos-aggregated-new-sfs-6-top-30': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
        "Tot_Suplr_Clms_sum",
            "Suplr_Sbmtd_Chrgs",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Tot_Suplr_Clms",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplr_Srvcs_sum",
            "Tot_Suplr_Clms_max",
            "Rfrg_Prvdr_Type",
            "Avg_Suplr_Sbmtd_Chrg_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Srvcs_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_std",
            "Suplr_Mdcr_Alowd_Amt",
            "Tot_Suplr_Clms_std",
            "Avg_Suplr_Sbmtd_Chrg_min",
            "Tot_Suplr_Srvcs",
            "Suplr_Mdcr_Pymt_Amt",
            "Tot_Suplr_Benes",
            "Avg_Suplr_Sbmtd_Chrg_std",
            "DME_Suplr_Sbmtd_Chrgs",
            "Tot_Suplr_Clms_mean",
            "Tot_Suplr_Benes_max",
            "Avg_Suplr_Mdcr_Pymt_Amt_median",
            "Avg_Suplr_Mdcr_Pymt_Amt_min",
            "Tot_Suplr_Srvcs_std",
            "Tot_Suplr_Benes_mean",
            "Rfrg_Prvdr_Gndr",
            "Tot_Suplrs_mean",
        ],
        'cat_features': [
        ]
    },
    'dmepos-aggregated-new-tbfs-top-10': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Clms_sum",
            "POS_Suplr_Mdcr_Alowd_Amt",
            "POS_Suplr_Mdcr_Pymt_Amt",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplrs",
            "Tot_Suplr_HCPCS_Cds",
            "POS_Suplr_Sbmtd_Chrgs",
        ],
        'cat_features': [
        ]
    },

    'dmepos-aggregated-new-tbfs-top-15': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Clms_sum",
            "POS_Suplr_Mdcr_Alowd_Amt",
            "POS_Suplr_Mdcr_Pymt_Amt",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplrs",
            "Tot_Suplr_HCPCS_Cds",
            "POS_Suplr_Sbmtd_Chrgs",
            "Bene_Feml_Cnt",
            "DME_Tot_Suplrs",
            "POS_Tot_Suplr_Benes",
            "POS_Tot_Suplrs",
            "POS_Tot_Suplr_Clms",            
        ],
        'cat_features': [
        ]
    },
    'dmepos-aggregated-new-tbfs-top-20': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Clms_sum",
            "POS_Suplr_Mdcr_Alowd_Amt",
            "POS_Suplr_Mdcr_Pymt_Amt",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplrs",
            "Tot_Suplr_HCPCS_Cds",
            "POS_Suplr_Sbmtd_Chrgs",
            "Bene_Feml_Cnt",
            "DME_Tot_Suplrs",
            "POS_Tot_Suplr_Benes",
            "POS_Tot_Suplrs",
            "POS_Tot_Suplr_Clms",
            "Tot_Suplr_Benes",
            "Rfrg_Prvdr_Type",
            "Tot_Suplrs_max",
            "DME_Tot_Suplr_HCPCS_Cds",
            "Bene_Age_LT_65_Cnt",            
        ],
        'cat_features': [
        ]
    },
    'dmepos-aggregated-new-tbfs-top-25': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Clms_sum",
            "POS_Suplr_Mdcr_Alowd_Amt",
            "POS_Suplr_Mdcr_Pymt_Amt",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplrs",
            "Tot_Suplr_HCPCS_Cds",
            "POS_Suplr_Sbmtd_Chrgs",
            "Bene_Feml_Cnt",
            "DME_Tot_Suplrs",
            "POS_Tot_Suplr_Benes",
            "POS_Tot_Suplrs",
            "POS_Tot_Suplr_Clms",
            "Tot_Suplr_Benes",
            "Rfrg_Prvdr_Type",
            "Tot_Suplrs_max",
            "DME_Tot_Suplr_HCPCS_Cds",
            "Bene_Age_LT_65_Cnt",
            "Bene_Age_75_84_Cnt",
            "Tot_Suplr_Benes_mean",
            "Tot_Suplrs_mean",
            "Tot_Suplr_Benes_std",
            "Tot_Suplr_Benes_max",            
        ],
        'cat_features': [
        ]
    },
    'dmepos-aggregated-new-tbfs-top-30': {
        'file_name': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'sample_file': '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-dmepos-aggregated-new-features-2013-2019-cb-encoded.csv.gz',
        'target': 'exclusion',
        'features':  [
            "Tot_Suplrs_sum",
            "Avg_Suplr_Sbmtd_Chrg_sum",
            "Avg_Suplr_Mdcr_Pymt_Amt_sum",
            "Tot_Suplr_Clms_sum",
            "POS_Suplr_Mdcr_Alowd_Amt",
            "POS_Suplr_Mdcr_Pymt_Amt",
            "Tot_Suplr_Benes_sum",
            "Tot_Suplrs",
            "Tot_Suplr_HCPCS_Cds",
            "POS_Suplr_Sbmtd_Chrgs",
            "Bene_Feml_Cnt",
            "DME_Tot_Suplrs",
            "POS_Tot_Suplr_Benes",
            "POS_Tot_Suplrs",
            "POS_Tot_Suplr_Clms",
            "Tot_Suplr_Benes",
            "Rfrg_Prvdr_Type",
            "Tot_Suplrs_max",
            "DME_Tot_Suplr_HCPCS_Cds",
            "Bene_Age_LT_65_Cnt",
            "Bene_Age_75_84_Cnt",
            "Tot_Suplr_Benes_mean",
            "Tot_Suplrs_mean",
            "Tot_Suplr_Benes_std",
            "Tot_Suplr_Benes_max",
            "Tot_Suplrs_median",
            "Bene_Dual_Cnt",
            "DME_Tot_Suplr_Benes",
            "Bene_Ndual_Cnt",
            "Suplr_Mdcr_Alowd_Amt",            
        ],
        'cat_features': [
        ]
    },                
    'dmepos-cleaned': {
        'file_name': f'/mnt/beegfs/home/jhancoc4/medicare-data/2019/corrected/dmepos-aggregated-new-features-cleaned_corrected.csv.gz',
        'sample_file': f'{sample_dir}/dmepos-aggregated-new-features-cleaned_corrected.csv',
        'target': 'exclusion',
        'features':  [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
            'Tot_Suplrs_mean',
            'Tot_Suplrs_median',
            'Tot_Suplrs_sum',
            'Tot_Suplrs_std',
            'Tot_Suplrs_min',
            'Tot_Suplrs_max',
            'Tot_Suplr_Benes_mean',
            'Tot_Suplr_Benes_median',
            'Tot_Suplr_Benes_sum',
            'Tot_Suplr_Benes_std',
            'Tot_Suplr_Benes_min',
            'Tot_Suplr_Benes_max',
            'Tot_Suplr_Clms_mean',
            'Tot_Suplr_Clms_median',
            'Tot_Suplr_Clms_sum',
            'Tot_Suplr_Clms_std',
            'Tot_Suplr_Clms_min',
            'Tot_Suplr_Clms_max',
            'Tot_Suplr_Srvcs_mean',
            'Tot_Suplr_Srvcs_median',
            'Tot_Suplr_Srvcs_sum',
            'Tot_Suplr_Srvcs_std',
            'Tot_Suplr_Srvcs_min',
            'Tot_Suplr_Srvcs_max',
            'Avg_Suplr_Sbmtd_Chrg_mean',
            'Avg_Suplr_Sbmtd_Chrg_median',
            'Avg_Suplr_Sbmtd_Chrg_sum',
            'Avg_Suplr_Sbmtd_Chrg_std',
            'Avg_Suplr_Sbmtd_Chrg_min',
            'Avg_Suplr_Sbmtd_Chrg_max',
            'Avg_Suplr_Mdcr_Pymt_Amt_mean',
            'Avg_Suplr_Mdcr_Pymt_Amt_median',
            'Avg_Suplr_Mdcr_Pymt_Amt_sum',
            'Avg_Suplr_Mdcr_Pymt_Amt_std',
            'Avg_Suplr_Mdcr_Pymt_Amt_min',
            'Avg_Suplr_Mdcr_Pymt_Amt_max',
            'Tot_Suplrs',
            'Tot_Suplr_HCPCS_Cds',
            'Tot_Suplr_Benes',
            'Tot_Suplr_Clms',
            'Tot_Suplr_Srvcs',
            'Suplr_Sbmtd_Chrgs',
            'Suplr_Mdcr_Alowd_Amt',
            'Suplr_Mdcr_Pymt_Amt',
            'DME_Tot_Suplrs',
            'DME_Tot_Suplr_HCPCS_Cds',
            'DME_Tot_Suplr_Benes',
            'DME_Tot_Suplr_Clms',
            'DME_Tot_Suplr_Srvcs',
            'DME_Suplr_Sbmtd_Chrgs',
            'DME_Suplr_Mdcr_Alowd_Amt',
            'DME_Suplr_Mdcr_Pymt_Amt',
            'POS_Tot_Suplrs',
            'POS_Tot_Suplr_HCPCS_Cds',
            'POS_Tot_Suplr_Benes',
            'POS_Tot_Suplr_Clms',
            'POS_Tot_Suplr_Srvcs',
            'POS_Suplr_Sbmtd_Chrgs',
            'POS_Suplr_Mdcr_Alowd_Amt',
            'POS_Suplr_Mdcr_Pymt_Amt',
            'Drug_Tot_Suplrs',
            'Drug_Tot_Suplr_HCPCS_Cds',
            'Drug_Tot_Suplr_Benes',
            'Drug_Tot_Suplr_Clms',
            'Drug_Tot_Suplr_Srvcs',
            'Drug_Suplr_Sbmtd_Chrgs',
            'Drug_Suplr_Mdcr_Alowd_Amt',
            'Drug_Suplr_Mdcr_Pymt_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre',
        ],
        'cat_features': [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
        ]
    },
    'dmepos-noisy': {
        'file_name': f'/mnt/beegfs/home/jhancoc4/medicare-data/2019/corrected/dmepos-aggregated-new-features-noise_corrected.csv.gz',
        'sample_file': f'{sample_dir}/dmepos-aggregated-new-features-noise_corrected.csv',
        'target': 'exclusion',
        'features':  [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
            'Tot_Suplrs_mean',
            'Tot_Suplrs_median',
            'Tot_Suplrs_sum',
            'Tot_Suplrs_std',
            'Tot_Suplrs_min',
            'Tot_Suplrs_max',
            'Tot_Suplr_Benes_mean',
            'Tot_Suplr_Benes_median',
            'Tot_Suplr_Benes_sum',
            'Tot_Suplr_Benes_std',
            'Tot_Suplr_Benes_min',
            'Tot_Suplr_Benes_max',
            'Tot_Suplr_Clms_mean',
            'Tot_Suplr_Clms_median',
            'Tot_Suplr_Clms_sum',
            'Tot_Suplr_Clms_std',
            'Tot_Suplr_Clms_min',
            'Tot_Suplr_Clms_max',
            'Tot_Suplr_Srvcs_mean',
            'Tot_Suplr_Srvcs_median',
            'Tot_Suplr_Srvcs_sum',
            'Tot_Suplr_Srvcs_std',
            'Tot_Suplr_Srvcs_min',
            'Tot_Suplr_Srvcs_max',
            'Avg_Suplr_Sbmtd_Chrg_mean',
            'Avg_Suplr_Sbmtd_Chrg_median',
            'Avg_Suplr_Sbmtd_Chrg_sum',
            'Avg_Suplr_Sbmtd_Chrg_std',
            'Avg_Suplr_Sbmtd_Chrg_min',
            'Avg_Suplr_Sbmtd_Chrg_max',
            'Avg_Suplr_Mdcr_Pymt_Amt_mean',
            'Avg_Suplr_Mdcr_Pymt_Amt_median',
            'Avg_Suplr_Mdcr_Pymt_Amt_sum',
            'Avg_Suplr_Mdcr_Pymt_Amt_std',
            'Avg_Suplr_Mdcr_Pymt_Amt_min',
            'Avg_Suplr_Mdcr_Pymt_Amt_max',
            'Tot_Suplrs',
            'Tot_Suplr_HCPCS_Cds',
            'Tot_Suplr_Benes',
            'Tot_Suplr_Clms',
            'Tot_Suplr_Srvcs',
            'Suplr_Sbmtd_Chrgs',
            'Suplr_Mdcr_Alowd_Amt',
            'Suplr_Mdcr_Pymt_Amt',
            'DME_Tot_Suplrs',
            'DME_Tot_Suplr_HCPCS_Cds',
            'DME_Tot_Suplr_Benes',
            'DME_Tot_Suplr_Clms',
            'DME_Tot_Suplr_Srvcs',
            'DME_Suplr_Sbmtd_Chrgs',
            'DME_Suplr_Mdcr_Alowd_Amt',
            'DME_Suplr_Mdcr_Pymt_Amt',
            'POS_Tot_Suplrs',
            'POS_Tot_Suplr_HCPCS_Cds',
            'POS_Tot_Suplr_Benes',
            'POS_Tot_Suplr_Clms',
            'POS_Tot_Suplr_Srvcs',
            'POS_Suplr_Sbmtd_Chrgs',
            'POS_Suplr_Mdcr_Alowd_Amt',
            'POS_Suplr_Mdcr_Pymt_Amt',
            'Drug_Tot_Suplrs',
            'Drug_Tot_Suplr_HCPCS_Cds',
            'Drug_Tot_Suplr_Benes',
            'Drug_Tot_Suplr_Clms',
            'Drug_Tot_Suplr_Srvcs',
            'Drug_Suplr_Sbmtd_Chrgs',
            'Drug_Suplr_Mdcr_Alowd_Amt',
            'Drug_Suplr_Mdcr_Pymt_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre',
        ],
        'cat_features': [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
        ]
    },
    'dmepos-noisy-clean-inverted': {
        'file_name': f'/mnt/beegfs/home/jhancoc4/medicare-data/2019/corrected/dmepos-aggregated-new-features-noise_clean_inverted_corrected.csv.gz',
        'sample_file': f'{sample_dir}/dmepos-aggregated-new-features-noise_clean_inverted_corrected.csv',
        'target': 'exclusion',
        'features':  [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
            'Tot_Suplrs_mean',
            'Tot_Suplrs_median',
            'Tot_Suplrs_sum',
            'Tot_Suplrs_std',
            'Tot_Suplrs_min',
            'Tot_Suplrs_max',
            'Tot_Suplr_Benes_mean',
            'Tot_Suplr_Benes_median',
            'Tot_Suplr_Benes_sum',
            'Tot_Suplr_Benes_std',
            'Tot_Suplr_Benes_min',
            'Tot_Suplr_Benes_max',
            'Tot_Suplr_Clms_mean',
            'Tot_Suplr_Clms_median',
            'Tot_Suplr_Clms_sum',
            'Tot_Suplr_Clms_std',
            'Tot_Suplr_Clms_min',
            'Tot_Suplr_Clms_max',
            'Tot_Suplr_Srvcs_mean',
            'Tot_Suplr_Srvcs_median',
            'Tot_Suplr_Srvcs_sum',
            'Tot_Suplr_Srvcs_std',
            'Tot_Suplr_Srvcs_min',
            'Tot_Suplr_Srvcs_max',
            'Avg_Suplr_Sbmtd_Chrg_mean',
            'Avg_Suplr_Sbmtd_Chrg_median',
            'Avg_Suplr_Sbmtd_Chrg_sum',
            'Avg_Suplr_Sbmtd_Chrg_std',
            'Avg_Suplr_Sbmtd_Chrg_min',
            'Avg_Suplr_Sbmtd_Chrg_max',
            'Avg_Suplr_Mdcr_Pymt_Amt_mean',
            'Avg_Suplr_Mdcr_Pymt_Amt_median',
            'Avg_Suplr_Mdcr_Pymt_Amt_sum',
            'Avg_Suplr_Mdcr_Pymt_Amt_std',
            'Avg_Suplr_Mdcr_Pymt_Amt_min',
            'Avg_Suplr_Mdcr_Pymt_Amt_max',
            'Tot_Suplrs',
            'Tot_Suplr_HCPCS_Cds',
            'Tot_Suplr_Benes',
            'Tot_Suplr_Clms',
            'Tot_Suplr_Srvcs',
            'Suplr_Sbmtd_Chrgs',
            'Suplr_Mdcr_Alowd_Amt',
            'Suplr_Mdcr_Pymt_Amt',
            'DME_Tot_Suplrs',
            'DME_Tot_Suplr_HCPCS_Cds',
            'DME_Tot_Suplr_Benes',
            'DME_Tot_Suplr_Clms',
            'DME_Tot_Suplr_Srvcs',
            'DME_Suplr_Sbmtd_Chrgs',
            'DME_Suplr_Mdcr_Alowd_Amt',
            'DME_Suplr_Mdcr_Pymt_Amt',
            'POS_Tot_Suplrs',
            'POS_Tot_Suplr_HCPCS_Cds',
            'POS_Tot_Suplr_Benes',
            'POS_Tot_Suplr_Clms',
            'POS_Tot_Suplr_Srvcs',
            'POS_Suplr_Sbmtd_Chrgs',
            'POS_Suplr_Mdcr_Alowd_Amt',
            'POS_Suplr_Mdcr_Pymt_Amt',
            'Drug_Tot_Suplrs',
            'Drug_Tot_Suplr_HCPCS_Cds',
            'Drug_Tot_Suplr_Benes',
            'Drug_Tot_Suplr_Clms',
            'Drug_Tot_Suplr_Srvcs',
            'Drug_Suplr_Sbmtd_Chrgs',
            'Drug_Suplr_Mdcr_Alowd_Amt',
            'Drug_Suplr_Mdcr_Pymt_Amt',
            'Bene_Avg_Age',
            'Bene_Age_LT_65_Cnt',
            'Bene_Age_65_74_Cnt',
            'Bene_Age_75_84_Cnt',
            'Bene_Age_GT_84_Cnt',
            'Bene_Feml_Cnt',
            'Bene_Male_Cnt',
            'Bene_Dual_Cnt',
            'Bene_Ndual_Cnt',
            'Bene_CC_AF_Pct',
            'Bene_CC_Alzhmr_Pct',
            'Bene_CC_Asthma_Pct',
            'Bene_CC_Cncr_Pct',
            'Bene_CC_CHF_Pct',
            'Bene_CC_CKD_Pct',
            'Bene_CC_COPD_Pct',
            'Bene_CC_Dprssn_Pct',
            'Bene_CC_Dbts_Pct',
            'Bene_CC_Hyplpdma_Pct',
            'Bene_CC_Hyprtnsn_Pct',
            'Bene_CC_IHD_Pct',
            'Bene_CC_Opo_Pct',
            'Bene_CC_RAOA_Pct',
            'Bene_CC_Sz_Pct',
            'Bene_CC_Strok_Pct',
            'Bene_Avg_Risk_Scre',
        ],
        'cat_features': [
            'Rfrg_Prvdr_Type',
            'Rfrg_Prvdr_Gndr',
        ]
    },
    'credit': {
        'file_name': f'/mnt/beegfs/groups/fau-bigdata-datasets/rkennedy/creditcard/creditcard11.csv',
        'sample_file': f'/mnt/beegfs/groups/fau-bigdata-datasets/rkennedy/creditcard/creditcard11.csv',
        'target': 'Class',
        'features':  [
                       "V1",
                       "V2",
                       "V3",
                       "V4",
                       "V5",
                       "V6",
                       "V7",
                       "V8",
                       "V9",
                       "V10",
                       "V11",
                       "V12",
                       "V13",
                       "V14",
                       "V15",
                       "V16",
                       "V17",
                       "V18",
                       "V19",
                       "V20",
                       "V21",
                       "V22",
                       "V23",
                       "V24",
                       "V25",
                       "V26",
                       "V27",
                       "V28",
                       "Amount",
        ],
        'cat_features': [
        ]
    },
    'diabetes': {
        'file_name': '/home/jhancoc4/diabetes.csv',
        'sample_file': '/home/jhancoc4/diabetes.csv',
        'target': 'Outcome',
        'features': [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ],
        'cat_features': []
    }
}

part_d_aggregated_new_pre_encoded = datasets_dict['part-d-aggregated-new'].copy()
del part_d_aggregated_new_pre_encoded['cat_features']
part_d_aggregated_new_pre_encoded['cat_features'] = []
part_d_aggregated_new_pre_encoded['file_name'] = '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz'
part_d_aggregated_new_pre_encoded['sample_file'] =  '/home/groups/fau-bigdata-datasets/medicare/2019/catboost_encoded/medicare-partd-aggregated-new-features-2013-2019-cb-encoded.csv.gz'
datasets_dict['part_d_aggregated_new_pre_encoded'] = part_d_aggregated_new_pre_encoded

def get_clf_params_dict() -> dict:
    """
    create dictionary of classifierg parameters with 
    specified random_state values
    iterate through datasets and add categorical features
    make one set of model parameters for every dataset
    :param seed: initial seed value for random_state
    :return: dictionary of classifiers
    """
    clf_params_dict = {
        "catboost":  {
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
            }
        },
        "catboost_100":  {
            # catboost with 100 trees
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'n_estimators': 100
            }
        },
        "catboost_200":  {
            # catboost with 200 trees
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'n_estimators': 200
            }
        },
        "catboost_400":  {
            # catboost with 400 trees
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'n_estimators': 400
            }
        },
        "catboost_depth":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 16,
                'n_estimators': 100
            }
        },
        "catboost_depth_24":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 24,
                'n_estimators': 100
            }
        },
        "catboost_depth_5":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 5,
            }
        },
        "catboost_depth_4":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 4,
            }
        },
        "catboost_depth_3":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 3,
            }
        },
        "catboost_depth_2":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 2,
            }
        },
        "catboost_depth_1":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 1,
            }
        },
    "catboost_depth_6":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 6,
            }
        },
        "catboost_depth_7":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 7,
            }
        },
        "catboost_depth_8":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 8,
            }
        },
        "catboost_depth_9":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 9,
            }
        },
        "catboost_depth_10":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 10,
            }
        },
        "catboost_depth_11":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 11,
            }
        },
        "catboost_depth_12":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 12,
            }
        },
        "catboost_depth_13":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 3,
            }
        },
        "catboost_depth_14":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 14,
            }
        },
        "catboost_depth_15":  {
            # catboost more like rf
            'model_name': 'catboost',
            'module name': 'catboost',
            'class name': 'CatBoostClassifier',
            'model_params': {
                'task_type': 'GPU',
                'devices': '0,1',
                'logging_level': 'Silent',
                'max_ctr_complexity': 1,
                'random_state': next_seed(),
                'max_depth': 15,
            }
        },
        "xgboost": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed()
            }
        },
        "xgboost_100": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 100
            }
        },
        "xgboost_200": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 200
            }
        },
        "xgboost_400": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 400
            }
        },
        "xgboost_depth": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 100,
                'max_depth': 16
            }
        },
        "xgboost_depth_4": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 4
            }
        },
        "xgboost_depth_1": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 1
            }
        },
        "xgboost_depth_2": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 2
            }
        },
        "xgboost_depth_3": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 3
            }
        },
        "xgboost_depth_4": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 4
            }
        },
        "xgboost_depth_4_estim_1000": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 4,
                'n_estimators': 1000
            }
        },
        "xgboost_depth_5": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 5
            }
        },
        "xgboost_depth_6": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 6
            }
        },
        "xgboost_depth_7": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 7
            }
        },
        "xgboost_depth_8": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 8
            }
        },
        "xgboost_depth_9": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 9
            }
        },
        "xgboost_depth_10": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 10
            }
        },
        "xgboost_depth_11": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 11
            }
        },
        "xgboost_depth_12": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 12
            }
        },
        "xgboost_depth_13": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 13
            }
        },
        "xgboost_depth_14": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 14
            }
        },
        "xgboost_depth_15": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'max_depth': 15
            }
        },
        "xgboost_depth_24": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 100,
                'max_depth': 24
            }
        },
        "xgboost_depth_30": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 100,
                'max_depth': 30
            }
        },
        "xgboost_depth_31": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 100,
                'max_depth': 31
            }
        },
        "xgboost_depth_32": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 100,
                'max_depth': 32
            }
        },
        "xgboost_depth_6": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 100,
                'max_depth': 6
            }
        },
        "xgboost_depth_48": {
            'model_name': 'xgboost',
            'module name': 'xgboost',
            'class name': 'XGBClassifier',
            'model_params': {
                'tree_method': 'gpu_hist',
                'gpu_id': '0',
                'random_state': next_seed(),
                'n_estimators': 100,
                'max_depth': 48
            }
        },
        "lightgbm": {
            'model_name': 'lightgbm',
            'module name': 'lightgbm',
            'class name': 'LGBMClassifier',
            'model_params': {
                'num_threads': 64,
                'random_state': next_seed(),
            }
        },
        "lightgbm_depth_4": {
            'model_name': 'lightgbm',
            'module name': 'lightgbm',
            'class name': 'LGBMClassifier',
            'model_params': {
                'num_threads': 64,
                'random_state': next_seed(),
                'max_depth': 4
            }
        },
        "lightgbm_depth_3": {
            'model_name': 'lightgbm',
            'module name': 'lightgbm',
            'class name': 'LGBMClassifier',
            'model_params': {
                'num_threads': 64,
                'random_state': next_seed(),
                'max_depth': 3
            }
        },
        "lightgbm_depth_2": {
            'model_name': 'lightgbm',
            'module name': 'lightgbm',
            'class name': 'LGBMClassifier',
            'model_params': {
                'num_threads': 64,
                'random_state': next_seed(),
                'max_depth': 2
            }
        },
        "lightgbm_depth_1": {
            'model_name': 'lightgbm',
            'module name': 'lightgbm',
            'class name': 'LGBMClassifier',
            'model_params': {
                'num_threads': 64,
                'random_state': next_seed(),
                'max_depth': 1
            }
        },
        "lightgbm_is_unbalance": {
            'model_name': 'lightgbm',
            'module name': 'lightgbm',
            'class name': 'LGBMClassifier',
            'model_params': {
                'random_state': next_seed(),
                'is_unbalance': True,
                'num_threads': 64
            }
        },
        "lightgbm_kaggle": {
            'model_name': 'lightgbm',
            'module name': 'lightgbm',
            'class name': 'LGBMClassifier',
            'model_params': {
                'random_state': next_seed(),
                'num_threads': 64,
                # options from kaggle
                # https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-lb-0-9680
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.01,
                # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
                # we should let it be smaller than 2^(max_depth)
                'num_leaves': 31,
                'max_depth': -1,  # -1 means no limit
                # Minimum number of data need in a child(min_data_in_leaf)
                'min_child_samples': 20,
                'max_bin': 255,  # Number of bucketed bin for feature values
                'subsample': 0.6,  # Subsample ratio of the training instance.
                'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
                # Subsample ratio of columns when constructing each tree.
                'colsample_bytree': 0.3,
                # Minimum sum of instance weight(hessian) needed in a child(leaf)
                'min_child_weight': 5,
                'subsample_for_bin': 200000,  # Number of samples for constructing bin
                'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
                'reg_alpha': 0,  # L1 regularization term on weights
                'reg_lambda': 0,  # L2 regularization term on weights
                'verbose': 0,
                'metric': 'auc'
            }
        },
        "lgb_cb": {
            'model_name': 'lightgbm',
            'module name': 'lightgbm',
            'class name': 'LGBMClassifier',
            'model_params': {
                'num_threads': 64,
                'random_state': next_seed(),
            }
        },
        "lgb_scale_weight": {
            'model_name': 'lightgbm',
            'module name': 'lightgbm',
            'class name': 'LGBMClassifier',
            'model_params': {
                'num_threads': 64,
                'random_state': next_seed(),
                'is_unbalance': True
            }
        },
        "rf": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
            },
        },
        "rf_depth_16": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 16
            },
        },
        "rf_depth_11": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 11
            },
        },
        "rf_depth_12": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 12
            },
        },
        
        "rf_depth_13": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 13
            },
        },
        "rf_depth_14": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 14
            },
        },
        "rf_depth_15": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 15
            },
        },

        "rf_depth_10": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 10
            },
        },
        "rf_depth_9": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 9
            },
        },
        "rf_depth_8": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 8
            },
        },
        "rf_depth_7": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 7
            },
        },

        "rf_depth_6": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 6
            },
        },
        "rf_depth_5": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 5
            },
        },
        "rf_depth_4": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 4,
                'n_jobs': -1
            },
        },
        "rf_depth_3": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 3
            },
        },
        "rf_depth_2": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 2
            },
        },
        "rf_depth_1": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 1
            },
        },
        "rf_depth_24": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 24
            },
        },
        "rf_depth_32": {
            'model_name': 'rf',
            'module name': 'sklearn.ensemble',
            'class name': 'RandomForestClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 32
            },
        },
        "dt": {
            'model_name': 'dt',
            'module name': 'sklearn.tree',
            'class name': 'DecisionTreeClassifier',
            'model_params': {
                'random_state': next_seed(),
            }
        },
        'et': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
            }
        },
        'et_depth_15': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 15
            }
        },
        'et_depth_16': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 16
            }
        },
        'et_depth_14': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 14
            }
        },
        'et_depth_13': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 13
            }
        },
        'et_depth_12': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 12
            }
        },
        'et_depth_11': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 11
            }
        },
        'et_depth_10': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 10
            }
        },
        'et_depth_9': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 9
            }
        },
        'et_depth_8': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 8
            }
        },
        'et_depth_7': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 7
            }
        },
        'et_depth_6': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 6
            }
        },
        'et_depth_5': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 5
            }
        },
        'et_depth_4': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 4
            }
        },
        'et_depth_3': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 3
            }
        },
        'et_depth_2': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 2
            }
        },
        'et_depth_1': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 1
            }
        },
        'et_depth_24': {
            'model_name': 'et',
            'module name': 'sklearn.ensemble',
            'class name': 'ExtraTreesClassifier',
            'model_params': {
                'random_state': next_seed(),
                'max_depth': 24
            }
        },
        'lr': {
            'model_name': 'lr',
            'module name': 'sklearn.linear_model',
            'class name': 'LogisticRegression',
            'model_params': {
                'random_state': next_seed()
            }
        },
        'nb': {
            'model_name': 'nb',
            'module name': 'sklearn.naive_bayes',
            'class name': 'GaussianNB',
            'model_params': {

            }
        }
    }
    return clf_params_dict


def get_metrics_helper(y_true, y_prob, threshold, threshold_name, logger):
    """
    split out to avoid duplicated code for computing
    metrics for train and test sets
    :param y_true: ground truth values for dependent variable
    :param y_prob: model output probabilities, assuming this is the slice of output
    of predict_proba or something similar, for the positive class
    return: dictionary of classification results
    """
    # results from current threshold
    tn, fp, tp, fn, tpr, tnr, fpr, fnr, mcc, auc, auprc = get_performance(y_true, y_prob,
                                                                          threshold)
    # found f-measure calcuated in notebook, wanted to compare with sklearn
    precision = tp/(tp + fp)


    f_measure = (2*precision * tpr)/(precision + tpr)

    y_pred = np.where(
        y_prob > threshold, np.ones_like(y_prob), np.zeros_like(y_prob)
    )
    dbg_df = pd.DataFrame({'y_pred': y_pred})

    f_measure_lib = f1_score(y_true, y_pred)


    if abs(f_measure - f_measure_lib) >= 0.001:
        logger.error(
            f'f measure scores do not match. f_measure = {f_measure}, f_measure_lib = {f_measure_lib}')

    g_mean = (tpr*tnr)**0.5
    logger.debug(f'g_mean: {g_mean}')

    result = {'AUC': auc,
              'AUPRC': auprc,
              'TP': tp,
              'FP': fp,
              'TN': tn,
              'FN': fn,
              'TPR': tpr,
              'TNR': tnr,
              'FPR': fpr,
              'FNR': fnr,
              'f_measure': f_measure_lib,
              'g_mean': g_mean,
              'mcc': mcc,
              'precision': precision,
              f'{threshold_name}_threshold': threshold
              }

    # flatten result for output to csv
    # we will get a g_mean_f_measure
    # which is the f_measure score for the threshold applied
    # for optimized geometric mean, etc.
    return_val = {f'{threshold_name}_{metric_name}': metric_val
                  for metric_name, metric_val in result.items()}
    return return_val


def get_metrics(y_test, y_prob, y_train, y_train_prob, dataset_name, logger):
    """
    common code for getting popular met
    :param y_test: ground truth
    :param y_prob: result of predict_proba function, only need positive class probability, i.e. should be arr[:,1] where arr is classifier output
    :result: dictionary of metrics
    """
    for arr, name in zip([y_test, y_prob, y_train, y_train_prob],
                         ['y_test', 'y_prob', 'y_train', 'y_train_prob']):
        logger.debug(f'len {name}: {len(arr)}')
        dbg_df = pd.DataFrame({name: arr})
        logger.debug(f'''{name} descriptive statistics:
        {dbg_df.describe()}''')

    f_measure_threshold = get_best_threshold_by_fmeasure(y_train, y_train_prob)

    f_measure_threshold_no_constraint = get_best_threshold_by_fmeasure(y_train, y_train_prob,
                                                                       False)
    g_mean_threshold = get_best_threshold_by_gmean(y_train, y_train_prob)

    g_mean_threshold_no_constraint = get_best_threshold_by_gmean(
        y_train, y_train_prob, False)

    mcc_threshold = get_best_threshold_by_mcc(y_train, y_train_prob)

    mcc_threshold_no_constraint = get_best_threshold_by_mcc(
        y_train, y_train_prob, False)

    precision_threshold = get_best_threshold_by_precision(
        y_train, y_train_prob)

    precision_threshold_no_constraint = get_best_threshold_by_precision(
        y_train, y_train_prob, False)

    prior_threshold = (y_train == 1).sum() / len(y_train)


    metrics = {'train': {}, 'test': {}}
    for threshold, threshold_name in zip([f_measure_threshold,
                                          g_mean_threshold, mcc_threshold, prior_threshold,
                                          precision_threshold, 0.5,
                                          f_measure_threshold_no_constraint,
                                          g_mean_threshold_no_constraint,
                                          mcc_threshold_no_constraint,
                                          precision_threshold_no_constraint], threshold_names):
        metrics['test'].update(get_metrics_helper(y_test, y_prob, threshold,
                                                  threshold_name, logger))
        metrics['train'].update(get_metrics_helper(y_train, y_train_prob, threshold,
                                                  threshold_name, logger))

    return metrics


def save_metrics(dataset_name, classifier_name, sampling, met, initial_seed, result_dir, logger):
    """
    save met to csv file 
    :param dataset_name: used to make result file name
    :param met: met from get_metrics
    :result_dir: directory where results are saved
    """
    out_file_name = f'{result_dir}/{dataset_name}_{initial_seed}_{classifier_name}{"_rus_" if sampling else ""}{sampling if sampling else ""}.csv'
    logger.debug(f'out_file_name: {out_file_name}')
    if not os.path.exists(out_file_name):
        # if file does not exist, create and add header
        f = open(out_file_name, 'w')
        header = ','.join(list(met.keys())) + '\n'
        f.write(header)
        logger.debug(f'save metrics, header: {header}')
    else:
        f = open(out_file_name, 'a')
    new_row = ','.join(list(map(str, met.values()))) + '\n'
    f.write(new_row)
    logger.debug(f'save metrics, new_row: {new_row}')
    pass


def cb_encode(X_train, y_train, X_test, cat_features, logger):
    """
    fit the encoder with the training data only, then encode
    the train and test data
    :param X_test: test data set
    :param X_train: training data
    :param y_train: labels, for CatBoost encoding
    :param cat_features: categorical features to be encoded
    """
    random_state = next_seed()
    logger.debug(
        f'fitting encoder on training data  with CatBoostEncoder seed = {random_state}')
    cb_encoder = CatBoostEncoder(cols=cat_features, verbose=0)
    cb_encoder.fit(X_train, y_train,
                   random_state=random_state)
    logger.debug('encoding training data')
    X_train = cb_encoder.transform(X_train)
    gc.collect()
    logger.debug('encoding test data')
    X_test = cb_encoder.transform(X_test)
    gc.collect()
    return X_train, X_test


def scale(X_train, X_test, scale_columns, logger):
    """
    scale training and test data in scale_columns to -1 1
    :param X_train: training data
    :param X_test: test data
    :param scale_columns: names of columns to scale
    :logger: for logging, Todo: we shouldn't be passing logger everywhere its
    a nuissance
    scaling technique from https://stackoverflow.com/a/61033063
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    logger.debug(f'fitting scaler for columns {scale_columns}')
    # ToDo use https://stackoverflow.com/a/56971065
    # to get rid of warning about copying slices
    scaler.fit(X_train[scale_columns])
    logger.debug('scaling training data')
    X_train[scale_columns] = scaler.transform(X_train[scale_columns])
    logger.debug('scaling test data')
    X_test[scale_columns] = scaler.transform(X_test[scale_columns])
    gc.collect()
    return X_train, X_test


def save_predictions(y_prob, args, logger):
    '''save predictions to disk for debugging
    just save last copy, don't want to generate
    too much data'''
    predictions_dir = 'predictions'
    logger.debug(f'saving y_prob to disk to {predictions_dir} ')
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    pd.DataFrame({'y_prob': y_prob}).to_csv(
        f'{predictions_dir}/{args.dataset_name}-{args.classifier_name}-{args.sampling}.csv', index=False)

def get_npi_column(dataset_name):
    '''
    get name of column with national provider id (NPI)
    it is different for every dataset

    :param dataset_name: name of medicare dataset will have partb,
     partd or dmepos in it
    '''
    lower_ds_name = dataset_name.lower()
    if 'dmepos' in lower_ds_name:
        return 'Rfrg_NPI'
    elif 'part-b' in lower_ds_name:
        return 'Rndrng_NPI'
    elif 'part-d' in lower_ds_name or 'part_d' in lower_ds_name:
        return 'Prscrbr_NPI'
    raise ValueError('attempt to get NPI column name from dataset name we do not recognize')

def k_fold_by_npi(x, y, k, npi_col, logger):
    logger.info(f"k_fold_by_npi: using npi_col {npi_col}")
    unique_npis = x[npi_col].unique()

    shuffled_unique_npis = np.random.choice(
        x[npi_col].unique(), size=len(unique_npis), replace=False
    )

    kf = KFold(n_splits=k, shuffle=False)

    for train_idx, test_idx in kf.split(shuffled_unique_npis):

        x_train = x.loc[x[npi_col].isin(shuffled_unique_npis[train_idx])].drop(
            columns=[npi_col]
        )
        x_test = x.loc[x[npi_col].isin(shuffled_unique_npis[test_idx])].drop(
            columns=[npi_col]
        )

        y_train = y[x_train.index]
        y_test = y[x_test.index]

        yield x_train, y_train, x_test, y_test

if __name__ == "__main__":
    logger = get_logger(logger_name='main', level=logging.DEBUG)
    logger.debug(f'starting up on {socket.gethostname()}')
    result_dir = f'{os.environ["HOME"]}/git/CMS-2019-Preprocessing/jh-exps/results'
    train_result_dir = f'{os.environ["HOME"]}/git/CMS-2019-Preprocessing/jh-exps/train_results'
    for dir_name in [result_dir, train_result_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # for LightGB and possibly CatBoost later
    # this flag indicates we do not need
    # to use CatBoost encoding because
    # we will use built-in encoding
    using_built_in_encoding = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'initial_seed', help='initial random number generarator seed', type=int)
    parser.add_argument('-d', '--dataset-name',
                        dest='dataset_name', help='dataset name')
    parser.add_argument('-c', '--classifier-name',
                        dest='classifier_name', help='classifier name')
    parser.add_argument('-s', '--sampling', nargs='?', const=1.0,
                        type=float, help='use random undersampling to 1-1 ratio')

    parser.add_argument(
        '-e', '--scale', help='list of columns to scale from -1 to 1; set to empty string for all columns ', nargs='+')

    parser.add_argument('-v', '--save', dest='save_predictions', action='store_true',
                        help='save test data output probablilties for debugging thresholding')

    args = parser.parse_args()
    logger.debug(f'args: {args}')

    file_name = datasets_dict[args.dataset_name]['file_name']
    #file_name = datasets_dict[args.dataset_name]['sample_file']
    logger.debug(f'reading {file_name}')
    df = pd.read_csv(file_name)

    # handle categorical features for lightgbm
    cat_features = datasets_dict[args.dataset_name]['cat_features']
    if 'lightgbm' in args.classifier_name:
        logger.debug(
            'for lightgbm: setting categorical features as type category')
        for c in cat_features:
            df[c] = df[c].astype('category')
        # df is not really encoded at this point,
        # only prepared for lightgbm to encode it when it calls its fit() function
        using_built_in_encoding = True

    initial_seed = next_seed(args.initial_seed)

    gc.collect()
    # do 10 iterations of 5-fold cross validation
    npi_col = get_npi_column(args.dataset_name)
    for i in range(10):
        logger.debug('computing test/train split')
        X = df[datasets_dict[args.dataset_name]['features'] + [npi_col]]
        y = df[datasets_dict[args.dataset_name]['target']]
        seed = next_seed()
        logger.debug(f'seed currently: {seed}')
        for X_train, y_train, X_test, y_test in k_fold_by_npi(X, y, 5, npi_col, logger):

            if not using_built_in_encoding and cat_features:
                X_train, X_test = cb_encode(
                    X_train, y_train, X_test, cat_features, logger)
            else:
                assert len(cat_features) == 0, 'cat features should be empty'
                logger.debug('cat_features list is empty, not encoding')

            if args.scale:
                X_train, X_test = scale(X_train, X_test,
                                        X_train.columns if args.scale == [''] else args.scale,
                                        logger)

            # get instance of classifier
            clf_params_dict = get_clf_params_dict()
            mod_name = clf_params_dict[args.classifier_name]['module name']
            module = importlib.import_module(mod_name)
            clf_params_dict = get_clf_params_dict()
            class_name = clf_params_dict[args.classifier_name]['class name']
            clf_class = getattr(module, class_name)
            clf = clf_class()
            mod_params = clf_params_dict[args.classifier_name]['model_params']
            clf.set_params(**mod_params)
            if 'random_state' in mod_params:
                logger.debug(
                    f'initialized classifer {clf}, seed = {mod_params["random_state"] }')
            else:
                logger.debug(
                    f'initialized classifer {clf}')

            # apply sampling if specified on command line
            # according to "Gradient Boosted Decision Tree Algorithms for Medicare Fraud Detection"
            # 1:1 ratio yields the best performance
            if args.sampling:
                logger.debug('applying random undersampling')
                rus = RandomUnderSampler(
                    random_state=next_seed(), sampling_strategy=args.sampling)
                X_train, y_train = rus.fit_resample(X_train, y_train)
                gc.collect()
            # train and test
            logger.debug('fit model')
            clf.fit(X_train, y_train)
            logger.debug('start computing met')
            y_prob = clf.predict_proba(X_test)

            y_train_prob = clf.predict_proba(X_train)
            if args.save_predictions:
                save_predictions(y_train_prob[:, 1], args, logger)

            met = get_metrics(y_test, y_prob[:, 1], y_train, y_train_prob[:, 1], args.dataset_name,
                              logger)
            save_metrics(args.dataset_name, args.classifier_name, args.sampling,
                         met['test'], args.initial_seed, result_dir, logger)
            save_metrics(args.dataset_name, args.classifier_name, args.sampling,
                         met['train'], args.initial_seed, train_result_dir, logger)
            gc.collect()
