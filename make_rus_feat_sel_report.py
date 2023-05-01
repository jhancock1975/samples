#!/usr/bin/env python3
# makes report for rus with feature selection
import os

import numpy as np
import pandas as pd
from rarity_reporting import get_clf, get_metric_name

results_dir = '/home/jthancoc/git/CMS-2019-Preprocessing/jh-exps/results'
file_dict = {
    'Part D': {
        '10': [f"{results_dir}/part-d-aggregated-new-sfs-top-10_1963_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-10_1964_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-10_1965_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-10_1966_lr_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-10_1967_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-10_1968_rf_depth_4_rus_0.0123.csv",
],
        '15': [f"{results_dir}/part-d-aggregated-new-sfs-top-15_1969_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-15_1970_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-15_1971_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-15_1972_lr_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-15_1973_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-15_1974_rf_depth_4_rus_0.0123.csv",
],
        '20': [f"{results_dir}/part-d-aggregated-new-sfs-top-20_1975_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-20_1976_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-20_1977_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-20_1978_lr_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-20_1979_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-20_1980_rf_depth_4_rus_0.0123.csv",
],
        '25': [f"{results_dir}/part-d-aggregated-new-sfs-top-25_1981_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-25_1982_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-25_1983_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-25_1984_lr_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-25_1985_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-new-sfs-top-25_1986_rf_depth_4_rus_0.0123.csv",
],
        '30': [f"{results_dir}/part-d-aggregated-sfs-new-top-30_1987_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-sfs-new-top-30_1988_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-sfs-new-top-30_1989_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-sfs-new-top-30_1990_lr_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-sfs-new-top-30_1991_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-d-aggregated-sfs-new-top-30_1992_rf_depth_4_rus_0.0123.csv",
],
        '82': [f"{results_dir}/part_d_aggregated_new_pre_encoded_1729_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part_d_aggregated_new_pre_encoded_1734_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part_d_aggregated_new_pre_encoded_1739_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part_d_aggregated_new_pre_encoded_1744_lr_rus_0.0123.csv",
f"{results_dir}/part_d_aggregated_new_pre_encoded_1749_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part_d_aggregated_new_pre_encoded_1754_rf_depth_4_rus_0.0123.csv",
]
    },
    'Part B': {
        '10': [f"{results_dir}/part-b-aggregated-new-top-10-sfs-6_1879_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-10-sfs-6_1880_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-10-sfs-6_1881_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-10-sfs-6_1882_lr_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-10-sfs-6_1883_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-10-sfs-6_1884_rf_depth_4_rus_0.0123.csv",
],
        '15': [f"{results_dir}/part-b-aggregated-new-top-15-sfs-6_1885_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-15-sfs-6_1886_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-15-sfs-6_1887_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-15-sfs-6_1888_lr_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-15-sfs-6_1889_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-15-sfs-6_1890_rf_depth_4_rus_0.0123.csv",
],
        '20': [f"{results_dir}/part-b-aggregated-new-top-20-sfs-6_1891_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-20-sfs-6_1892_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-20-sfs-6_1893_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-20-sfs-6_1894_lr_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-20-sfs-6_1895_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-20-sfs-6_1896_rf_depth_4_rus_0.0123.csv",
],
        '25': [f"{results_dir}/part-b-aggregated-new-top-25-sfs-6_1897_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-25-sfs-6_1898_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-25-sfs-6_1899_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-25-sfs-6_1900_lr_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-25-sfs-6_1901_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-25-sfs-6_1902_rf_depth_4_rus_0.0123.csv",
],
        '30': [f"{results_dir}/part-b-aggregated-new-top-30-sfs-6_1903_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-30-sfs-6_1904_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-30-sfs-6_1905_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-30-sfs-6_1906_lr_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-30-sfs-6_1907_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-top-30-sfs-6_1908_rf_depth_4_rus_0.0123.csv",
],
        '81': [f"{results_dir}/part-b-aggregated-new-pre_encoded_1759_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-pre_encoded_1764_et_depth_8_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-pre_encoded_1769_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-pre_encoded_1774_lr_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-pre_encoded_1779_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/part-b-aggregated-new-pre_encoded_1784_rf_depth_4_rus_0.0123.csv",
]
    },
    'DMEPOS': {
        '10': [f"{results_dir}/dmepos-aggregated-new-sfs-6-top-10_1909_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-10_1910_et_depth_8_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-10_1911_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-10_1912_lr_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-10_1913_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-10_1914_rf_depth_4_rus_0.0123.csv",
],
        '15': [f"{results_dir}/dmepos-aggregated-new-sfs-6-top-15_1915_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-15_1916_et_depth_8_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-15_1917_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-15_1918_lr_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-15_1919_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-15_1920_rf_depth_4_rus_0.0123.csv",
],
        '20': [f"{results_dir}/dmepos-aggregated-new-sfs-6-top-15_1920_rf_depth_4_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-20_1921_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-20_1922_et_depth_8_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-20_1923_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-20_1924_lr_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-20_1925_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-20_1926_rf_depth_4_rus_0.0123.csv",
],
        '25': [f"{results_dir}/dmepos-aggregated-new-sfs-6-top-20_1925_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-25_1927_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-25_1928_et_depth_8_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-25_1929_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-25_1930_lr_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-25_1931_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-25_1932_rf_depth_4_rus_0.0123.csv",
],
        '30': [f"{results_dir}/dmepos-aggregated-new-sfs-6-top-25_1930_lr_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-30_1933_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-30_1934_et_depth_8_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-30_1935_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-30_1936_lr_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-30_1937_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-sfs-6-top-30_1938_rf_depth_4_rus_0.0123.csv",
],
        '96': [f"{results_dir}/dmepos-aggregated-new-pre-encoded_1789_catboost_depth_5_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-pre-encoded_1794_et_depth_8_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-pre-encoded_1799_xgboost_depth_3_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-pre-encoded_1804_lr_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-pre-encoded_1809_lightgbm_depth_4_rus_0.0123.csv",
f"{results_dir}/dmepos-aggregated-new-pre-encoded_1814_rf_depth_4_rus_0.0123.csv",
]
    },
}

classifer_order = ['CatBoost', 'Random Forest', 'Logistic Regression', 'LightGBM', 'ET',
                   'XGBoost']
classifier_key = {value:index for index, value in enumerate(classifer_order)}

features_order = ['10', '15', '20', '25', '30', '80', '82', '96']
features_key = {value:index for index, value in enumerate(features_order)}


def make_tex_table(file_dict, ds_name, output_dir):
    for metric in ['g_mean_AUC', 'g_mean_AUPRC']:
        metric_name = get_metric_name(metric)
        table_dict = {'Classifier': [], metric_name: [], 'Features': []}
        for features, file_list in file_dict[ds_name].items():
            for file_name in file_list:
                df = pd.read_csv(file_name).tail(n=50)
                clf = get_clf(file_name)
                mean_metric = df[metric].mean()
                table_dict['Classifier'].append(clf)
                table_dict[metric_name].append(mean_metric)
                table_dict['Features'].append(features)
        table_df = pd.DataFrame(table_dict)
        sorted_indices = np.lexsort((table_df['Classifier'].map(
                classifier_key).values, table_df['Features'].map(
            features_key).values))
        table_df = table_df.iloc[sorted_indices]
        label = f'tab:{ds_name}-{metric_name}'
        caption = f'Mean {metric_name} values by classifier and count of \
features for ten iterations of five-fold cross validation, for classifying the \
Medicare {ds_name} data'
        table = table_df.to_latex(position='H', label=label, caption=caption,
                                  index=False, float_format='%.4f', escape=False)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f'{output_dir}/{ds_name.replace(" ", "-")}-{metric_name}.tex', 'w') as f:
            f.write(table)


def make_csv_file(file_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_arr = []
    for features, file_list in file_dict.items():
        for file_name in file_list:
            df = pd.read_csv(file_name).tail(n=50)
            df['Features'] = features
            df['Classifier'] = get_clf(file_name)
            df.rename(columns={"f_measure_AUC": "AUC",
                               "f_measure_AUPRC": "AUPRC"},
                      inplace=True)
            df_arr.append(df)
    pd.concat(df_arr).to_csv(f'{output_dir}/anova-file.csv', index=False)

if __name__ == '__main__':
    for ds_name, tables_dir, stats_dir in zip(
           ['Part D', 'Part B', 'DMEPOS'],
            ['part-d-rus-feat-sel-tables', 'part-b-rus-feat-sel-tables',
             'dmepos-rus-feat-sel-tables'],
            ['part-d-rus-feat-sel-stats', 'part-b-rus-feat-sel-stats',
             'dmepos-rus-feat-sel-stats']):
        make_tex_table(file_dict, ds_name, tables_dir)
        make_csv_file(file_dict[ds_name], stats_dir)
