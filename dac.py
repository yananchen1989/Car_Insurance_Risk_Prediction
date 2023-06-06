import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import glob,argparse
import scipy,datetime
import numpy as np 
print(xgb.__version__)

# parser = argparse.ArgumentParser()
# parser.add_argument("--dummy", action="store_true")
# parser.add_argument("--sample_cnt", default=10000000, type=int)
# parser.add_argument("--max_cat_to_onehot", default=10, type=int)
# parser.add_argument("--max_cat_threshold", default=1000, type=int)
# parser.add_argument("--tree_method", default="exact", type=str, choices=['exact', 'approx', 'hist'])
# args = parser.parse_args()

# col_names_train = ['label'] + ['I' + str(i) for i in range(1, 14)] + ['C' + str(i) for i in range(1, 27)]
# col_names_test = col_names_train[1:]

# train_df = pd.read_csv('./dac/train.txt',sep='\t', names=col_names_train).sample(args.sample_cnt)

from pyspark.sql.types import *
schema =  StructType([StructField("label", IntegerType(), nullable = False)] + \
          [StructField('I' + str(i), FloatType(), nullable = True) for i in range(1, 14)] + \
          [StructField('C' + str(i), StringType(), nullable = True)  for i in range(1, 27)]
          ) 


train_df_spark = spark.read.option("delimiter", "\t")\
            .csv("gs://definity-aw-prod-dswork-pidc/temp/dac/train.txt", header=False, schema=schema)

train_df = train_df_spark.sample(0.1).toPandas()
print(train_df.info())

'''
cat_features_s = []
for col in train_df.columns:
    if col.startswith("C"):
        print(col, train_df[col].unique().shape)

        if train_df[col].unique().shape[0] <= 500:
            cat_features_s.append(col)

C1 (1277,)
C2 (551,)
C3 (362997,)
C4 (141699,)
C5 (275,)
C6 (15,)
C7 (11200,)
C8 (565,)
C9 (3,)
C10 (31775,)
C11 (4887,)
C12 (323222,)
C13 (3146,)
C14 (26,)
C15 (9475,)
C16 (246817,)
C17 (10,)
C18 (4077,)
C19 (1849,)
C20 (4,)
C21 (291243,)
C22 (14,)
C23 (14,)
C24 (45052,)
C25 (70,)
C26 (32546,)
'''
# for col in train_df.columns:
#     print(col, train_df.loc[train_df[col].isnull()].shape[0] / train_df.shape[0])

#     if train_df.loc[train_df[col].isnull()].shape[0] > 0:
#         print(train_df.loc[train_df[col].isnull()][col].unique())


# for col, tp in zip(train_df.columns, train_df.dtypes.tolist()):
#     if tp == 'object' or tp == 'bool': 
#         print(col, train_df[col].unique().shape)

for col, tp in zip(train_df.columns, train_df.dtypes.tolist()):
    if tp == 'object' or tp == 'bool': 
        train_df[col] = train_df[col].astype("category")

'''
dummy_cols = []
for col, tp in zip(train_df.columns, train_df.dtypes.tolist()):
    if tp == 'object' or tp == 'bool': 
        if args.dummy:  
            if train_df[col].unique().shape[0]<=300:
                dummy_cols.append(col)
            else:
                group_dic = dict(train_df[col].value_counts(dropna=False))
                train_df[col] = train_df[col].map(lambda x: group_dic[x])
        else:
            train_df[col] = train_df[col].astype("category")

if args.dummy:
    train_df = pd.get_dummies(train_df, columns = dummy_cols)
    print("dummy_cols===>", dummy_cols)
'''



# df_raw = pd.read_csv("./dac/dac_df_assemble_columns_10000000m_raw/part-00000-b5b0b581-3fb0-43c2-be97-f45f8ee96468-c000.csv")

dall = xgb.DMatrix(train_df.drop(['label'], axis=1, errors='ignore'), train_df["label"], enable_categorical=True)

for tree_method in ['exact', 'approx', 'hist']:
    for max_cat_to_onehot in [10, 20, 50, 100, 200]:
        for max_cat_threshold in [100, 200, 500, 1000]:
            param = {"tree_method": tree_method, 
                    "max_cat_to_onehot": max_cat_to_onehot, 
                    "max_cat_threshold": max_cat_threshold, 
                    'max_depth':7, 'eta': 0.12,
                      'subsample': 0.7, 'min_child_weight':0.2, 'colsample_bytree':0.7,
                      'alpha':0,'lambda':0, 'max_bin':256,'disable_default_eval_metric':1,
                      'eval_metric':'auc', 'objective': "binary:logistic" } # 'tweedie_variance_power':1.5

            result = xgb.cv(param, dall, num_boost_round = 2000, nfold=5, early_stopping_rounds=100, verbose_eval=50, maximize=True)

            # print(result)
            print(tree_method, max_cat_to_onehot, max_cat_threshold, \
                    result['train-auc-mean'].max(), result['test-auc-mean'].max())


# param['scale_pos_weight'] = df["label"].value_counts()[0] / df["label"].value_counts()[1]
# if args.reg_obj == "pseudohubererror":
#     param['huber_slope'] = args.huber_slope