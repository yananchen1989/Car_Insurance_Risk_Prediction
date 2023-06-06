import datetime,math,argparse,os
import dateutil.relativedelta
import numpy as np
import pandas as pd 
import glob,argparse,scipy
import xgboost as xgb
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None 
assert xgb.__version__ == "1.7.5"

parser = argparse.ArgumentParser()
parser.add_argument("--train_off_dt", default= "2020-12-31", type=str)
parser.add_argument("--prov", default= "ON", type=str) # ON,AB,QC, NB,NS,PE
# parser.add_argument("--test_month", default=12, type=int)
# parser.add_argument("--label_apply", default= "loss_mean", type=str)
# parser.add_argument("--ver", default="ver3", type=str)
parser.add_argument("--split", default= "cv", type=str)
# parser.add_argument("--train_dt_trunc", default= "2017-01-01", type=str)
parser.add_argument("--use_stats", default= 0, type=int)
parser.add_argument("--frac", default= 1, type=float)
args = parser.parse_args()
print("summary args====>", args)

CVGS = ["DC", "COMPSP", "APCOLL", "TPL",  "AB"]

loss_cols = ["{}_loss".format(cvg) for cvg in CVGS] 
loss_cols_infer = []
for i in [1,3,6]:
    for cvg in CVGS:
        loss_cols_infer.append("{}_loss_{}".format(cvg, i))

event_keys = ["policy_num", "vin", "dln", "ver_eff"]


if not args.use_stats:
    stats_cols = ['term_prem_vin_1', 'term_prem_mean_vin_1', 'ver_cnt_vin_1', 'ver_days_min_vin_1', 'ver_days_max_vin_1', 'term_days_mean_vin_1', 'cntd_dln_vin_1', 'cntd_policy_num_vin_1', 'term_loss_mean_DC_vin_1', 'term_loss_DC_vin_1', 'term_loss_mean_COMPSP_vin_1', 'term_loss_COMPSP_vin_1', 'term_loss_mean_APCOLL_vin_1', 'term_loss_APCOLL_vin_1', 'term_loss_mean_TPL_vin_1', 'term_loss_TPL_vin_1', 'term_loss_mean_UIN_vin_1', 'term_loss_UIN_vin_1', 'term_loss_mean_AB_vin_1', 'term_loss_AB_vin_1', 'term_dayssince_vin_1', 'term_prem_vin_2', 'term_prem_mean_vin_2', 'ver_cnt_vin_2', 'ver_days_min_vin_2', 'ver_days_max_vin_2', 'term_days_mean_vin_2', 'cntd_dln_vin_2', 'cntd_policy_num_vin_2', 'term_loss_mean_DC_vin_2', 'term_loss_DC_vin_2', 'term_loss_mean_COMPSP_vin_2', 'term_loss_COMPSP_vin_2', 'term_loss_mean_APCOLL_vin_2', 'term_loss_APCOLL_vin_2', 'term_loss_mean_TPL_vin_2', 'term_loss_TPL_vin_2', 'term_loss_mean_UIN_vin_2', 'term_loss_UIN_vin_2', 'term_loss_mean_AB_vin_2', 'term_loss_AB_vin_2', 'term_dayssince_vin_2', 'term_prem_vin_3', 'term_prem_mean_vin_3', 'ver_cnt_vin_3', 'ver_days_min_vin_3', 'ver_days_max_vin_3', 'term_days_mean_vin_3', 'cntd_dln_vin_3', 'cntd_policy_num_vin_3', 'term_loss_mean_DC_vin_3', 'term_loss_DC_vin_3', 'term_loss_mean_COMPSP_vin_3', 'term_loss_COMPSP_vin_3', 'term_loss_mean_APCOLL_vin_3', 'term_loss_APCOLL_vin_3', 'term_loss_mean_TPL_vin_3', 'term_loss_TPL_vin_3', 'term_loss_mean_UIN_vin_3', 'term_loss_UIN_vin_3', 'term_loss_mean_AB_vin_3', 'term_loss_AB_vin_3', 'term_dayssince_vin_3', 'term_prem_vin_4', 'term_prem_mean_vin_4', 'ver_cnt_vin_4', 'ver_days_min_vin_4', 'ver_days_max_vin_4', 'term_days_mean_vin_4', 'cntd_dln_vin_4', 'cntd_policy_num_vin_4', 'term_loss_mean_DC_vin_4', 'term_loss_DC_vin_4', 'term_loss_mean_COMPSP_vin_4', 'term_loss_COMPSP_vin_4', 'term_loss_mean_APCOLL_vin_4', 'term_loss_APCOLL_vin_4', 'term_loss_mean_TPL_vin_4', 'term_loss_TPL_vin_4', 'term_loss_mean_UIN_vin_4', 'term_loss_UIN_vin_4', 'term_loss_mean_AB_vin_4', 'term_loss_AB_vin_4', 'term_dayssince_vin_4', 'term_prem_dln_1', 'term_prem_mean_dln_1', 'ver_cnt_dln_1', 'ver_days_min_dln_1', 'ver_days_max_dln_1', 'term_days_mean_dln_1', 'cntd_vin_dln_1', 'cntd_policy_num_dln_1', 'term_loss_mean_DC_dln_1', 'term_loss_DC_dln_1', 'term_loss_mean_COMPSP_dln_1', 'term_loss_COMPSP_dln_1', 'term_loss_mean_APCOLL_dln_1', 'term_loss_APCOLL_dln_1', 'term_loss_mean_TPL_dln_1', 'term_loss_TPL_dln_1', 'term_loss_mean_UIN_dln_1', 'term_loss_UIN_dln_1', 'term_loss_mean_AB_dln_1', 'term_loss_AB_dln_1', 'term_dayssince_dln_1', 'term_prem_dln_2', 'term_prem_mean_dln_2', 'ver_cnt_dln_2', 'ver_days_min_dln_2', 'ver_days_max_dln_2', 'term_days_mean_dln_2', 'cntd_vin_dln_2', 'cntd_policy_num_dln_2', 'term_loss_mean_DC_dln_2', 'term_loss_DC_dln_2', 'term_loss_mean_COMPSP_dln_2', 'term_loss_COMPSP_dln_2', 'term_loss_mean_APCOLL_dln_2', 'term_loss_APCOLL_dln_2', 'term_loss_mean_TPL_dln_2', 'term_loss_TPL_dln_2', 'term_loss_mean_UIN_dln_2', 'term_loss_UIN_dln_2', 'term_loss_mean_AB_dln_2', 'term_loss_AB_dln_2', 'term_dayssince_dln_2', 'term_prem_dln_3', 'term_prem_mean_dln_3', 'ver_cnt_dln_3', 'ver_days_min_dln_3', 'ver_days_max_dln_3', 'term_days_mean_dln_3', 'cntd_vin_dln_3', 'cntd_policy_num_dln_3', 'term_loss_mean_DC_dln_3', 'term_loss_DC_dln_3', 'term_loss_mean_COMPSP_dln_3', 'term_loss_COMPSP_dln_3', 'term_loss_mean_APCOLL_dln_3', 'term_loss_APCOLL_dln_3', 'term_loss_mean_TPL_dln_3', 'term_loss_TPL_dln_3', 'term_loss_mean_UIN_dln_3', 'term_loss_UIN_dln_3', 'term_loss_mean_AB_dln_3', 'term_loss_AB_dln_3', 'term_dayssince_dln_3', 'term_prem_dln_4', 'term_prem_mean_dln_4', 'ver_cnt_dln_4', 'ver_days_min_dln_4', 'ver_days_max_dln_4', 'term_days_mean_dln_4', 'cntd_vin_dln_4', 'cntd_policy_num_dln_4', 'term_loss_mean_DC_dln_4', 'term_loss_DC_dln_4', 'term_loss_mean_COMPSP_dln_4', 'term_loss_COMPSP_dln_4', 'term_loss_mean_APCOLL_dln_4', 'term_loss_APCOLL_dln_4', 'term_loss_mean_TPL_dln_4', 'term_loss_TPL_dln_4', 'term_loss_mean_UIN_dln_4', 'term_loss_UIN_dln_4', 'term_loss_mean_AB_dln_4', 'term_loss_AB_dln_4', 'term_dayssince_dln_4', 'term_prem_policy_num_1', 'term_prem_mean_policy_num_1', 'ver_cnt_policy_num_1', 'ver_days_min_policy_num_1', 'ver_days_max_policy_num_1', 'term_days_mean_policy_num_1', 'cntd_vin_policy_num_1', 'cntd_dln_policy_num_1', 'term_loss_mean_DC_policy_num_1', 'term_loss_DC_policy_num_1', 'term_loss_mean_COMPSP_policy_num_1', 'term_loss_COMPSP_policy_num_1', 'term_loss_mean_APCOLL_policy_num_1', 'term_loss_APCOLL_policy_num_1', 'term_loss_mean_TPL_policy_num_1', 'term_loss_TPL_policy_num_1', 'term_loss_mean_UIN_policy_num_1', 'term_loss_UIN_policy_num_1', 'term_loss_mean_AB_policy_num_1', 'term_loss_AB_policy_num_1', 'term_dayssince_policy_num_1', 'term_prem_policy_num_2', 'term_prem_mean_policy_num_2', 'ver_cnt_policy_num_2', 'ver_days_min_policy_num_2', 'ver_days_max_policy_num_2', 'term_days_mean_policy_num_2', 'cntd_vin_policy_num_2', 'cntd_dln_policy_num_2', 'term_loss_mean_DC_policy_num_2', 'term_loss_DC_policy_num_2', 'term_loss_mean_COMPSP_policy_num_2', 'term_loss_COMPSP_policy_num_2', 'term_loss_mean_APCOLL_policy_num_2', 'term_loss_APCOLL_policy_num_2', 'term_loss_mean_TPL_policy_num_2', 'term_loss_TPL_policy_num_2', 'term_loss_mean_UIN_policy_num_2', 'term_loss_UIN_policy_num_2', 'term_loss_mean_AB_policy_num_2', 'term_loss_AB_policy_num_2', 'term_dayssince_policy_num_2', 'term_prem_policy_num_3', 'term_prem_mean_policy_num_3', 'ver_cnt_policy_num_3', 'ver_days_min_policy_num_3', 'ver_days_max_policy_num_3', 'term_days_mean_policy_num_3', 'cntd_vin_policy_num_3', 'cntd_dln_policy_num_3', 'term_loss_mean_DC_policy_num_3', 'term_loss_DC_policy_num_3', 'term_loss_mean_COMPSP_policy_num_3', 'term_loss_COMPSP_policy_num_3', 'term_loss_mean_APCOLL_policy_num_3', 'term_loss_APCOLL_policy_num_3', 'term_loss_mean_TPL_policy_num_3', 'term_loss_TPL_policy_num_3', 'term_loss_mean_UIN_policy_num_3', 'term_loss_UIN_policy_num_3', 'term_loss_mean_AB_policy_num_3', 'term_loss_AB_policy_num_3', 'term_dayssince_policy_num_3', 'term_prem_policy_num_4', 'term_prem_mean_policy_num_4', 'ver_cnt_policy_num_4', 'ver_days_min_policy_num_4', 'ver_days_max_policy_num_4', 'term_days_mean_policy_num_4', 'cntd_vin_policy_num_4', 'cntd_dln_policy_num_4', 'term_loss_mean_DC_policy_num_4', 'term_loss_DC_policy_num_4', 'term_loss_mean_COMPSP_policy_num_4', 'term_loss_COMPSP_policy_num_4', 'term_loss_mean_APCOLL_policy_num_4', 'term_loss_APCOLL_policy_num_4', 'term_loss_mean_TPL_policy_num_4', 'term_loss_TPL_policy_num_4', 'term_loss_mean_UIN_policy_num_4', 'term_loss_UIN_policy_num_4', 'term_loss_mean_AB_policy_num_4', 'term_loss_AB_policy_num_4', 'term_dayssince_policy_num_4']
else:
    stats_cols = []


gs_path = "~" #"gs://definity-aw-prod-dswork-pidc/temp/dynamic_time_frame"

df_raw = pd.read_parquet("{}/df_events".format(gs_path))
df = df_raw.loc[(df_raw['policy_province'].isin(args.prov.split(","))) \
              & (df_raw['ver_eff'] < pd.to_datetime(args.train_off_dt).date() )]\
            .drop(stats_cols, axis=1, errors='ignore')

print("ver_eff===>", df['ver_eff'].min(), df['ver_eff'].max())

for col, tp in zip(df.columns, df.dtypes.tolist()):
    print(col, tp)

df_inforce_infer = pd.read_parquet("{}/df_inforce_infer_{}".format(gs_path, args.train_off_dt))
df_infer = df_inforce_infer.loc[df_inforce_infer['policy_province'].isin(args.prov.split(","))]

# train_off_dt = args.train_off_dt
# test_on_dt = (datetime.datetime.strptime(train_off_dt, "%Y-%m-%d") \
#                     + dateutil.relativedelta.relativedelta(days=1)).strftime('%Y-%m-%d')
# test_off_dt = (datetime.datetime.strptime(test_on_dt, "%Y-%m-%d") \
#                     + dateutil.relativedelta.relativedelta(months=args.test_month) \
#                     - dateutil.relativedelta.relativedelta(days=1)).strftime('%Y-%m-%d')
# print("summary ==>", train_off_dt, test_on_dt, test_off_dt)


def process(df, event_keys):
    for ii, tp in zip(df.columns, df.dtypes.tolist()):
        if ii in event_keys:
            continue
        if tp in ("object", "bool"):
            df[ii] = df[ii].astype("category")
    return df 

def map_label(df, cvg):
    if cvg == 'ALL':
        df['label'] = df[loss_cols].sum(axis=1).map(lambda x: math.log(1 + x))
    else:
        df['label'] = df["{}_loss".format(cvg)].map(lambda x: math.log(1 + x))
    return df.drop(loss_cols, axis=1, errors='ignore')


def spearmanr(predt: np.ndarray, dtrain: xgb.DMatrix):
    return 'spearmanr', float(scipy.stats.spearmanr(predt, dtrain.get_label()).correlation)


# objective = ("reg:tweedie", "reg:pseudohubererror", "reg:squarederror", "reg:gamma")
# tree_method : 'exact', 'approx', 'hist'
param = {"tree_method": "exact",
    'max_depth':7, 'eta': 0.1, 
      'subsample': 0.7, 'min_child_weight':0.2, 'colsample_bytree':0.7,
      'alpha':0,'lambda':0, 'max_bin':256,'disable_default_eval_metric':False, 
      'max_cat_to_onehot': 10, 'max_cat_threshold': 1000,
      'eval_metric': 'rmse', 'objective':'reg:tweedie' }

df_process = process(df, event_keys)
df_infer  = process(df_infer, event_keys)

rm_cols = ["label", 'preds'] + event_keys + ['policy_province'] + loss_cols_infer


cvg_bst_dic = {}
for target_cvg in CVGS:
    df_training_valid = map_label(df_process, target_cvg)

    # init_top_ratio, init_bot_ratio = 0.3, 0.3
    # bst_best = None
    # for ite in range(10):
    if args.split == 'cv':
        df_train, df_val = train_test_split(df_training_valid, test_size=0.1)
    elif args.split == 'tt':
        df_training_valid_sorted = df_training_valid.sort_values("ver_eff")
        df_val = df_training_valid_sorted[int(df_training_valid_sorted.shape[0] * 0.9):]
        df_train = df_training_valid_sorted[:int(df_training_valid_sorted.shape[0] * 0.9)].sample(frac=1)

    X_train, y_train = df_train.drop(rm_cols, axis=1, errors='ignore'), df_train["label"]
    X_val, y_val = df_val.drop(rm_cols, axis=1, errors='ignore'), df_val["label"]

    dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)    

    es = xgb.callback.EarlyStopping(
            rounds=100,
            min_delta =0.0001,
            save_best=True,
            maximize=False,
            data_name="val",
            metric_name= 'rmse'
        )
    bst = xgb.train(param, dtrain, num_boost_round=1000, callbacks=[es],
                    custom_metric = spearmanr, 
                    evals=[(dtrain, 'train'),  (dval, 'val')],  verbose_eval=False) 
    
    cvg_bst_dic[target_cvg] = bst
    # auc_train, auc_val, auc_test = bst.eval(dtrain), bst.eval(dval), bst.eval(dtest)
    # train_metric = round(float(auc_train.split(":")[-1]), 4)
    # val_metric = round(float(auc_val.split(":")[-1]), 4)
    # test_metric = round(float(auc_test.split(":")[-1]), 4)


print('----------- ---------------')
for i in [1,3,6]:
    loss_cols_i = ["{}_loss_{}".format(cvg, i) for cvg in CVGS]
    df_infer['loss_cols_i'] = df_infer[loss_cols_i].sum(axis=1)
    X_infer, y_infer = df_infer[X_train.columns.tolist()], df_infer['loss_cols_i']
    dinfer = xgb.DMatrix(X_infer, y_infer, enable_categorical=True) 
    for cvg in CVGS:
        df_infer['preds_{}'.format(cvg)] = cvg_bst_dic[cvg].predict(dinfer) 
 
    df_infer['preds'] = df_infer[['preds_{}'.format(cvg) for cvg in CVGS]].sum(axis=1)

    for cvg in CVGS:
        del df_infer['preds_{}'.format(cvg)]

    df_infer['ratio'] = df_infer['loss_cols_i']  - df_infer['prem_{}'.format(i)] 
    for pred_col in ['preds', 'mpa_score']:
        corr_test_event = round(float(scipy.stats.spearmanr(df_infer[pred_col].values, df_infer['loss_cols_i'].values).correlation), 4)
        corr_test_event_ratio = round(float(scipy.stats.spearmanr(df_infer[pred_col].values, df_infer['ratio'].values).correlation), 4)
        
        percentile = df_infer[pred_col].quantile(0.95)

        df_top = df_infer.loc[df_infer[pred_col] >= percentile]
        df_bot = df_infer.loc[df_infer[pred_col] < percentile]

        top_ratio = df_top['loss_cols_i'].sum() / df_top['prem_{}'.format(i)].sum()
        bottom_ratio = df_bot['loss_cols_i'].sum() / df_bot['prem_{}'.format(i)].sum() 
        print(i, pred_col, top_ratio, bottom_ratio, corr_test_event, corr_test_event_ratio)
    del df_infer['preds'], df_infer['loss_cols_i'], df_infer['ratio']
    print()
print('--------------------------')

    # if i == 3 and top_ratio_infer > init_top_ratio and bottom_ratio_infer < init_bot_ratio: 
    #     # bst.save_model('bst.json')
    #     # print("model saved")
    #     init_top_ratio = top_ratio_infer
    #     init_bot_ratio = bottom_ratio_infer
    #     bst_best = bst

# print("summary===>", df_training_valid.shape[0], df_test.shape[0],
#       np.array(top_ratio_l).mean(), np.array(bottom_ratio_l).mean(), 
#       np.array(top_ratio_log_l).mean(), np.array(bottom_ratio_log_l).mean(),
#       np.array(corr_l).mean() )





# bst_best = xgb.Booster()
# bst_best.load_model("bst_.json")


'''
print("feature importance ====>")
feat_dic = {i:[] for i in bst_best.feature_names}

for metric in ("weight", "gain", "cover"):
    print(metric)
    feature_score = pd.DataFrame(bst_best.get_score(importance_type=metric).items(), columns=['feature','score']).sort_values(by=["score"], ascending=False)
    feature_score['rank_{}'.format(metric)] = feature_score['score'].rank(method='max') / feature_score.shape[0]
    
    # print(feature_score)
    for ix, row in feature_score.iterrows():
        feat_dic[row['feature']].append(row['rank_{}'.format(metric)])
    # print("\n\n")


dff = pd.DataFrame([(i, 0) if not j else (i, sum(j)/3 ) for i, j in feat_dic.items()], columns=['feature', 'score']).sort_values(by=["score"], ascending=False)

for ix, row in dff.iterrows():
    print(row['feature'], row['score'])

'''












# import uuid
# str(uuid.uuid4())


# sns.kdeplot(dfpd['label'])