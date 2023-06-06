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
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("--dv", default="vin", type=str, choices=['vin','dln'])
parser.add_argument("--target", default="cls", type=str, choices=['cls','reg'])
parser.add_argument("--train_dt", default="2021-01-01", type=str)
parser.add_argument("--test_dt", default="2022-01-01", type=str)
parser.add_argument("--tree_method", default="exact", type=str, choices=['exact', 'approx', 'hist'])
parser.add_argument("--scale_pos_weight", action="store_true")
parser.add_argument("--dummy", action="store_true")
parser.add_argument("--huber_slope", default=100, type=int)
parser.add_argument("--reg_target", default="claim", choices=['claim','earn','diff','ratio'], type=str)
parser.add_argument("--reg_obj", default="tweedie", choices=['tweedie','pseudohubererror'], type=str)
parser.add_argument("--cls_target", default='claim', choices=['claim','diff'], type=str)
parser.add_argument("--cls_thres", default=2000, type=int)
parser.add_argument("--cvtt", default="cv", choices=['cv','tt'], type=str)
parser.add_argument("--ban_history", action="store_true")
parser.add_argument("--month", default=12, type=int)


# parser.add_argument("--cvg", default="AB", type=str, choices=['AB', 'TPL', 'APCOLL', 'COMPSP', 'DC'])
args = parser.parse_args()
print("args====>", args)

def spearmanr(predt: np.ndarray, dtrain: xgb.DMatrix):
    return 'spearmanr', float(scipy.stats.spearmanr(predt, dtrain.get_label()).correlation)


# objective = ("reg:tweedie", "reg:pseudohubererror", "reg:squarederror", "reg:gamma")
param = {"tree_method": args.tree_method, #"max_cat_to_onehot": 5, "max_cat_threshold": 64, 
        'max_depth':7, 'eta': 0.2, 
          'subsample': 0.7, 'min_child_weight':0.2, 'colsample_bytree':0.7,
          'alpha':0,'lambda':0, 'max_bin':64,'disable_default_eval_metric':1 } # 'tweedie_variance_power':1.5

target_metric_dic = {'reg':'mae', 'cls':'auc'}
param['eval_metric'] = [target_metric_dic[args.target]]

if args.target == 'reg': 
    param['objective'] = "reg:{}".format(args.reg_obj)
    custom_metric = spearmanr
    if args.reg_obj == "pseudohubererror":
        param['huber_slope'] = args.huber_slope
elif args.target == 'cls':
    param['objective'] = "binary:logistic"
    if args.scale_pos_weight:
        param['scale_pos_weight'] = df["label"].value_counts()[0] / df["label"].value_counts()[1]
    custom_metric = None


CVGS = ("COMPSP", "AB", "TPL", "APCOLL", "DC", "COMBINE")


def cv(df, df_test=None, cvg=""):

    df_cvg = process_df(df.loc[~df["claimAmt_{}".format(cvg)].isnull()], cvg)

    if args.cvtt == 'tt':
        df_test_cvg = process_df(df_test.loc[~df_test["claimAmt_{}".format(cvg)].isnull()], cvg)
        assert df_cvg.columns.tolist() == df_test_cvg.columns.tolist()
    else:
        df_test_cvg = None


    drop_list = ["label", "id"] + \
                    ['claimAmt_{}'.format(ii) for ii in CVGS] + \
                     ['earningAmt_{}'.format(ii) for ii in CVGS] 

    if args.cvtt == 'cv':
        dall = xgb.DMatrix(df_cvg.drop(drop_list, axis=1, errors='ignore'), df_cvg["label"], enable_categorical=True)
        result = xgb.cv(param, dall, num_boost_round = 5000, nfold=5, early_stopping_rounds=100,
                        verbose_eval=10, custom_metric=custom_metric, maximize=True)
        if args.target == 'reg':
            print(args.dv, args.train_dt, args.test_dt, cvg, 
                    result['train-{}-mean'.format(target_metric_dic[args.target])].min(), \
                    result['test-{}-mean'.format(target_metric_dic[args.target])].min(), \
                    result['train-spearmanr'].max(), \
                    result['test-spearmanr'].max()
                    )
        elif args.target == 'cls':
            print(args.dv, args.train_dt, args.test_dt, cvg, 
                    df_cvg["label"].value_counts()[0], \
                    df_cvg["label"].value_counts()[1], \
                    result['train-{}-mean'.format(target_metric_dic[args.target])].max(), \
                    result['test-{}-mean'.format(target_metric_dic[args.target])].max())

    elif args.cvtt == 'tt':
        assert df_test_cvg is not None

        df_id_train, df_id_val = train_test_split(pd.DataFrame(df_cvg['id'].unique(), columns=['id']), test_size=0.1)

        df_train = pd.merge(df_cvg, df_id_train, on='id', how="inner")
        df_val = pd.merge(df_cvg, df_id_val, on='id', how="inner")

        X_train, y_train = df_train.drop(drop_list, axis=1, errors='ignore'), df_train["label"]
        X_val, y_val = df_val.drop(drop_list, axis=1, errors='ignore'), df_val["label"]
        X_test, y_test = df_test_cvg.drop(drop_list, axis=1, errors='ignore'), df_test_cvg["label"]

        dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)    
        dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

        es = xgb.callback.EarlyStopping(
                rounds=50,
                min_delta =1e-3,
                save_best=True,
                maximize=True,
                data_name="val",
                metric_name=param['eval_metric'][0],
            )

        bst = xgb.train(param, dtrain, num_boost_round=5000, early_stopping_rounds=50, callbacks=[es],
                            evals=[(dtrain, 'train'),  (dval, 'val')],  verbose_eval=10) 

        auc_train, auc_val, auc_test = bst.eval(dtrain), bst.eval(dval), bst.eval(dtest)
        train_metric = round(float(auc_train.split(":")[-1]), 4)
        val_metric = round(float(auc_val.split(":")[-1]), 4)
        test_metric = round(float(auc_test.split(":")[-1]), 4)
        print(args.dv, args.train_dt, args.test_dt, cvg, train_metric, val_metric, test_metric)
        # bst.save_model("test_model.json")

        df_test_cvg['preds_{}'.format(cvg)] = bst.predict(dtest)
        # df_test_cvg.to_csv("df_test_{}.csv".format(cvg) , index=False)

        df_test_cvg_sort = df_test_cvg.sort_values(by=['preds_{}'.format(cvg)], ascending=False)
        df_top = df_test_cvg_sort[:int(df_test_cvg_sort.shape[0]* 0.01)]
        df_town = df_test_cvg_sort[int(df_test_cvg_sort.shape[0]* 0.01):]
        print(cvg, 'summary ====>')

        print((df_top['claimAmt_{}'.format(cvg)] / df_top['earningAmt_{}'.format(cvg)]).mean(),
              (df_town['claimAmt_{}'.format(cvg)] / df_town['earningAmt_{}'.format(cvg)]).mean())

        print(df_top['claimAmt_{}'.format(cvg)].sum() / df_top['earningAmt_{}'.format(cvg)].sum(),
              df_town['claimAmt_{}'.format(cvg)].sum() / df_town['earningAmt_{}'.format(cvg)].sum() ) 

        print(df_top['claimAmt_{}'.format(cvg)].mean(), df_town['claimAmt_{}'.format(cvg)].mean())

def earn_claim_label(earningAmt, claimAmt):
    if args.target == 'reg':
        if args.reg_target == 'claim':
            if not claimAmt or np.isnan(claimAmt):
                return 0   
            else:
                return claimAmt  
        elif args.reg_target == 'earn':
            if not earningAmt or np.isnan(earningAmt):
                return 0   
            else:
                return earningAmt  
        elif args.reg_target == 'diff':
            if not earningAmt or np.isnan(earningAmt):
                earningAmt = 0 
            if not claimAmt or np.isnan(claimAmt):
                claimAmt = 0
            return max(min(claimAmt - earningAmt, 10000), 0)
        elif args.reg_target == 'ratio':
            return (claimAmt + 1) / (earningAmt + 1)


    elif args.target == 'cls':
        if not claimAmt or np.isnan(claimAmt):
            return 0  
        else:
            if args.cls_target == 'diff':
                if earningAmt >= claimAmt:
                    return 0 
                else:
                    return 1 
            elif args.cls_target == 'claim':
                if claimAmt >= args.cls_thres:
                    return 1 
                else:
                    return 0


def parse_model_vin(model):
    tokens = model.split()
    tokens_ = []
    wd, dr,shap, version = "-", "-", "-", "-"
    for t in tokens:
        if (t[0].isdigit() or t[0] =="A") and t[1:] == "WD":
            wd = t
        elif t[0].isdigit() and t[1:] == "DR":
            dr = t 
        elif t in ['WAGON','SEDAN','HATCHBACK', 'PICKUP', 'CAB','SUPERCAB']:
            shap = t
        else:
            tokens_.append(t)

    version = " ".join(tokens_)
    return  ",".join([wd, dr, shap, version])


def modify_df_model(df):
    df["model_"] = df['model'].map(lambda x: parse_model_vin(x))
    df["wd"] = df["model_"].map(lambda x: x.split(",")[0])
    df["dr"] = df["model_"].map(lambda x: x.split(",")[1])
    df["shap"] = df["model_"].map(lambda x: x.split(",")[2])
    df["version"] = df["model_"].map(lambda x: x.split(",")[3])
    del df["model_"], df["model"]
    return df 



''' 
from sentence_transformers import SentenceTransformer
transformer = SentenceTransformer('all-MiniLM-L6-v2')

from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
embeddings = transformer.encode(df["model"])
embeds_score = cosine_similarity(embeddings[0], embeddings[1])
'''



def process_df(df, cvg):
    # df['label'] = df['label'].clip(upper=2000)
    # if 'id' in df.columns:
    #     del df['id']

    # df['label'] = df.apply(lambda x: earn_claim_label(x.earningAmt, x.claimAmt), axis=1)
    if args.target == 'cls':
        df['label'] = df['claimAmt_{}'.format(cvg)].map(lambda x: 1 if x >= args.cls_thres else 0)
    elif args.target == 'reg':
        df['label'] = df['claimAmt_{}'.format(cvg)].fillna(0)

    ######### feature process ##################
    if  args.ban_history:
        del_claim_cols = [col for col in df.columns if col.startswith("claimAmt_history") or col.startswith("earningAmt_history")]
        assert len(del_claim_cols) > 0
    else:
        del_claim_cols = []

    df_ = df.drop(del_claim_cols + [ 'yr', 'update_time', 'vehicle_description'], axis=1, errors='ignore')
    
    for dt_col in ('date_purchased', 'effective_date', 'expiration_date', 'iso_transfer_date','rsp_transfer_date'):
        if dt_col in df_.columns:
            df_[dt_col + '_diff'] = (pd.Timestamp(datetime.datetime(2022, 11, 16)) - pd.to_datetime(df_[dt_col]).dt.tz_localize(None))\
                                    .map(lambda x: x.days)
            del df_[dt_col]

    dummy_cols = []
    for col, tp in zip(df_.columns, df_.dtypes.tolist()):
        if tp == 'object' or tp == 'bool': 
            if args.dummy and df_[col].unique().shape[0]<=300:
                    dummy_cols.append(col)
            else:
                df_[col] = df_[col].astype("category")

    if args.dummy:
        df_ = pd.get_dummies(df_, columns = dummy_cols)
        print("dummy_cols===>", dummy_cols)

    return df_ 


if args.cvtt == 'tt':
    i = 1 
elif args.cvtt == 'cv':
    i = 0

files = glob.glob("./test/df-{}-{}-month-{}-csv/*.csv".format(args.dv.upper(), args.train_dt.strip(), args.month ))
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
df["earningAmt_COMBINE"] = df[["earningAmt_{}".format(cvg) for cvg in CVGS if cvg != "COMBINE"]].sum(axis=1)
df["claimAmt_COMBINE"] = df[["claimAmt_{}".format(cvg) for cvg in CVGS if cvg != "COMBINE"]].sum(axis=1)

assert df.loc[df['earningAmt_COMBINE'].isnull()].shape[0] == 0 and df.loc[df['claimAmt_COMBINE'].isnull()].shape[0] == 0

print(df['earningAmt_COMBINE'].describe())
print(df['claimAmt_COMBINE'].describe())

if args.cvtt == 'tt':
    files = glob.glob("./test/df-{}-{}-csv/*.csv".format(args.dv.upper(), args.test_dt.strip() ))
    df_test = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    df_test["earningAmt_COMBINE"] = df_test[["earningAmt_{}".format(cvg) for cvg in CVGS if cvg != "COMBINE"]].sum(axis=1)
    df_test["claimAmt_COMBINE"] = df_test[["claimAmt_{}".format(cvg) for cvg in CVGS if cvg != "COMBINE"]].sum(axis=1)
    print(df.shape, df_test.shape)
else:
    df_test = None 


# for cvg in CVGS[-1:]:
    
    # assert df.loc[df['cvg']==cvg].shape[0] == df.loc[df['cvg']==cvg]['id'].unique().shape[0]

cv(df, df_test, "COMBINE")
    







################## sklearn#####################
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html

'''
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_tweedie_deviance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def load_mtpl2():
    # Fetch the French Motor Third-Party Liability Claims dataset.

    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = fetch_openml(data_id=41214, as_frame=True)["data"]
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = fetch_openml(data_id=41215, as_frame=True)["data"]

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"].fillna(0, inplace=True)

    # unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")
    return df

df = load_mtpl2()

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
import numpy as np 
# Note: filter out claims with zero amount, as the severity model
# requires strictly positive target values.
df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

# Correct for unreasonable observations (that might be data error)
# and a few exceptionally large claim amounts
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)


# sns.histplot(df, x='ClaimNb')
# plt.show()

# Insurances companies are interested in modeling the Pure Premium, that is
# the expected total claim amount per unit of exposure for each policyholder
# in their portfolio:
df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]

# This can be indirectly approximated by a 2-step modeling: the product of the
# Frequency times the average claim amount per claim:
df["Frequency"] = df["ClaimNb"] / df["Exposure"]
df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)


feat_cols = ['Area', 'VehPower','VehAge','DrivAge','BonusMalus', 'VehBrand', 'VehGas','Density', 'Region']

target_cols = ['ClaimNb', 'ClaimAmount', 'PurePremium', 'AvgClaimAmount', 'Frequency']

df['VehGas'] = df['VehGas'].astype("category")

objective = ("reg:tweedie", "reg:pseudohubererror", "reg:squarederror")
for target in target_cols:
    dall = xgb.DMatrix(df[feat_cols], df[target], enable_categorical=True)

    for obj in objective:
        param = {"tree_method": "exact",
                'max_depth':7, 'eta': 0.2, 'objective':obj,'eval_metric':['rmse','mae'],\
                  'subsample': 0.7, 'min_child_weight':0.2, 'colsample_bytree':0.7,\
                  'alpha':0,'lambda':0, 'max_bin':64,'disable_default_eval_metric':0, 'tweedie_variance_power':1.5} 

        # evallist= [(dtrain, 'train'),(dtest, 'eval')]
        # bst = xgb.train(param, dtrain,  num_boost_round=3000, evals =evallist , early_stopping_rounds=400,verbose_eval=10) # xgb_model

        result = xgb.cv(param, dall, num_boost_round=5000, nfold=5, early_stopping_rounds=100,verbose_eval=500)

        print(target, obj, result['test-rmse-mean'].min())
'''




'''
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor

def squeeze_label_5class(x):
    # 97% of the claim is 0
    if x == 0:
        return 0.0 
    elif x > 0 and x <= 2000:
        return 1.0 
    elif x > 2000 and x <= 10000:
        return 2.0 
    elif x > 10000 and x <= 50000:
        return 3.0 
    elif x > 50000:
        return 4.0
    else:
        raise ValueError(f'Invaid number: {x}')

def squeeze_label_1w(x):
    return min(float(x) / 10000, 1.0)

def squeeze_label_log(x):
    assert x >= 0
    return math.log(1 + x)

reg_funcs = {"5c":squeeze_label_5class, "1w":squeeze_label_1w, "log":squeeze_label_log}

def ith_(v, i):
    try:
        return float(v[i])
    except ValueError:
        return None

ith = F.udf(ith_, DoubleType())

def boolean2int(x):
    if x is True:
        return 1 
    elif x is False:
        return 0 
    else:
        return x 
udf_func_boolean2int = F.udf(lambda x: boolean2int(x), IntegerType())


def loss_premium_label(session_loss, session_prem):
    return math.log(1+session_loss) / math.log(1+session_prem)

udf_loss_premium_label = F.udf(loss_premium_label, FloatType()) 

def train_test_split(df_all_comb_label_binary, train_off_dt, test_on_dt, test_off_dt):
    sel_cols = ["policy_num", "vin", "dln", "term_eff", "ver_eff", "ver_exp", "ver_days", "session_prem", "session_loss", "features_assemble_dense", "label"]
    training_Valid_Data_raw = df_all_comb_label_binary\
            .select(sel_cols)\
            .filter(col("term_eff") <= train_off_dt)

    test_data = df_all_comb_label_binary\
            .select(sel_cols)\
            .filter((col("term_eff") >= test_on_dt) & (col("term_eff") <= test_off_dt))

    print("train cnt==>", training_Valid_Data_raw.count(), "test cnt==>", test_data.count())    
    return training_Valid_Data_raw, test_data

def feature_post_processing(df, train_off_dt):
    no_features_cols = ("policy_num", "vin", "dln",
                        "term_eff", "term_days", \
                        "ver_eff", "ver_exp", "ver_days",\
                        "session_prem", "session_loss")

    boolean_columns = []
    category_columns = []
    for ii, tp in df.dtypes:
        if ii in no_features_cols:
            continue
        else:
            if tp == "string":
                category_columns.append(ii)
            elif tp == "boolean":
                boolean_columns.append(ii)

    # print("boolean_columns===>", boolean_columns )
    # print("category_columns===>", category_columns )

    indexer = StringIndexer(inputCols = category_columns,
                            outputCols = [c + "_cate2int" for c in category_columns],
                            stringOrderType="frequencyAsc")\
            .setHandleInvalid("skip")\
            .fit(df.filter(col("ver_exp") <= train_off_dt))

    # indexer_saved_path = "gs://definity-aw-prod-dswork-pidc/temp/df_dynamic_session_indexer"
    # indexer.write().overwrite().save(indexer_saved_path)
    # loadedIndexer = StringIndexer.load(indexer_saved_path)

    for ii in boolean_columns:
        df = df\
                .withColumn("{}_bool2int".format(ii), udf_func_boolean2int(col(ii)))\
                .drop(ii)

    # print("boolean_columns transformed")

    df = indexer\
                .setHandleInvalid("keep")\
                .transform(df)\
                .drop(*category_columns)

    # print("category_columns transformed")
    assembler = VectorAssembler(
                            inputCols= [c for c in df.columns if c not in no_features_cols],
                            outputCol="features_assemble",
                            handleInvalid="keep")

    return assembler.transform(df)\
                    .withColumn('features_assemble_dense', F.udf(lambda v: Vectors.dense(v.toArray()), VectorUDT())('features_assemble'))\
                    .drop("features_assemble"), [c for c in df.columns if c not in no_features_cols]


df_all_comb = earning_claim_feature_df


train_off_dt = "2020-12-31"
test_on_dt = (datetime.datetime.strptime(train_off_dt, "%Y-%m-%d") + dateutil.relativedelta.relativedelta(days=1)).strftime('%Y-%m-%d')
test_off_dt = "2021-03-31"
print(train_off_dt, test_on_dt, test_off_dt)
# train cnt==> 2005547 test cnt==> 233728


for attr, experiment_trim_feats in attr_dic.items():
    # if attr not in ['all']:
    #     continue
    print(attr, len(experiment_trim_feats))

    df_all_comb_post, feature_names = feature_post_processing(df_all_comb.drop(*experiment_trim_feats), train_off_dt)
    
    feature_cnt = df_all_comb_post.select("features_assemble_dense").take(10)[0]['features_assemble_dense'].shape[0]
    assert feature_cnt == len(feature_names)  
    print("feature_cnt===>", feature_cnt)

    evaluator = RegressionEvaluator(metricName="rmse")


    df_all_comb_label_binary = df_all_comb_post\
            .withColumn("label", F.udf(lambda x: reg_funcs['log'](x), FloatType())(col("session_loss")))
                 
    training_Valid_Data_raw, test_data = train_test_split(df_all_comb_label_binary,\
                                     train_off_dt, test_on_dt, test_off_dt)

    metric_infer_result = {}
    auc_train_l, auc_valid_l, auc_test_l = [], [], []
    top_ratio_l, bottom_ratio_l = [], []
    for ite in range(5):
        training_valid_data = training_Valid_Data_raw.withColumn("validationIndicatorCol", F.rand(1) >= 0.9)
        xgbm = SparkXGBRegressor(
                  features_col= "features_assemble_dense",
                  label_col= "label",
                  n_estimators = 3000, 
                  max_depth = 7,
                  max_bin  = 256,
                  learning_rate = 0.1,
                  min_child_weight = 0.2,
                  subsample = 0.7,
                  colsample_bytree = 0.7,
                  # reg_alpha  = 0,
                  # reg_lambda =0,
                  num_workers = 16,
                  max_cat_to_onehot = 10,
                  max_cat_threshold = 1000,
                  eval_metric="rmse",
                  early_stopping_rounds = 50,
                  # missing=None,
                  maximize_evaluation_metrics = False,
                  validation_indicator_col="validationIndicatorCol"
        )

        # train and return the model
        model = xgbm.fit(training_valid_data)

        predictions_test = model.transform(test_data)
        predictions_train_valid = model.transform(training_valid_data)

        auc_test = evaluator.evaluate(predictions_test)
        auc_train = evaluator.evaluate(predictions_train_valid.filter(col("validationIndicatorCol")==False)) 
        auc_valid = evaluator.evaluate(predictions_train_valid.filter(col("validationIndicatorCol")==True)) 

        # predictions_test.show()

        auc_train_l.append(auc_train)
        auc_valid_l.append(auc_valid)
        auc_test_l.append(auc_test)

        infer_result = predictions_test.withColumnRenamed("prediction", "score").drop("features_assemble_dense")
        metric_infer_result[auc_test] = infer_result
        

        percentile_cut = 0.05
        percentile = infer_result.select(F.percentile_approx("score", [1-percentile_cut])).collect()[0][0][0]

        top_ratio = infer_result.filter(col("score")>=percentile).agg(F.sum("session_loss")).collect()[0][0] / \
                    infer_result.filter(col("score")>=percentile).agg(F.sum("session_prem")).collect()[0][0]

        bottom_ratio = infer_result.filter(col("score")<percentile).agg(F.sum("session_loss")).collect()[0][0] / \
                       infer_result.filter(col("score")<percentile).agg(F.sum("session_prem")).collect()[0][0]

        top_ratio_l.append(top_ratio)
        bottom_ratio_l.append(bottom_ratio)
        print("top vs bottom:", ite, top_ratio, bottom_ratio)
        print(infer_result.filter(col("score")>=percentile).select('vin', 'policy_num', 'term_eff').distinct().count(), 
              infer_result.filter(col("score")< percentile).select('vin', 'policy_num', 'term_eff').distinct().count())
        # print("metrics:", auc_train, auc_valid, auc_test)
        print()
    
    print("summary====>", np.array(top_ratio_l).mean(), np.array(bottom_ratio_l).mean())
    
    print()
'''