
def update_store(tmp, store, added_col):
    for ii in tmp:
        if not store.get(ii[kk], None):
            store[ii[kk]] = {}
        if not store[ii[kk]].get(ii['dt'].strftime('%Y-%m-%d'), None):
            store[ii[kk]][ii['dt'].strftime('%Y-%m-%d')] = {}
        store[ii[kk]][ii['dt'].strftime('%Y-%m-%d')].update({added_col: ii[added_col]})
    return store  

dts = pd.date_range("2016-08-01", "2022-12-31", freq='1m')
dt_offset_max = dts[0] - datetime.timedelta(days=365 * 4)

triplet = ["vin", "dln", "policy_num"]
########################### claim ###################################

for kk in triplet:
    ids = fetch_claim_unit(dt_offset_max.strftime('%Y-%m-%d'), dts[-1].strftime('%Y-%m-%d'), "ALL")\
                    .select(kk)\
                    .distinct().cache()

    dsl = [ids.withColumn("dt", F.lit(dt.strftime('%Y-%m-%d'))).withColumn("dt", F.to_date("dt")) for dt in dts]
    df_kk = reduce(DataFrame.unionAll, dsl).cache()

    loss_cvg_dic = {}
    for cvg in CVGS:
        loss_cvg_dic[cvg] = fetch_claim_unit(dt_offset_max.strftime('%Y-%m-%d'), dts[-1].strftime('%Y-%m-%d'), cvg).cache()


    for cvg in CVGS:
        print(cvg)
        store = {}
        for months_delta in range(RETRO_MONTH + 1):

            df_kk_offeset = df_kk\
                                 .withColumn("dt_offset_i", F.udf(lambda x: x.strftime('%Y-%m-%d'), StringType())(F.add_months(col("dt"), -(months_delta+1))))\
                                 .withColumn("dt_offset_j", F.udf(lambda x: x.strftime('%Y-%m-%d'), StringType())(F.add_months(col("dt"), -(months_delta))))\
                                 .withColumn("dt_offset_i", F.to_date("dt_offset_i"))\
                                 .withColumn("dt_offset_j", F.to_date("dt_offset_j"))

            added_col = "{}_loss_sum_retro_bin_{}_{}".format(kk, cvg, months_delta)
            tmp1 = df_kk_offeset.join(loss_cvg_dic[cvg], 
                    (df_kk_offeset[kk] == loss_cvg_dic[cvg][kk]) & 
                    (df_kk_offeset["dt_offset_i"] <= loss_cvg_dic[cvg]["loss_dt"]) & 
                    (df_kk_offeset["dt_offset_j"] >  loss_cvg_dic[cvg]["loss_dt"]), "inner" )\
                .drop(loss_cvg_dic[cvg][kk])\
                .select(kk, 'dt', 'int_claim_num', 'loss_dt', 'loss_sum')\
                .distinct()\
                .groupBy(kk, 'dt').agg(F.sum("loss_sum").alias(added_col))\
                .withColumn(added_col, F.col(added_col).cast(IntegerType()))\
                .filter(col(added_col) > 0)\
                .collect()
            store = update_store(tmp1, store, added_col)

            added_col = "{}_loss_sum_retro_acum_{}_{}".format(kk, cvg, months_delta)
            tmp2 = df_kk_offeset.join(loss_cvg_dic[cvg], 
                    (df_kk_offeset[kk] == loss_cvg_dic[cvg][kk]) & 
                    (df_kk_offeset["dt_offset_i"] <= loss_cvg_dic[cvg]["loss_dt"]) & 
                    (df_kk_offeset["dt"]           > loss_cvg_dic[cvg]["loss_dt"]), "inner" )\
                .drop(loss_cvg_dic[cvg][kk])\
                .select(kk, 'dt', 'int_claim_num', 'loss_dt', 'loss_sum')\
                .distinct()\
                .groupBy(kk, 'dt').agg(F.sum("loss_sum").alias(added_col))\
                .withColumn(added_col, F.col(added_col).cast(IntegerType()))\
                .filter(col(added_col) > 0)\
                .collect()

            store = update_store(tmp2, store, added_col)
                
            print(months_delta, len(tmp1), len(tmp2)) 
        

        schema = StructType([
          StructField(kk, StringType(), True),
          StructField('properties', MapType(StringType(),MapType(StringType(), IntegerType() ) ), True)
        ])

        df_store = spark.createDataFrame(data=store.items(), schema = schema)
        df_store.show()
        print(df_store.count())

        df_store.write.mode("overwrite")\
            .parquet("gs://definity-aw-prod-dswork-pidc/temp/dynamic_time_frame/feature_store/feature_store_claim_{}_{}".format(kk, cvg))



########################### earning ###################################

EarningUnitDF_bundle = fetch_earning_unit(dt_offset_max.strftime('%Y-%m-%d'), dts[-1].strftime('%Y-%m-%d'), None, None, None, False)\
                .cache()


for kk in ('vin', 'dln'):
    ids = EarningUnitDF_bundle\
                    .select(kk)\
                    .distinct().cache()

    dsl = [ids.withColumn("dt", F.lit(dt.strftime('%Y-%m-%d'))).withColumn("dt", F.to_date("dt")) for dt in dts]
    df_kk = reduce(DataFrame.unionAll, dsl).cache()
    
    store = {}
    for mj, mi in [(0, 6), (0, 12), (0,24), (0,36), (0, 48), (12, 24), (24, 36), (36, 48)]:

        df_kk_offeset = df_kk\
                             .withColumn("dt_offset_j", F.udf(lambda x: x.strftime('%Y-%m-%d'), StringType())(F.add_months(col("dt"), -mj)))\
                             .withColumn("dt_offset_i", F.udf(lambda x: x.strftime('%Y-%m-%d'), StringType())(F.add_months(col("dt"), -mi)))\
                             .withColumn("dt_offset_i", F.to_date("dt_offset_i"))\
                             .withColumn("dt_offset_j", F.to_date("dt_offset_j"))

        tmp = df_kk_offeset.join( EarningUnitDF_bundle,
                    (df_kk_offeset[kk] == EarningUnitDF_bundle[kk]) & 
                    (df_kk_offeset['dt_offset_i'] <= EarningUnitDF_bundle['ver_eff']) &
                    (df_kk_offeset['dt_offset_j'] > EarningUnitDF_bundle['ver_exp']) , "inner")\
                    .drop(EarningUnitDF_bundle[kk])\
                    .groupBy(kk, 'dt')\
                    .agg(F.sum("session_prem").alias("{}_prem_sum_retro_accum_{}_{}".format(kk, mj, mi)),
                         F.variance("session_prem").alias("{}_prem_var_retro_accum_{}_{}".format(kk, mj, mi)),
                         F.mean("session_prem").alias("{}_prem_mean_retro_accum_{}_{}".format(kk, mj, mi)),
                         F.countDistinct([i for i in triplet if i != kk][0]).alias("{}_retro_cntdist_{}_{}_{}".format(kk, [i for i in triplet if i != kk][0], mj, mi)),
                         F.countDistinct([i for i in triplet if i != kk][1]).alias("{}_retro_cntdist_{}_{}_{}".format(kk, [i for i in triplet if i != kk][1], mj, mi)),
                         F.countDistinct("term_eff").alias("{}_retro_cntdist_term_eff_{}_{}".format(kk, mj, mi)),
                         F.countDistinct("ver_eff").alias("{}_retro_cntdist_ver_eff_{}_{}".format(kk, mj, mi)),
                         F.mean("ver_days").alias("{}_retro_mean_ver_days_{}_{}".format(kk, mj, mi)),
                         F.variance("ver_days").alias("{}_retro_var_ver_days_{}_{}".format(kk, mj, mi))
                        )\
                    .na.fill(0).cache()        

        for ii in tmp.collect():
            if not store.get(ii[kk], None):
                store[ii[kk]] = {}
            if not store[ii[kk]].get(ii['dt'].strftime('%Y-%m-%d'), None):
                store[ii[kk]][ii['dt'].strftime('%Y-%m-%d')] = {}

            for added_col in tmp.columns:
                if added_col not in (kk, "dt"):
                    store[ii[kk]][ii['dt'].strftime('%Y-%m-%d')].update({added_col: int(ii[added_col])})

        print(mj, mi, tmp.count()) 
     


    schema = StructType([
      StructField(kk, StringType(), True),
      StructField('properties', MapType(StringType(), MapType(StringType(), IntegerType() ) ), True)
    ])

    df_store = spark.createDataFrame(data=store.items(), schema = schema).cache()
    df_store.show()
    print(df_store.count())
    
    df_store.write.mode("overwrite")\
        .parquet("gs://definity-aw-prod-dswork-pidc/temp/dynamic_time_frame/feature_store/feature_store_prem_{}".format(kk))

'''
store: {vin : {dt: }}
'''








earning_claim_df = spark.read.parquet("gs://definity-aw-prod-dswork-pidc/temp/dynamic_time_frame/earning_claim_df")


def append_stats_features_as_of(earning_claim_df):

    stats_bin_acum_cols = []
    for kk in ('vin', 'dln'):
        for cvg in CVGS:
            for months_delta in range(24 + 1):
                for ba in ('bin', 'acum'):
                    stats_bin_acum_cols.append("{}_loss_sum_retro_{}_{}_{}".format(kk, ba, cvg, months_delta))

    earning_claim_df_asof = earning_claim_df.cache()

    point_in_time_cols = []
    for kk in ('vin', 'dln'):
        for cvg in CVGS:
            feature_store_kk_cvg = spark.read\
                    .parquet("gs://definity-aw-prod-dswork-pidc/temp/dynamic_time_frame/feature_store_{}_{}".format(kk, cvg))
            added_col = 'point_in_time_properties_{}_{}'.format(kk, cvg)
            earning_claim_df_asof = earning_claim_df_asof\
                                            .join(feature_store_kk_cvg, kk, "left")\
                                            .withColumn(added_col, point_in_time_udf('ver_eff', 'properties'))\
                                            .drop("properties")
            point_in_time_cols.append(added_col)

    earning_claim_df_states_vec =  earning_claim_df_asof\
                                    .withColumn("V_tuple", F.array([col(c) for c in point_in_time_cols ]))\
                                    .withColumn("V_dic", dic_comb_udf('V_tuple'))\
                                    .drop(*point_in_time_cols)\
                                    .drop("V_tuple")\
                                    .withColumn("stats_vec", dic2vec_udf(stats_bin_acum_cols)(col("V_dic")))\
                                    .drop("V_dic")

    for i, c in enumerate(stats_bin_acum_cols):
        earning_claim_df_states_vec = earning_claim_df_states_vec\
                                            .withColumn(c, col("stats_vec")[i])

    return earning_claim_df_states_vec.drop("stats_vec"), stats_bin_acum_cols

def append_stats_features_as_of_03(earning_claim_df):
    triplet = ["vin", "dln", "policy_num"]
    df_prem = fetch_earning_unit("2005-01-01", "2022-12-31")

    df_prem_loss = df_prem
    for cvg in CVGS:
        claim_tmp = fetch_claim_unit("2008-01-01",  cvg).cache()
        earning_claim_df_tmp = join_earning_claim(df_prem, claim_tmp).cache()
        df_prem_loss = df_prem_loss.join(earning_claim_df_tmp.drop("session_prem"), session_keys, "inner")\
                                    .withColumnRenamed('session_loss', 'session_loss_{}'.format(cvg))

    feature_store = {}
    for kk in triplet:
        kk_pair0 = [i for i in triplet if i != kk][0]
        kk_pair1 = [i for i in triplet if i != kk][1]

        df_kk_term_prem = df_prem_loss\
            .groupBy(kk, 'term_eff').agg(F.sum("session_prem").alias("term_prem_{}".format(kk)), 
                                        (F.sum("session_prem") / F.sum('ver_days')).alias("term_prem_mean_{}".format(kk)),
                                         F.countDistinct('ver_eff').alias("ver_cnt_{}".format(kk)),
                                         F.min('ver_days').alias("ver_days_min_{}".format(kk)),
                                         F.max('ver_days').alias("ver_days_max_{}".format(kk)),
                                         F.mean('term_days').alias("term_days_mean_{}".format(kk)),
                                         F.countDistinct(kk_pair0).alias("cntd_{}_{}".format(kk_pair0, kk)),
                                         F.countDistinct(kk_pair1).alias("cntd_{}_{}".format(kk_pair1, kk)),
                                        (F.sum("session_loss_DC") / F.sum('ver_days')).alias("term_loss_mean_DC_{}".format(kk)),
                                         F.sum("session_loss_DC").alias("term_loss_DC_{}".format(kk)), 
                                        (F.sum("session_loss_COMPSP") / F.sum('ver_days')).alias("term_loss_mean_COMPSP_{}".format(kk)),
                                         F.sum("session_loss_COMPSP").alias("term_loss_COMPSP_{}".format(kk)), 
                                        (F.sum("session_loss_APCOLL") / F.sum('ver_days')).alias("term_loss_mean_APCOLL_{}".format(kk)),
                                         F.sum("session_loss_APCOLL").alias("term_loss_APCOLL_{}".format(kk)), 
                                        (F.sum("session_loss_TPL") / F.sum('ver_days')).alias("term_loss_mean_TPL_{}".format(kk)),
                                         F.sum("session_loss_TPL").alias("term_loss_TPL_{}".format(kk)),  
                                        (F.sum("session_loss_UIN") / F.sum('ver_days')).alias("term_loss_mean_UIN_{}".format(kk)),
                                         F.sum("session_loss_UIN").alias("term_loss_UIN_{}".format(kk)),                                                                          
                                        (F.sum("session_loss_AB") / F.sum('ver_days')).alias("term_loss_mean_AB_{}".format(kk)),
                                         F.sum("session_loss_AB").alias("term_loss_AB_{}".format(kk)) 
                                         )\
            .withColumnRenamed("term_eff", "term_eff_kk")

        windowSpec  = Window.partitionBy([kk, 'term_eff']).orderBy(col("term_eff_kk").desc())

        df_rownum = earning_claim_df.select(kk, "term_eff").distinct()\
            .join(df_kk_term_prem, [kk], "inner")\
            .filter(df_kk_term_prem["term_eff_kk"] < earning_claim_df["term_eff"])\
            .withColumn("row_number", F.row_number().over(windowSpec))\
            .withColumn("term_dayssince_{}".format(kk), F.datediff(col("term_eff"), col("term_eff_kk")))\
            .drop("term_eff_kk")

        feat_cols = [i for i in df_rownum.columns if i not in (kk, 'term_eff')]

        dfplrn = earning_claim_df.select(kk, "term_eff").distinct()
        for rn in (1, 2, 3, 4):
            dfplrn = dfplrn.join(df_rownum.filter(col("row_number")==rn).drop("row_number"), [kk, 'term_eff'], 'left')
            for c in feat_cols:
                dfplrn = dfplrn.withColumnRenamed(c, c + "_" + str(rn))
        feature_store[kk] = dfplrn
        print(kk, dfplrn.count(), dfplrn.select(kk, "term_eff").distinct().count())

    for kk in triplet:
        earning_claim_df = earning_claim_df.join(feature_store[kk], [kk,'term_eff'], 'left')

    stats_cols = [i for i in earning_claim_df.columns if i not in session_keys + ['session_prem','session_loss'] ]
    return earning_claim_df, stats_cols



def point_in_time(ver_eff, properties):
    if not properties:
        return {}
    dts = [dt for dt, dd in properties.items() if dt <= ver_eff.strftime('%Y-%m-%d')]
    if not dts:
        return {}
    dt_point = max(dts)
    return properties[dt_point]
point_in_time_udf = F.udf(point_in_time, MapType(StringType(), IntegerType()))

def dic_comb(V_tuple):
    dic = {}
    for t in V_tuple:
        dic.update(t)
    return dic 
dic_comb_udf = F.udf(dic_comb, MapType(StringType(), IntegerType()))


def dic2vec(dic, stats_cols):

    init = [0] * len(stats_cols)
    if not dic:
        return init

    for i, c in enumerate(stats_cols):
        init[i] = dic.get(c, 0)
    return init
def dic2vec_udf(stats_cols):
    return F.udf(lambda x: dic2vec(x, stats_cols) , ArrayType(IntegerType()) )
