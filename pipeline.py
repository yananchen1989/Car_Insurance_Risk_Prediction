from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.ml.linalg import DenseVector, SparseVector, Vectors, VectorUDT
import datetime,math,argparse
import dateutil.relativedelta
import numpy as np
import pandas as pd 
from functools import reduce
from pyspark.sql import DataFrame
from operator import add
# import matplotlib.pyplot as plt
# import seaborn as sns

CUT_ON_DT =  "2016-01-01" 
CUT_OFF_DT = "2022-12-31" # datetime.datetime.today().strftime('%Y-%m-%d')   

CVGS = ["DC", "COMPSP", "APCOLL", "TPL",  "AB"] # 
session_keys = ["policy_num", "vin", "dln", "term_eff", "term_days", "ver_eff", "ver_exp", "ver_days"] # , "expos_num" "vehNum_riskId",
event_keys = ["policy_num", "vin", "dln", "ver_eff"]
loss_cols = ["{}_loss".format(cvg) for cvg in CVGS] 

dln_noise_ids = ['COMPANY','UNLICENSED','COMPAGNIE','FLEET','$$UNLICENSED','$$COMPANY','1234567',"", "UNKNOWN", "Unknown_DLN"]
vin_noise_ids = ["", '1234567', '12345678', 'SEARCH', '123456', 'NF']

def nacheck(df):
    for c in df.columns:
        print(c, round(df.filter(col(c).isNull()).count()/ df.count(), 4), df.select(c).distinct().count()  )

def read_gcp_hive_table(project, table_name):
    return spark.read\
                .format("bigquery")\
                .option("project", project)\
                .option("table", table_name)\
                .load()

def get_vin_dln_census():
    ################## remove useless features ################
    # rm_cols = []
    # for ii in vin_census_df.columns:
    #     if  vin_census_df.filter(col(ii).isNull()).count() / vin_census_df.count() >= 0.999 or vin_census_df.select(ii).distinct().count() == 1:
    #             print(vin_census_df.select(ii).dtypes)
    #             rm_cols.append(ii)

    # print("rm_cols====>", rm_cols)

    dln_rm_cols = ['applicable_good_driver_discount',
                 'do_not_order_mvr',
                 'group_discount',
                 'has_good_driver_discount',
                 'number_of_accidents',
                 'number_of_violations',
                 'pol_process_date_auto_plus',
                 'years_licensed_auto_plus',
                 'years_licensed_exp_auto_plus',
                 'legacy_prior_term_grid_step',
                 'ubi_driving_res_disc_override',
                 'ubi_drv_res_disc_override_val',
                 'is_ubi_score_reached_threshold',
                 'ubi_consent_date',
                 'withdrawn_from_ubi']
    vin_rm_cols = ['iso_transfer_date',
                     'rsp_transfer_date',
                     'stated_value_amt',
                     'fuel_tank_attached',
                     'hauling_for_others',
                     'minimum_radius',
                     'percentage_of_pleasure_use',
                     'lowest_years_licensed_rating',
                     'days_in_use',
                     'height_eco',
                     'km_at_purchase',
                     'length_eco',
                     'number_of_wheels',
                     'us_exposure_time',
                     'min_os_yrs_lic_or_yrs_exp_rating',
                     'num_of_passengers',
                     'prior_term_prem_for_deviation',
                     'unrepaired_damage',
                     'carry_paypass_or_delivery',
                     'is_emerg_veh_or_rented',
                     'is_uber',
                     'cupping_factor_override',
                     'capping_eligibility_level',
                     'capping_factor_override',
                     'is_emergency_vehicle',
                     'is_rented_others_vehicle',
                     'is_turo',
                     'ubi_enabled']
    rm_cols = dln_rm_cols + vin_rm_cols

    #################### load census features ########################
    census_da_mapping = read_gcp_hive_table("aw-prod-dswork-pidc-b02d", "dp-prod-derived-2072.product_pi_all.external_census_2016_da_mapping")
    census_da_short = read_gcp_hive_table("aw-prod-dswork-pidc-b02d", "dp-prod-derived-2072.product_pi_all.external_census_2016_da_short").drop('dlh_batch_ts', 'dlh_process_ts')                
    
    census_df = census_da_mapping.join(census_da_short, ["da"], "inner")\
                .select(["postalcode", "province", "municipalityname", "longitude", "latitude"] + census_da_short.drop("da").columns)

    VinPostalDF = spark.read.parquet("gs://definity-aw-prod-dswork-pidc/temp/dfSchemaVinPostal").filter((~col("vin").isin(vin_noise_ids)))
    vin_census_df = VinPostalDF.join(census_df, ["postalcode"], "inner").drop(*rm_cols).distinct().cache()


    ########### load vin features #####################
    # vin_df_raw = read_gcp_hive_table("aw-prod-dswork-pidc-b02d", "dp-prod-curated-b30b.uw_pc.vehicle")
    vin_df_raw = spark.read.parquet("gs://definity-aw-prod-dswork-pidc/odc/Parquet/from_edd/vehicle").cache()
    vin_sel_cols = [ "vin",  
             "processed_date",  
             "abs",  
             "accident_benefits_rate_group",  
             "air_bags",  
             "annualmileage",  
             "audible_alarm",  
             "body_type",  
             "car_code",  
             "collision_rate_group",  
             "commuting_miles",  
             "comprehensive_cov_only",  
             "comprehensive_rate_group",  
             "condition_purchased",  
             "cost_new_amt",  
             "date_purchased",  
             "drivers_count",  
             # "effective_date",  
             # "expiration_date",  
             "extended_car_code",  
             "fixed_id",  
             "garage_location_public_id",  
             "has_iso_driver",  
             "horse_power",  
             # "iso_transfer_date",  ##
             "make",  
             "make_region",  
             "meets_grid_criteria",  
             "model",  
             "ownership",  
             "percentage_of_business_use",  
             "rental_extension_bundle",  
             "rsp_ceded",  
             "rsp_override",  
             # "rsp_transfer_date", ## 
             "size_code",  
             "stability_control",  
             "stated_value_amt",  
             "substandard_vehicle",  
             "total_cost_amount",  
             "vehicle_age",  
             # "vehicle_number",  
             "vehicle_type",  
             "vicc_vehicle_code",  
             "vroom_bundle",  
             "weight",  
             "wheel_base",  
             "winter_tires",  
             "year",  
             "years_owned",  
             # "update_time",  
             "is_downward_deviated",  
             "fuel_type",  
             "renewal_deviation_factor",  
             "has_protection_plus",  
             "after_market_alarm_type",  
             "vehicle_modified",  
             "vehicle_modifications",  
             "canc_non_pay_surcharge",  
             "drive_train_code",  
             "years_since_conv",  
             "assigned_towing_vehicle_public_id",  
             "cargo_delivery_frequency",  
             "cargo_delivery_type",  
             "commercial_vehicle",  
             "fuel_tank_attached",  
             "hauling_for_others",  
             "hired_snowplow_for_others",  
             "liability_rate_group",  
             "maximum_radius",  
             "minimum_radius",  
             "normal_radius",  
             "percentage_of_pleasure_use",  
             "trailer_type",  
             "trailer_use",  
             "trips_beyond_normal_radius",  
             "type_of_use",  
             "type_of_use_category",  
             "vdm_override",  
             "vehicle_classification",  
             "vehicle_description",  
             "engine_forced_induction",  
             "lowest_years_licensed_rating",  
             "traction_control",  
             "vehicle_total_cost",  
             "years_of_experience_heavy_rating",  
             "acct_commercial_mvd_cnt",  
             "acct_personal_mvd_cnt",  
             "applied_flex_current_job",  
             "available_flex",  
             "calculated_flex",  
             "days_in_use",  
             "engine_displacement",  
             "flex_remaining_amt",  
             "flex_risk_total_prem_amt",  
             "has_motor_home_plus",  
             "height_eco",  
             "is_eligible_for_flex",  
             "km_at_purchase",  
             "length_eco",  
             "lowest_age_secondary_operator_rating",  
             "number_of_wheels",  
             "pa_cov_suspend",  
             "percentage_applied_flex",  
             "percentage_available_flex",  
             "premium_with_flex_amt",  
             "premium_without_flex_amt",  
             "rating_car_code",  
             "recreational_mvd_indicator",  
             "us_exposure_time",  
             "age_of_po_rating",  
             "aggr_load_flying_off_veh_rating",  
             "cap_major_serious_conv_rating",  
             "cap_new_chargeable_claim_rating",  
             "cap_new_conv_rating",  
             "class_eco",  
             "cooking_rating",  
             "dangerous_nondangerous_rating",  
             "engine_configuration",  
             "engine_stroke",  
             "farm_use",  
             "has_vehicle_accident_forgiveness",  
             "is_excess_trailer",  
             "legacy_prior_term_rsp_ceded",  
             "min_os_yrs_lic_or_yrs_exp_rating",  
             "num_of_passengers",  
             "prem_under_cap_rating",  
             "primary_driver_unchanged_rating",  
             "primary_use",  
             "prior_term_prem_for_deviation", 
             "purchase_price", 
             "rec_drive_train",  
             "roadside_exposure_rating",  
             "susp_of_cov_discount",  
             "terms_with_company",  
             "tilt_shift_slosh_jacknife_rating",  
             "time_constraint_rating",  
             "uneven_terrain_rating",  
             "unrepaired_damage",  
             # "veh_exist_on_latest_bound_period",  
             "veh_movement_rating",  
             "csio_fixed_id",  
             "primary_ibc_code",  
             "iso_ibc_code",  
             "gaa_vehicle_type_code",  
             "gaa_vehicle_usage_code",  
             "acct_comm_mvd_cnt_override",  
             "acct_mh_mvd_cnt_override",  
             "acct_ppv_mvd_cnt_override",  
    ]
    vin_df = vin_df_raw.select(vin_sel_cols)\
                        .filter(~col("vin").isin(["", "0", "NOT APPLICABLE", "TRAILER", "XXXXXXXXXXXXXXXXX","XXX", "XXXX", "X", "00000000", "123456", "000000", "00000", "0000"]))\
                        .drop(*rm_cols)\
                        .distinct()

    # vin_df.groupBy("vin").agg(F.countDistinct("processed_date").alias("processed_date_dist_cnt"))\
    #         .orderBy(F.desc("processed_date_dist_cnt"))\
    #         .show()

    ############# load dln features ######################
    # dln_df_raw = read_gcp_hive_table("aw-prod-dswork-pidc-b02d", "dp-prod-curated-b30b.uw_pc.policy_driver")
    dln_df_raw = spark.read.parquet("gs://definity-aw-prod-dswork-pidc/odc/temp/policy_driver").cache()
    dln_sel_cols = ["license_number", 
                "processed_date", 
                "gender_type", 
                "license_province", 
                "license_status", 
                "license_class", 
                "insurance_fraud_cancels10y", 
                "misrep_cancels3y", 
                "non_pay_cancels3y", 
                "non_pays", 
                "age_licensed", 
                "conviction_free", 
                "driver_training_completed", 
                "graduating_licensing_l1", 
                "number_years_licensed", 
                "years_of_full_license", 
                "years_of_full_license_label", 
                "order_status", 
                "icc_infractions", 
                "pdi_infractions", 
                "sub_standard", 
                "rsp_ceded", 
                "number_years_experience", 
                "retiree_discount", 
                "first_chance_discount", 
                "graduated_license_discount", 
                "away_at_school", 
                "farmer", 
                "group_discount", 
                "conv_free_disc_override", 
                "serious_convictions", 
                "major_convictions", 
                "minor_convictions", 
                "years_since_last_conv", 
                "insured_elsewhere", 
                "admin_suspended_flag", 
                "age", 
                "applicable_good_driver_discount", 
                "at_fault_accidents", 
                "date_first_license", 
                "date_of_full_license", 
                "do_not_order_mvr", 
                "driving_record", 
                "excluded", 
                "graduating_licensing_l1_date", 
                "graduating_licensing_l2", 
                "graduating_licensing_l2_date", 
                "grid_step", 
                "has_good_driver_discount", 
                "number_of_accidents", 
                "number_of_violations", 
                "years_licensed", 
                "autoplus_fcsa_response", 
                "date_autoplus_fcsa_ordered", 
                "prior_insurance_in_canada", 
                "date_insured_since", 
                "date_mvr_ordered", 
                "has_previous_policies", 
                "occupation", 
                "years_suspended", 
                "disclose_losses_or_claims", 
                "mvr_response", 
                "pol_process_date_auto_plus", 
                "years_licensed_auto_plus", 
                "years_licensed_exp_auto_plus", 
                "years_on_auto_plus", 
                "mvr_order_status", 
                "commercial_driver_only", 
                "date_insured_by_prior_carrier", 
                "has_auto_license", 
                "has_motorcycle_license", 
                "license_type_motorcycle", 
                "motorcycle_insured_elsewhere", 
                "physical_impairment", 
                "grad_lic_disc_override", 
                "grid_step_disc_override", 
                "report_source", 
                "effective_date_year", 
                "effective_date_month"]
    dln_df = dln_df_raw.select(dln_sel_cols).withColumnRenamed("license_number","dln").drop(*rm_cols).distinct()

    # dln_df.groupBy("dln").agg(F.countDistinct("processed_date").alias("processed_date_dist_cnt"))\
    #         .orderBy(F.desc("processed_date_dist_cnt"))\
    #         .show()
    return vin_df, dln_df, vin_census_df

def fetch_earning_unit(CUT_ON_DT, CUT_OFF_DT):
    file_path_earining_cdw = "gs://definity-aw-prod-dswork-pidc/odc/hnh/temp/internal/EarningUnit_CDW"
    file_path_earining_gw = "gs://definity-aw-prod-dswork-pidc/odc/hnh/temp/internal/EarningUnit_GW"

    EarningUnitDF_gw = spark.read.parquet(file_path_earining_gw)#.withColumn("source", F.lit("cdw"))
    EarningUnitDF_cdw = spark.read.parquet(file_path_earining_cdw)#.withColumn("source", F.lit("cdw"))
    EarningUnitDF = EarningUnitDF_gw.union(EarningUnitDF_cdw)\
                    .withColumn("ver_eff", F.to_date("ver_eff"))\
                    .withColumn("ver_exp", F.to_date("ver_exp"))\
                    .withColumn("term_eff", F.to_date("term_eff"))\
                    .withColumn("term_days", F.col("term_days").cast(IntegerType()))\
                    .cache()

    # EarningUnitDF.agg(F.min("term_eff"), F.max("term_eff"), F.min("ver_eff"), F.max("ver_eff"), F.mean("term_days")).show()
    '''
    +-------------+-------------+
    |min(term_eff)|max(term_eff)|
    +-------------+-------------+
    |   2000-01-01|   2023-06-01|
    +-------------+-------------+
    '''
    EarningUnitDF_filter = EarningUnitDF\
          .filter((col("term_eff") >= CUT_ON_DT) & (col("term_eff") <= CUT_OFF_DT))\
          .drop("veh_type")\
          .filter( (~col("vin").isin(vin_noise_ids)) & (~col("dln").isin(dln_noise_ids)))\
          .withColumn("ver_days", F.datediff(col("ver_exp"), col("ver_eff")))\
          .filter(col("ft_prem")  >= 0)\
          .withColumnRenamed("pol_num","policy_num")\
          .withColumn("exposure_number", F.col("expos_num").cast(IntegerType()))\
          .drop("company", "dln_prov", "expos_num")\
          .distinct()\
          .cache() # 130385492

    EarningUnitDF_filter_ = EarningUnitDF_filter.join( 
        EarningUnitDF_filter\
                .groupBy("policy_num", "vin", "term_eff")\
                .agg(F.countDistinct("term_days").alias("cnt"))\
                .filter(col("cnt")==1)\
                .distinct(),            \
        ["policy_num", "vin", "term_eff"], "inner").join(
        EarningUnitDF_filter\
            .groupBy("vin", "dln", "policy_num", "term_eff", "term_days")\
            .agg(F.countDistinct("exposure_number").alias("expos_num_cnt"))\
            .filter(col("expos_num_cnt")==1).drop("expos_num_cnt").distinct(),
        ["vin", "dln", "policy_num", "term_eff", "term_days"], "inner")\
        .cache()

    # 21% pol-vin-term has more than 1 driver

    # EarningUnitDF_filter_.groupBy("policy_num", "vin", "term_eff")\
    #             .agg(F.countDistinct("dln", 'driver_type').alias("cnt"))\
    #             .filter(col("cnt") > 1).show()

    # EarningUnitDF_filter_.groupBy("policy_num", "vin", "term_eff", "exposure_number", "ver_eff")\
    #             .agg(F.countDistinct("dln").alias("cnt"))\
    #             .filter(col("cnt") > 1).show()

    EarningUnitDF_agg = EarningUnitDF_filter_\
            .groupBy(["policy_num", "vin","dln", "term_eff", "term_days", "ver_eff", "ver_exp", "ver_days"] + ['driver_type','loaded_timest'])\
            .agg(F.sum("ft_prem").alias("ft_prem_sum"))\
            .withColumn("session_prem", F.round(col("ft_prem_sum") / col("term_days") * col("ver_days"), 2))\
            .drop("ft_prem_sum")


    session_keys_driver_type = ["vin","policy_num", "ver_eff"]
    windowSpec_vin_ver = Window.partitionBy(session_keys_driver_type).orderBy(col("loaded_timest").desc())

    EarningUnitDF_agg_sorted = EarningUnitDF_agg.withColumn("row_number", F.row_number().over(windowSpec_vin_ver))\
                .filter(col("row_number")==1)\
                .drop("row_number", "loaded_timest")\
                .distinct()

    print("filter loaded_timest most updated===>", EarningUnitDF_agg_sorted.count() / EarningUnitDF_agg.count())
    assert EarningUnitDF_agg_sorted.count() == EarningUnitDF_agg_sorted.select(session_keys_driver_type).distinct().count()

    # df_prem_primary = EarningUnitDF_agg_sorted.filter(col("driver_type") == "primary")\
    #                         .withColumnRenamed("dln", "dln_primary")\
    #                         .withColumnRenamed("session_prem", "session_prem_primary")\
    #                         .drop("driver_type")

    # df_prem_iso = EarningUnitDF_agg_sorted.filter(col("driver_type") == "iso")\
    #                             .withColumnRenamed("dln", "dln_iso")\
    #                             .withColumnRenamed("session_prem", "session_prem_iso")\
    #                             .drop("driver_type")

    # df_prem_secondary = EarningUnitDF_agg_sorted.filter(col("driver_type") == "secondary")\
    #                                 .withColumnRenamed("dln", "dln_secondary")\
    #                                 .withColumnRenamed("session_prem", "session_prem_secondary")\
    #                                 .drop("driver_type")
    

    # df_prem_driver_type = df_prem_primary\
    #                 .join(df_prem_iso, ["policy_num", "vin", "term_eff", "term_days", "ver_eff", "ver_exp", "ver_days"], 'left')\
    #                .join(df_prem_secondary, ["policy_num", "vin", "term_eff", "term_days", "ver_eff", "ver_exp", "ver_days"], 'left')\
    #                .na.fill(0, ['session_prem_iso'])\
    #                .withColumn("session_prem", col("session_prem_primary") + col("session_prem_iso"))\
    #                .drop("session_prem_secondary", "session_prem_primary", "session_prem_iso")\
    #                .cache()


    df_prem_driver_type_bundle = EarningUnitDF_agg_sorted.withColumn("session_prem_driver_type", 
                                        F.when(EarningUnitDF_agg_sorted.driver_type == "secondary", 0)\
                                         .otherwise(EarningUnitDF_agg_sorted.session_prem))\
                            .drop("driver_type", "session_prem")\
                            .withColumnRenamed("session_prem_driver_type", "session_prem")

                            # .groupBy("vin", "dln", "policy_num", "term_eff", "term_days")\
                            # .agg(F.min("ver_eff").alias("ver_eff"),  
                            #      F.max("ver_exp").alias("ver_exp"), 
                            #      F.sum("ver_days").alias("ver_days"),
                            #      F.sum(col("session_prem_driver_type")).alias("session_prem") )

    # print("after bundle==>",  df_prem_driver_type_bundle.count() / EarningUnitDF_agg_sorted.count() )

    # if filtervindln:
    #     EarningUnitDF_bundle_withf = EarningUnitDF_bundle\
    #         .join(vin_df.select("vin").distinct(), ["vin"], "inner")\
    #         .join(dln_df.select("dln").distinct(), ["dln"], "inner")\
    #         .join(vin_census_df.select("vin").distinct(), ["vin"], "inner")\
    #         .cache()

    #     print("filter join dln vin census ===>", EarningUnitDF_bundle_withf.count() / EarningUnitDF_bundle.count()  )
    #     return EarningUnitDF_bundle_withf
    # else:
    return df_prem_driver_type_bundle#, EarningUnitDF_agg_sorted.drop("driver_type")

def map_policy_province():

    datamart_dim_province_schema =  StructType([\
              StructField("province_dim_id", IntegerType(), nullable = False), \
              StructField("country_code", StringType(), nullable = False),\
              StructField("province", StringType(), nullable = False),\
              StructField("prov_code", StringType(), nullable = False),\
              StructField("prov_abbrev", StringType(), nullable = False),\
              StructField("region", StringType(), nullable = False),\
              StructField("alternate_region", StringType(), nullable = False)]) 

    datamart_dim_province = spark.read.option("header",False).option("delimiter", "|")\
                        .schema(datamart_dim_province_schema)\
                        .csv("gs://definity-prod-derived-bucket/product_edh_datalake/datamart_dim_province/dp_data.txt")\
                        .select("prov_code", "prov_abbrev")\
                        .withColumnRenamed("prov_code", "prov_cd")\
                        .distinct().cache()

    pre_guidewire_raw = read_gcp_hive_table("dp-prod-derived-2072", "dp-prod-derived-2072.product_cdw.tpv_policy").cache()
    pre_guidewire = pre_guidewire_raw\
                .select("POL_NUM", "POL_PROV_CD")\
                .withColumnRenamed("POL_NUM","policy_num")\
                .withColumnRenamed("POL_PROV_CD","prov_cd")\
                .join(datamart_dim_province, ['prov_cd'], "inner")\
                .drop("prov_cd")\
                .withColumnRenamed("prov_abbrev", "prov_cd")\
                .distinct()

    post_guidewire_raw = spark.read.parquet("gs://definity-prod-curated-bucket/uw_pc/uw_pc_policy_period").cache()
    post_guidewire = post_guidewire_raw.select("policy_number", "province_code")\
                .withColumnRenamed("policy_number","policy_num")\
                .withColumnRenamed("province_code","prov_cd")\
                .distinct()

    policy_province = pre_guidewire.union(post_guidewire).distinct().dropDuplicates(['policy_num'])
    assert policy_province.select("policy_num").distinct().count() == policy_province.count()
    return policy_province

def fetch_claim_unit(CUT_ON_DT):
    ################# claims ######################
    CUT_OFF_DT_claim = datetime.datetime.today().strftime('%Y-%m-%d')   

    schema =  StructType([  StructField("line_of_business", StringType(), nullable = False),
                            StructField("line_type", IntegerType(), nullable = False),
                            StructField("line_type_desc", StringType(), nullable = False),
                            StructField("cvg_type_cd", StringType(), nullable = False),
                            StructField("modeling_id_cl", StringType(), nullable = False)])

    df_CoverageMapping_Claim = spark.read.option("delimiter", ",")\
            .csv("gs://definity-aw-prod-dswork-pidc/temp/dynamic_time_frame/CoverageMapping_Claim.txt", header=True, schema=schema)

    claim_eclm_coverage_summary_auto_hive_tb =  read_gcp_hive_table("aw-prod-dswork-pidc-b02d", "dp-prod-curated-b30b.eclm.claim_coverage_summary_auto")\
                .filter(col("line_of_business") == "AUTO")\
                .filter((col("loss_date") >= CUT_ON_DT) & (col("loss_date") <= CUT_OFF_DT_claim))\
                .select("policy_num", "int_claim_num", "line_type", "line_type_desc", "loss_date", \
                        "processed_date", "last_process_date", \
                        "incurred_amt", "net_paid_amt", "status_desc", "cvg_type_cd", "company")\
                .join(df_CoverageMapping_Claim.drop("line_of_business"),
                                            ['line_type','line_type_desc','cvg_type_cd'], "inner")\
                .withColumnRenamed("modeling_id_cl", "cvg")
    '''
    +--------------+--------------------------------+
    |modeling_id_cl|count(policy_num, int_claim_num)|
    +--------------+--------------------------------+
    |            DC|                           90984|
    |        COMPSP|                          113775|
    |           UIN|                            1307|
    |        APCOLL|                          142249|
    |           TPL|                           52929|
    |            AB|                           30125|
    +--------------+--------------------------------+
    '''

    claim_eclm_common_auto_hive_tb = read_gcp_hive_table("aw-prod-dswork-pidc-b02d", "dp-prod-curated-b30b.eclm.claim_common_auto")\
            .filter(col("line_of_business") == "AUTO")\
            .filter((col("loss_date") >= CUT_ON_DT) & (col("loss_date") <= CUT_OFF_DT_claim))\
            .select("policy_num", "int_claim_num", "ins_lic_num", "serial_num", "loss_date","processed_date","last_process_date")\
            .withColumnRenamed("ins_lic_num", "dln")\
            .withColumnRenamed("serial_num", "vin")\
            .filter((~col("dln").isin(dln_noise_ids) ) & (~col("vin").isin(vin_noise_ids)))\
            .cache()


    windowSpec_claim  = Window.partitionBy(["policy_num", "int_claim_num", "cvg_type_cd", "loss_dt"])\
                                .orderBy(col("processed_date").desc())

    claim_join_cvg = claim_eclm_coverage_summary_auto_hive_tb\
                          .filter(col("incurred_amt") > 0)\
                          .select("policy_num", "int_claim_num",  "loss_date", "cvg_type_cd", "cvg", "processed_date", "incurred_amt")\
                          .withColumn("loss_dt", F.to_date("loss_date"))\
                          .drop("loss_date")\
                          .distinct()\
                          .withColumn("row_number", F.row_number().over(windowSpec_claim))\
                          .filter(col("row_number")==1)\
                          .drop("row_number", "processed_date", "cvg_type_cd")\
                          .groupBy("policy_num", "int_claim_num", "loss_dt", "cvg")\
                          .agg(F.sum("incurred_amt").alias("loss_sum"))\
                    .join(claim_eclm_common_auto_hive_tb\
                        .select("policy_num", "int_claim_num", "dln", "vin", "loss_date")\
                        .withColumn("loss_dt", F.to_date("loss_date"))\
                        .drop("loss_date")\
                        .distinct(),\
                               ["policy_num", "int_claim_num", "loss_dt"], "inner"
                      )\
                    .withColumn("loss_sum", F.col("loss_sum").cast(FloatType()))\
                    .cache()

    claim_join_cvg_agg =  claim_join_cvg.filter(col("cvg") != "UIN")\
                    .groupBy("policy_num", "vin", "dln","loss_dt", "cvg").agg(F.sum("loss_sum").alias("loss_sum"))      


    claim_info = claim_join_cvg_agg.select("policy_num", "vin", "dln", "loss_dt").distinct()
    claim_keys = claim_info.columns
    for cvg in CVGS:
        claim_info = claim_info.join(claim_join_cvg_agg\
                                        .filter(col("cvg")==cvg).drop("int_claim_num", "cvg")\
                                        .withColumn("{}_loss".format(cvg), F.col("loss_sum").cast(FloatType()))\
                                        .drop("loss_sum"), 
                                    claim_keys, "left")
    claim_info = claim_info.na.fill(0, subset=["{}_loss".format(cvg) for cvg in CVGS])
    claim_info.agg(F.min("loss_dt"), F.max("loss_dt")).show()
    claim_info.filter(col("dln")=="D6401-79009-25310")\
              .filter(col("vin")=="3CZRU6H23KM103296").show()
    return claim_info

# def join_earning_claim(EarningUnitDF, claim_info):
#     earning_claim_df = EarningUnitDF\
#                         .join(claim_info, 
#                            (EarningUnitDF['policy_num'] == claim_info['policy_num']) &
#                             (EarningUnitDF['vin'] == claim_info['vin']) &
#                             (EarningUnitDF['dln'] == claim_info['dln']) & #| (EarningUnitDF['dln_iso'] == claim_info['dln']) | (EarningUnitDF['dln_secondary'] == claim_info['dln']) ) &
#                             (claim_info['loss_dt'] >= EarningUnitDF['ver_eff']) &
#                             (claim_info['loss_dt'] <= EarningUnitDF['ver_exp']), "left")\
#                         .drop(claim_info['policy_num'])\
#                         .drop(claim_info['vin'])\
#                         .drop(claim_info['dln'])\
#                         .na.fill(0, subset=["loss_sum"])\
#                         .groupBy(session_keys + ["session_prem"])\
#                         .agg(F.sum("loss_sum").alias("session_loss"))\
#                         .distinct()\
#                         .withColumn("session_loss", F.col("session_loss").cast(FloatType()))\
#                         .withColumn("session_prem", F.col("session_prem").cast(FloatType()))\
#                         .withColumn("ver_days", F.col("ver_days").cast(IntegerType()))

#     return earning_claim_df

def join_feature_latest(df, feature_df, join_id):
    assert "ver_eff" in df.columns
    windowSpec  = Window.partitionBy(df.columns).orderBy(col("processed_date").desc())

    earning_claim_feature_df = df\
                        .join(feature_df, join_id, "inner")\
                        .filter(col("processed_date") < col("ver_eff"))\
                        .withColumn("row_number", F.row_number().over(windowSpec))\
                        .filter(col("row_number")==1)\
                        .drop("row_number", "processed_date")
    assert earning_claim_feature_df.select(df.columns).distinct().count() <= df.distinct().count()
    assert earning_claim_feature_df.select(df.columns).distinct().count() == earning_claim_feature_df.count()
    print("remain:", earning_claim_feature_df.count() / df.count())
    return earning_claim_feature_df

def join_dln_vin_features(df, dln_df, vin_df, vin_census_df):

    earning_claim_dln_feature_df    = join_feature_latest(df.select(event_keys), dln_df, ["dln"]).cache()

    earning_claim_vin_feature_df    = join_feature_latest(df.select(event_keys), vin_df, ["vin"]).cache()
    
    earning_claim_vin_census_feature_df = join_feature_latest(df.select(event_keys), 
                                            vin_census_df.drop("ver_eff").withColumnRenamed("job_closed_dt", "processed_date"), ["vin"]).cache()
    
    join_manner = "left"
    earning_claim_feature_df = df\
                            .join(earning_claim_dln_feature_df,           event_keys,   join_manner)\
                            .join(earning_claim_vin_feature_df,           event_keys,   join_manner)\
                            .join(earning_claim_vin_census_feature_df,    event_keys,   join_manner)\
                            .drop("vehNum_riskId", "expos_num")\
                            .drop("vehicle_description", "fixed_id", "garage_location_public_id", "rsp_ceded")\
                            .distinct()
    assert earning_claim_feature_df.select(event_keys).distinct().count() == df.select(event_keys).distinct().count()
    assert earning_claim_feature_df.count() == df.count()
    return earning_claim_feature_df

def add_timestamp_features(earning_claim_feature_df):
    ################# transform timestamp features #################
    # ts_cols = ['date_first_license', 'date_of_full_license', 'graduating_licensing_l1_date', 'graduating_licensing_l2_date', 
    #             'date_autoplus_fcsa_ordered', 'date_insured_since', 'date_mvr_ordered', 
    #             'date_insured_by_prior_carrier', 'date_purchased']

    for ii in earning_claim_feature_df.dtypes:
        if ii[1] == "timestamp":
            print("timestamp feature====>", ii)
            earning_claim_feature_df = earning_claim_feature_df\
                                        .withColumn(ii[0]+"_dayssince", F.datediff(col("ver_eff"), col(ii[0])))\
                                        .drop(ii[0])

    ii = 'ver_eff'
    earning_claim_feature_df = earning_claim_feature_df.withColumn(ii+"_dow", F.dayofweek(ii))\
                                                       .withColumn(ii+"_dom", F.dayofmonth(ii))\
                                                       .withColumn(ii+"_month", F.month(ii))
    for c, ty in earning_claim_feature_df.dtypes:
        if ty.startswith("decimal") or ty == "double":
            earning_claim_feature_df = earning_claim_feature_df.withColumn(c, F.col(c).cast(FloatType()))
                                                           
    return earning_claim_feature_df

def assembly(df):
    vin_df, dln_df, vin_census_df = get_vin_dln_census()
    policy_province = map_policy_province().cache() # policy_num - prov_cd, each policy only has one prov_cd

    assert df.count() == df.select(event_keys).distinct().count()
    earning_claim_stats_vindln_df = join_dln_vin_features(df, dln_df, vin_df, vin_census_df).cache()
    assert earning_claim_stats_vindln_df.count() == df.count()
    earning_claim_stats_vindln_prov_df = earning_claim_stats_vindln_df\
                        .join(policy_province, ['policy_num'], "inner")\
                        .withColumn('policy_province' , F.udf(lambda x: 'ATL' if x in ('NB', 'NS', 'PE') else x, StringType())(col("prov_cd") ))\
                        .filter(col("policy_province").isin(['ON', 'AB', 'ATL']))\
                        .drop("prov_cd").cache()
    # timestamp features
    earning_claim_feature_df = add_timestamp_features(earning_claim_stats_vindln_prov_df).distinct().cache()
    return earning_claim_feature_df

df_prem = fetch_earning_unit(CUT_ON_DT, CUT_OFF_DT)
print("df_prem==>",  df_prem.count())
df_prem.agg(F.min("term_eff"), F.max("term_eff"), F.min("ver_eff"), F.max("ver_exp")).show()

# earning join claim
claim_info = fetch_claim_unit(CUT_ON_DT).cache()

def join_earning_claim_dt(df_prem, claim_info):
    # claim_info_agg = claim_info\
    #                             .withColumn("loss_sum",reduce(add, [F.col("{}_loss".format(cvg)) for cvg in CVGS]))\
    #                             .drop(*["{}_loss".format(cvg) for cvg in CVGS])

    df_claim_0 =  df_prem\
            .join(claim_info, 
               (df_prem['policy_num'] == claim_info['policy_num']) &
                (df_prem['vin'] == claim_info['vin']) &
                (df_prem['dln'] == claim_info['dln']) &
                (claim_info['loss_dt'] >= df_prem['ver_eff']) &
                (claim_info['loss_dt'] <= df_prem['ver_exp']), "left")\
                .drop(claim_info['policy_num'])\
                .drop(claim_info['dln'])\
                .drop(claim_info['vin'])\
                .filter(col("loss_dt").isNull())\
                .drop("loss_dt")\
                .withColumnRenamed("ver_eff", "loss_dt")\
                .drop("term_eff", "term_days", "ver_exp", "ver_days", "session_prem")\
                .na.fill(0, loss_cols)

    # df_claim_0.show()
    return claim_info.union(df_claim_0).withColumnRenamed("loss_dt", "ver_eff")

    # df_prem_claim_agg = df_prem\
    #                     .join(claim_info, 
    #                        (df_prem['policy_num'] == claim_info_agg['policy_num']) &
    #                         (df_prem['vin'] == claim_info_agg['vin']) &
    #                         (df_prem['dln'] == claim_info_agg['dln']) &
    #                         (claim_info_agg['loss_dt'] >= df_prem['ver_eff']) &
    #                         (claim_info_agg['loss_dt'] <= df_prem['ver_exp']), "left")\
    #                     .withColumn("loss_sum", F.col("loss_sum").cast(StringType()))\
    #                     .withColumn("loss_dt",  F.col("loss_dt").cast(StringType()))\
    #                     .withColumn("loss_tuple", F.array("loss_dt", "loss_sum"))\
    #                     .drop("loss_dt", "loss_sum", "int_claim_num")\
    #                     .drop(claim_info_agg['policy_num'])\
    #                     .drop(claim_info_agg['dln'])\
    #                     .drop(claim_info_agg['vin'])\
    #                     .groupBy("policy_num", "vin", "dln", "term_eff", "term_days", "ver_eff", "ver_exp", "ver_days", "session_prem")\
    #                     .agg(F.collect_list("loss_tuple").alias("loss_tuples"))\
    #                     .cache()

    # df_claim_0 = df_prem_claim_agg.filter(col("loss_tuples")[0][0].isNull())\
    #                                 .drop("loss_tuples")\
    #                                 .withColumn("session_loss", F.lit(0))

    # df_claim_1 = df_prem_claim_agg.filter(col("loss_tuples")[0][0].isNotNull())\
    #                 .select("policy_num", "vin", "dln", "term_eff", "term_days", "ver_eff", "ver_exp", "ver_days", "session_prem", 
    #                         F.explode("loss_tuples").alias("loss_tuple"))\
    #                 .withColumn("loss_dt", col("loss_tuple")[0])\
    #                 .withColumn("session_loss", col("loss_tuple")[1])\
    #                 .drop("loss_tuple")\
    #                 .drop("ver_eff")\
    #                 .withColumnRenamed("loss_dt", "ver_eff")\
    #                 .drop("ver_days")\
    #                 .withColumn("ver_days", F.lit(1))\
    #                 .drop("ver_exp")\
    #                 .withColumn("ver_exp", col("ver_eff"))

    # earning_claim_df = df_claim_0.union(df_claim_1.select(df_claim_0.columns))\
    #                              .withColumn("session_loss", F.col("session_loss").cast(FloatType()))\
    #                              .withColumn("session_prem", F.col("session_prem").cast(FloatType()))\
    #                              .withColumn("ver_eff", F.to_date("ver_eff"))
    # return earning_claim_df

earning_claim_df = join_earning_claim_dt(df_prem, claim_info).cache()

for cvg in CVGS:
    print(cvg, earning_claim_df.filter(col("{}_loss".format(cvg))>0).count() / earning_claim_df.count())

# earning_claim_df = join_earning_claim(df_prem, claim_info).cache()
assert earning_claim_df.count() ==  earning_claim_df.select(event_keys).distinct().count()
print("earning_claim_df===>", earning_claim_df.count())
earning_claim_df.agg(F.min("ver_eff"), F.max("ver_eff")).show()
print()

earning_claim_feature_df = assembly(earning_claim_df)
print(earning_claim_feature_df.count())
earning_claim_feature_df.agg( F.min("ver_eff"), F.max("ver_eff")).show()


earning_claim_feature_df.write.mode("overwrite")\
        .parquet("gs://definity-aw-prod-dswork-pidc/temp/dynamic_time_frame/df_events")






######################## truncation ###########################
from scipy import stats   
def get_inforce(v):
    sel_cols = ['policy_number', 'veh_vin', 'exposure_number', 'term_eff_dt']
    xgb_cols_dic = {
            'AB': ['xgb_freqxsev_pd', 'xgb_lc_coll', 'xgb_lc_bi', 'xgb_lc_comp', 'xgb_freqxsev_ab'],
            'ATL':['xgb_freqxsev_coll', 'xgb_freqxsev_ab', 'xgb_freqxsev_liab', 'xgb_lc_comp', 'xgb_freqxsev_dc'],
            'ON': ['xgb_lc_coll', 'xgb_freqxsev_liab', 'xgb_freqxsev_dc', 'xgb_freqxsev_ab', 'xgb_freqxsev_comp'],
            'QC': ['xgb_lc_comp', 'xgb_lc_coll', 'xgb_lc_liab']}
    df_raw = spark.read.option("delimiter", ",")\
                .csv("gs://definity-aw-prod-dswork-pidc/temp/reference_csv/inforce_ppv_on_{}.csv".format(v), header=True)
    df_inforce = df_raw\
                .withColumnRenamed("veh_vin", "vin")\
                .withColumnRenamed("policy_number", "policy_num")\
                .withColumnRenamed("term_eff_dt", "term_eff")\
                .withColumnRenamed("ver_eff_dt", "ver_eff")\
                .withColumn("exposure_number", F.col("exposure_number").cast(IntegerType()))\
                .withColumnRenamed("drv_license_number", "dln")\
                .withColumn("mpa_score", reduce(add, [F.col(x) for x in xgb_cols_dic['ON']]))\
                .drop(*xgb_cols_dic['ON'])\
                .select('vin', 'policy_num', 'term_eff', 'dln', 'ver_eff', 'mpa_score')\
                .withColumn("month", F.lit(v))\
                .drop("month", "term_eff", "ver_eff")\
                .distinct()
    return df_inforce

def truncation_prem(df_prem, trunc_on, trunc_off):
    
    df_1 = df_prem\
            .filter(trunc_on <= col("ver_eff"))\
            .filter(col("ver_exp") <=  trunc_off)\
            .withColumn("overlap_days", col("ver_days"))\
            .withColumn("prem_segmented_sub", col("session_prem") )

    df_0 = df_prem\
            .filter(col("ver_eff") < trunc_on)\
            .filter( (col("ver_exp") >= trunc_on) & (col("ver_exp") <= trunc_off)  )\
            .withColumn("overlap_days", F.datediff(col("ver_exp"), F.lit(trunc_on)))\
            .withColumn("prem_segmented_sub", col("session_prem") / col("ver_days") * col("overlap_days") )

    df_2 = df_prem\
            .filter(col("ver_exp") > trunc_off)\
            .filter( (col("ver_eff")>=trunc_on) & (col("ver_eff") <= trunc_off)  )\
            .withColumn("overlap_days", F.datediff(F.lit(trunc_off), col("ver_eff")))\
            .withColumn("prem_segmented_sub", col("session_prem") / col("ver_days") * col("overlap_days") )

    df_9 = df_prem\
            .filter( (col("ver_eff") < trunc_on) & (col("ver_exp") > trunc_off)  )\
            .withColumn("overlap_days", F.datediff(F.lit(trunc_off), F.lit(trunc_on)))\
            .withColumn("prem_segmented_sub", col("session_prem") / col("ver_days") * col("overlap_days") )

    
    dfu = df_1.union(df_0).union(df_2).union(df_9)

    assert dfu.select(['policy_num',"vin", 'term_eff', 'term_days', 'ver_eff', 'ver_exp', 'ver_days']).distinct().count() \
                == dfu.count()

    return dfu

df_prem = fetch_earning_unit("2021-01-01", "2022-12-31")
claim_info = fetch_claim_unit("2022-01-01").cache()

for inforcem in ["202112", "202201", "202202", "202203", "202204", "202205", "202206", "202207"]:
    df_inforce = get_inforce(inforcem)  
    trunc_on = (datetime.datetime.strptime(inforcem, "%Y%m") + dateutil.relativedelta.relativedelta(months=1) ).strftime('%Y-%m-%d') 
    print(inforcem, trunc_on, df_inforce.count())

    for infore_month in (1, 3, 6):
        trunc_off = trunc_off = (datetime.datetime.strptime(trunc_on, "%Y-%m-%d") + dateutil.relativedelta.relativedelta(months=infore_month) \
                                - dateutil.relativedelta.relativedelta(days=1) ).strftime('%Y-%m-%d')
        dfu = truncation_prem(df_prem, trunc_on, trunc_off)

        df_inforce_trunc_claim =  df_inforce\
                            .join(claim_info.filter(col("loss_dt") >= trunc_on).filter(col("loss_dt") <= trunc_off),
                                     ['policy_num', 'vin', 'dln'], "left")\
                            .na.fill(0, subset=loss_cols)\
                            .groupBy(["vin", "dln", "policy_num", "mpa_score"])\
                            .agg(F.sum("DC_loss").alias("DC_loss"),
                                 F.sum("COMPSP_loss").alias("COMPSP_loss"),
                                 F.sum("APCOLL_loss").alias("APCOLL_loss"),
                                 F.sum("TPL_loss").alias("TPL_loss"),
                                 F.sum("AB_loss").alias("AB_loss"))

        assert df_inforce_trunc_claim.count() == df_inforce.count()

        df_inforce_trunc_claim_prem = df_inforce_trunc_claim.join(
                                dfu.groupBy("policy_num", "vin", "dln").agg(F.sum("prem_segmented_sub").alias("prem")),
                                 ["policy_num", "vin", "dln"], "inner")\
                                .drop("ver_eff")\
                                .withColumnRenamed("prem", "prem_{}".format(infore_month))\
                                .withColumnRenamed("DC_loss", "DC_loss_{}".format(infore_month))\
                                .withColumnRenamed("COMPSP_loss", "COMPSP_loss_{}".format(infore_month))\
                                .withColumnRenamed("APCOLL_loss", "APCOLL_loss_{}".format(infore_month))\
                                .withColumnRenamed("TPL_loss", "TPL_loss_{}".format(infore_month))\
                                .withColumnRenamed("AB_loss", "AB_loss_{}".format(infore_month))\
                                .cache()

        df_inforce = df_inforce.join(df_inforce_trunc_claim_prem, ["policy_num", "vin", "dln", "mpa_score"], "inner") 

        percentile = df_inforce.select(F.percentile_approx("mpa_score", [0.95])).collect()[0][0][0]

        dfm_top = df_inforce.filter(col("mpa_score") >= percentile)
        dfm_bottom = df_inforce.filter(col("mpa_score") < percentile)

        cvg_loss_cols = [col("{}_loss_{}".format(cvg, infore_month)) for cvg in CVGS]

        print(infore_month, 
              dfm_top.withColumn("loss_sum", reduce(add, cvg_loss_cols)).agg(F.sum("loss_sum")).collect()[0][0] / dfm_top.agg(F.sum("prem_{}".format(infore_month))).collect()[0][0],
              dfm_bottom.withColumn("loss_sum", reduce(add, cvg_loss_cols)).agg(F.sum("loss_sum")).collect()[0][0] / dfm_bottom.agg(F.sum("prem_{}".format(infore_month))).collect()[0][0] )



    df_inforce_infer = assembly(df_inforce.withColumn("ver_eff", F.to_date(F.lit(trunc_on))) )

    assert df_inforce_infer.count() == df_inforce_infer.select(event_keys).distinct().count()
    df_inforce_infer.show()
    df_inforce_infer.write.mode("overwrite")\
        .parquet("gs://definity-aw-prod-dswork-pidc/temp/dynamic_time_frame/df_inforce_infer_{}".format(trunc_on))













    # # df_mpa_loss_ = pd.DataFrame( [(ii['mpa_score'], ii['loss']) for ii in claim_df_trunc.select("mpa_score", "loss").collect()],
    # #                             columns=['mpa_score','loss'])

    # df_mpa_loss = pd.DataFrame( [(ii['mpa_score'], ii['loss']) for ii in earning_claim_df_trunc_inforce.select("mpa_score", "loss").collect()],
    #                             columns=['mpa_score','loss'])

    # print('spearmanr:', stats.spearmanr(df_mpa_loss['mpa_score'].values,  df_mpa_loss['loss'].values).correlation)

    # earning_claim_df_infer = earning_claim_df_trunc_inforce.drop("ver_eff", "ver_exp", "session_prem")\
    #                                             .withColumnRenamed("trunc_on", "ver_eff")\
    #                                             .withColumnRenamed("prem", "session_prem")\
    #                                             .withColumnRenamed("loss", "session_loss")\
    #                                             .drop("mpa_score", "month")

    # earning_claim_df_infer.agg(F.min("session_loss"), F.max("session_loss"), F.mean("session_loss"), \
    #                            F.min("session_prem"), F.max("session_prem"), F.mean("session_prem")).show()

    # earning_claim_stats_df, stats_cols = append_stats_features_as_of_03(earning_claim_df_infer )










# nacheck(earning_claim_feature_df)

##################################################################################

