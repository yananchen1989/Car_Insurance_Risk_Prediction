scored_inforce_mpa <- fread("/rdev/MPA/PI_Pricing_Datasets/PPV/ATL/Inforce/inforce_ppv_atl_202303.csv") %>% mutate(term_eff_dt = as.Date(term_eff_dt),

ver_eff_dt = as.Date(ver_eff_dt))


scored_inforce_final <- scored_inforce_mpa %>%
rename(xgb_freqxsev_comp = xgb_lc_comp) %>%

mutate(TotalTheoLC = rowSums(select(., matches('^xgb_')), na.rm = T),

term_eff_dt = as.Date(term_eff_dt)) %>%

select(policy_number, veh_vin, exposure_number, term_eff_dt, ver_eff_dt, matches('^xgb_'), TotalTheoLC)%>% select(-ends_with(c("raw")))






 #general inforce load

 inforce_rsp <-fread(paste0("./", prov, "_", comp, "_", inforce_date, "_Inforce.csv"), blank.lines.skip=TRUE)


 ceding_constants <- fread(paste0("./RSP_ceding_constants_2022.csv")) %>%

 filter(Province == prov & Company == comp)


 cession_percent = ceding_constants$cession_percent

 annual_pool_cost = ceding_constants$annual_pool_cost

 expense_allowance = ceding_constants$expense_allowance



 inforce_rsp_final <- inforce_rsp %>%

 mutate(TotalChargedPrem = rowSums(select(., matches('^(ab|ap|liab|dc|coll|comp)_prem$')), na.rm = T),

 term_eff_dt = as.Date(term_eff_dt)) %>%

 select(policy_number, vin, exposure_number, term_eff_dt, ver_eff_dt, matches('^(ab|ap|liab|dc|coll|comp)_prem$'), TotalChargedPrem, comprehensive_cov_only, endt_6a)


 # Join MPA and RSP inforce

 inforce_final <- scored_inforce_final %>%

 inner_join(inforce_rsp_final, by = c("policy_number", "veh_vin" = "vin", "exposure_number", "term_eff_dt", 'ver_eff_dt')) %>%

 distinct(policy_number,veh_vin,exposure_number,term_eff_dt, .keep_all = TRUE)



 inforce_final <-inforce_final %>%

 distinct(policy_number, veh_vin, exposure_number, term_eff_dt, .keep_all = T) %>%

 filter(comprehensive_cov_only != TRUE) %>%

 filter(endt_6a == "N") %>%

 select(policy_number, veh_vin, term_eff_dt, matches('^xgb_'), matches('^(ab|ap|liab|dc|coll|comp)_prem$'), TotalTheoLC, TotalChargedPrem) %>%

 group_by(policy_number, veh_vin, term_eff_dt) %>%

 summarise_all(.funs = c(sum="sum")) %>%

 ungroup() %>%

 mutate(exp_savings = round(cession_percent * (TotalTheoLC_sum - (1 - expense_allowance) * TotalChargedPrem_sum - annual_pool_cost))

 # cede_decision = if_else(exp_savings > 100, "Y", "N")

 )


