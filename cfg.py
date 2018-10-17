# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:43:27 2018

@author: bfdeboer
"""
import datetime
import os

date_full = datetime.datetime.now()
date = '{}{:02}{:02}'.format(date_full.year, date_full.month, date_full.day)

#   Make result directories.
method = '_source_shift/'
result_dir_path = 'result/'+date+method
priority_setting_dir_name = '1_priority_setting/'
imp_pME__vs__y_dir_name = '2_imp_pME__vs__y/'
improv_dir_name ='3_pot_improv/'
improv_agg_dir_name='4_pot_improv_agg/'

list_dir_name = []
list_dir_name.append(priority_setting_dir_name)
list_dir_name.append(imp_pME__vs__y_dir_name)
list_dir_name.append(improv_agg_dir_name)
list_dir_name.append(improv_dir_name)

pdf_dir_name = 'pdf/'
png_dir_name = 'png/'

for dir_name in list_dir_name:
    dir_path = result_dir_path+dir_name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path+pdf_dir_name)
        os.makedirs(dir_path+png_dir_name)
#
#if not os.path.exists(dir_result):
#    os.makedirs(dir_result)
#
#dir_result_contribution_analysis = dir_result+'contribution_analysis/'
#if not os.path.exists(dir_result_contribution_analysis):
#    os.makedirs(dir_result_contribution_analysis)
#
#dir_result_contribution_analysis_pdf = dir_result_contribution_analysis+'pdf/'
#if not os.path.exists(dir_result_contribution_analysis_pdf):
#    os.makedirs(dir_result_contribution_analysis_pdf)
#
#dir_result_contribution_analysis_png = dir_result_contribution_analysis+'png/'
#if not os.path.exists(dir_result_contribution_analysis_png):
#    os.makedirs(dir_result_contribution_analysis_png)
#
#dir_result_imp_pME__vs__y = dir_result+'imp_pME__vs__y/'
#if not os.path.exists(dir_result_imp_pME__vs__y):
#    os.makedirs(dir_result_imp_pME__vs__y)
#
#dir_result_imp_pME__vs__y_pdf = dir_result_imp_pME__vs__y+'pdf/'
#if not os.path.exists(dir_result_imp_pME__vs__y_pdf):
#    os.makedirs(dir_result_imp_pME__vs__y_pdf)
#
#dir_result_imp_pME__vs__y_png = dir_result_imp_pME__vs__y+'png/'
#if not os.path.exists(dir_result_imp_pME__vs__y_png):
#    os.makedirs(dir_result_imp_pME__vs__y_png)
#
#dir_result_pot_improv_prod = dir_result+'potential_improvement/'
#if not os.path.exists(dir_result_pot_improv_prod):
#    os.makedirs(dir_result_pot_improv_prod)
#
#dir_result_pot_improv_prod_pdf = dir_result_pot_improv_prod+'pdf/'
#if not os.path.exists(dir_result_pot_improv_prod_pdf):
#    os.makedirs(dir_result_pot_improv_prod_pdf)
#
#dir_result_pot_improv_prod_png = dir_result_pot_improv_prod+'png/'
#if not os.path.exists(dir_result_pot_improv_prod_png):
#    os.makedirs(dir_result_pot_improv_prod_png)
#
#dir_result_pot_improv_prod_agg = dir_result+'potential_improvement_agg/'
#if not os.path.exists(dir_result_pot_improv_prod_agg):
#    os.makedirs(dir_result_pot_improv_prod_agg)
#
#dir_result_pot_improv_prod_agg_pdf = dir_result_pot_improv_prod_agg+'pdf/'
#if not os.path.exists(dir_result_pot_improv_prod_agg_pdf):
#    os.makedirs(dir_result_pot_improv_prod_agg_pdf)
#
#dir_result_pot_improv_prod_agg_png = dir_result_pot_improv_prod_agg+'png/'
#if not os.path.exists(dir_result_pot_improv_prod_agg_png):
#    os.makedirs(dir_result_pot_improv_prod_agg_png)
#
#
##   Make result directories.
#method = 'consolidated_priority_setting/'
#dir_result = 'result/'+date+method
#
#if not os.path.exists(dir_result):
#    os.makedirs(dir_result)
#
#dir_result_contribution_analysis = dir_result+'contribution_analysis/'
#if not os.path.exists(dir_result_contribution_analysis):
#    os.makedirs(dir_result_contribution_analysis)
#
#dir_result_contribution_analysis_pdf = dir_result_contribution_analysis+'pdf/'
#if not os.path.exists(dir_result_contribution_analysis_pdf):
#    os.makedirs(dir_result_contribution_analysis_pdf)
#
#dir_result_contribution_analysis_png = dir_result_contribution_analysis+'png/'
#if not os.path.exists(dir_result_contribution_analysis_png):
#    os.makedirs(dir_result_contribution_analysis_png)
#
#dir_result_contribution_analysis_agg = dir_result+'contribution_analysis_agg/'
#if not os.path.exists(dir_result_contribution_analysis_agg):
#    os.makedirs(dir_result_contribution_analysis_agg)
#
#dir_result_contribution_analysis_agg_pdf = (
#        dir_result_contribution_analysis_agg+'pdf/')
#if not os.path.exists(dir_result_contribution_analysis_agg_pdf):
#    os.makedirs(dir_result_contribution_analysis_agg_pdf)
#
#dir_result_contribution_analysis_agg_png = (
#        dir_result_contribution_analysis_agg+'png/')
#if not os.path.exists(dir_result_contribution_analysis_agg_png):
#    os.makedirs(dir_result_contribution_analysis_agg_png)
#
#dir_result_imp_pME__vs__y = dir_result+'imp_pME__vs__y/'
#if not os.path.exists(dir_result_imp_pME__vs__y):
#    os.makedirs(dir_result_imp_pME__vs__y)
#
#dir_result_imp_pME__vs__y_pdf = dir_result_imp_pME__vs__y+'pdf/'
#if not os.path.exists(dir_result_imp_pME__vs__y_pdf):
#    os.makedirs(dir_result_imp_pME__vs__y_pdf)
#
#dir_result_imp_pME__vs__y_png = dir_result_imp_pME__vs__y+'png/'
#if not os.path.exists(dir_result_imp_pME__vs__y_png):
#    os.makedirs(dir_result_imp_pME__vs__y_png)
#
#dir_result_pot_improv_prod = dir_result+'potential_improvement/'
#if not os.path.exists(dir_result_pot_improv_prod):
#    os.makedirs(dir_result_pot_improv_prod)
#
#dir_result_pot_improv_prod_pdf = dir_result_pot_improv_prod+'pdf/'
#if not os.path.exists(dir_result_pot_improv_prod_pdf):
#    os.makedirs(dir_result_pot_improv_prod_pdf)
#
#dir_result_pot_improv_prod_png = dir_result_pot_improv_prod+'png/'
#if not os.path.exists(dir_result_pot_improv_prod_png):
#    os.makedirs(dir_result_pot_improv_prod_png)
#
#dir_result_pot_improv_prod_agg = dir_result+'potential_improvement_agg/'
#if not os.path.exists(dir_result_pot_improv_prod_agg):
#    os.makedirs(dir_result_pot_improv_prod_agg)
#
#dir_result_pot_improv_prod_agg_pdf = dir_result_pot_improv_prod_agg+'pdf/'
#if not os.path.exists(dir_result_pot_improv_prod_agg_pdf):
#    os.makedirs(dir_result_pot_improv_prod_agg_pdf)
#
#dir_result_pot_improv_prod_agg_png = dir_result_pot_improv_prod_agg+'png/'
#if not os.path.exists(dir_result_pot_improv_prod_agg_png):
#    os.makedirs(dir_result_pot_improv_prod_agg_png)
