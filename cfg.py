# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:30:38 2018

@author: bfdeboer
"""

import datetime
import matplotlib.pyplot as plt

def get_date():
    date = datetime.datetime.now()
    return '{}{:02}{:02}'.format(date.year, date.month, date.day)

date = get_date()

method = '_source_shift/'
result_dir_path = 'result/'+date+method
priority_setting_dir_name = '1_priority_setting/'
shift_dir_name = '2_shift/'
reduc_dir_name = '3_reduction/'
reduc_agg_dir_name = '4_reduction_agg/'

pdf_dir_name = 'pdf/'
png_dir_name = 'png/'
txt_dir_name = 'txt/'
list_output_dir_name = [pdf_dir_name, png_dir_name, txt_dir_name]

data_path = 'data/'
eu28_file_name = 'EU28.txt'
e_fp_file_name = 'list_impact_emission.txt'
m_fp_file_name = 'list_impact_material.txt'
r_fp_file_name = 'list_impact_resource.txt'
country_code_file_name = 'country_codes.txt'
prod_long_file_name = 'prod_long.txt'
prod_short_file_name = 'prod_short.txt'
dict_eb_file_name = 'dict_eb_proc.pkl'
cf_long_footprint_file_name = 'cf_long_footprint.txt'
cf_magnitude_file_name = 'cf_magnitude.txt'
prod_order_file_name = 'prod_order.txt'

font_size = 8.0
plt.rc('mathtext', default = 'regular')
plt.rc('font', size = font_size)
plt.rc('axes', titlesize = font_size)

imp_cum_lim_priority = 0.5
imp_cum_lim_full = 1.1

imp_cum_lim_source_shift = 0.5

reduc_alpha = 0.5

x_prod_cntr_min = 0.5
