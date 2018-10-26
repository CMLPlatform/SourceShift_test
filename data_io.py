# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:28:42 2018

@author: bfdeboer
"""
from collections import OrderedDict

import csv
import numpy as np
import operator
import os
import pandas as pd
import pickle

import cfg
import exiobase as eb


def get_dict_df_imp(dict_cf, dict_eb, df_tY):
    #   Diagonalize final demand.
    ar_tY = np.diag(df_tY)
    df_tYd = pd.DataFrame(ar_tY, index=df_tY.index, columns=df_tY.index)

    #   Calculate absolute impact of imported products to EU28.
    df_cQe = dict_cf['e']
    df_cQm = dict_cf['m']
    df_cQr = dict_cf['r']

    df_cRe = dict_eb['cRe']
    df_cRm = dict_eb['cRm']
    df_cRr = dict_eb['cRr']

    df_cL = dict_eb['cL']

    dict_df_imp = {}
    dict_df_imp['e'] = df_cQe.dot(df_cRe).dot(df_cL).dot(df_tYd)
    dict_df_imp['m'] = df_cQm.dot(df_cRm).dot(df_cL).dot(df_tYd)
    dict_df_imp['r'] = df_cQr.dot(df_cRr).dot(df_cL).dot(df_tYd)

    return dict_df_imp


def get_dict_imp(dict_df_imp):
    dict_imp = {}
    for cat in dict_df_imp:
        dict_df = dict_df_imp[cat].T.to_dict()
        for imp_cat in dict_df:
            dict_imp[imp_cat] = {}
            for tup_prod_cntr in dict_df[imp_cat]:
                cntr, prod = tup_prod_cntr
                if prod not in dict_imp[imp_cat]:
                    dict_imp[imp_cat][prod] = {}
                dict_imp[imp_cat][prod][cntr] = dict_df[imp_cat][tup_prod_cntr]
    return dict_imp


def get_dict_imp_cat_unit():
    dict_imp_cat_unit = {}
    dict_imp_cat_unit['kg CO2 eq.'] = r'$Pg\/CO_2\/eq.$'
    dict_imp_cat_unit['kt'] = r'$Gt$'
    dict_imp_cat_unit['Mm3'] = r'$Mm^3$'
    dict_imp_cat_unit['km2'] = r'$Gm^2$'
    return dict_imp_cat_unit


def get_dict_imp_prod_sort(dict_df_imp, imp_cum_lim):
    dict_imp_prod_sort = {}
    for cat in dict_df_imp:
        df_imp_prod = dict_df_imp[cat].sum(axis=1, level=1)

        dict_imp_prod = df_imp_prod.T.to_dict()
        dict_imp_prod_sum = dict_df_imp[cat].sum(axis=1).to_dict()
        dict_imp_prod_sort_cat = OrderedDict()
        for imp_cat in dict_imp_prod:
            dict_imp_prod_sort_cat[imp_cat] = OrderedDict()
            list_imp_sort = sorted(dict_imp_prod[imp_cat].items(),
                                   key=operator.itemgetter(1), reverse=True)
            imp_cum = 0
            bool_add = True
            for tup_prod_abs_id, tup_prod_abs in enumerate(list_imp_sort):
                (prod, imp_abs) = tup_prod_abs
                imp_rel = imp_abs/dict_imp_prod_sum[imp_cat]
                imp_cum = imp_cum + imp_rel
                if imp_cum < imp_cum_lim:
                    dict_imp_prod_sort_cat[imp_cat][prod] = imp_abs
                elif bool_add:
                    dict_imp_prod_sort_cat[imp_cat][prod] = imp_abs
                    bool_add = False

        for imp_cat in dict_imp_prod_sort_cat:
            dict_imp_prod_sort[imp_cat] = dict_imp_prod_sort_cat[imp_cat]

    return dict_imp_prod_sort


def get_cf(file_path, df_cQ):
    list_imp = []
    with open(file_path) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_imp.append(tuple(row))
    return df_cQ.loc[list_imp]


def get_dict_cf(dict_eb):
    dict_cf = {}
    dict_cf['e'] = get_cf(cfg.data_path+cfg.e_fp_file_name,
                          dict_eb['cQe'])
    dict_cf['m'] = get_cf(cfg.data_path+cfg.m_fp_file_name,
                          dict_eb['cQm'])
    dict_cf['r'] = get_cf(cfg.data_path+cfg.r_fp_file_name,
                          dict_eb['cQr'])
    return dict_cf


def get_dict_eb():
    if cfg.dict_eb_file_name in os.listdir(cfg.data_path):
        dict_eb = pickle.load(open(cfg.data_path+cfg.dict_eb_file_name, 'rb'))
    else:
        dict_eb = eb.process(eb.parse())
        save_eb = True
        if save_eb:
            pickle.dump(dict_eb, open(cfg.data_path+cfg.dict_eb_file_name, 'wb'))
    return dict_eb


def get_dict_prod_long_short():
    list_prod_long = []
    with open(cfg.data_path+cfg.prod_long_file_name) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_prod_long.append(row[0])

    list_prod_short = []
    with open(cfg.data_path+cfg.prod_short_file_name) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_prod_short.append(row[0])

    dict_prod_long_short = {}
    for prod_id, prod_long in enumerate(list_prod_long):
        prod_long = list_prod_long[prod_id]
        prod_short = list_prod_short[prod_id]
        dict_prod_long_short[prod_long] = prod_short
    return dict_prod_long_short


def get_dict_cntr_short_long():
    dict_cntr_short_long = {}
    with open(cfg.data_path+cfg.country_code_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            cntr_short = row[0]
            cntr_long = row[1]
            dict_cntr_short_long[cntr_short] = cntr_long
    return dict_cntr_short_long

def get_dict_imp_cat_fp():
    dict_imp_cat_fp = {}
    with open(cfg.data_path+cfg.cf_long_footprint_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            imp_cat = tuple(row[:-1])
            fp = row[-1]
            dict_imp_cat_fp[imp_cat] = fp
    return dict_imp_cat_fp


def get_dict_imp_cat_magnitude():
    dict_imp_cat_magnitude = {}
    with open(cfg.data_path+cfg.cf_magnitude_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            imp_cat = tuple(row[:-1])
            magnitude = int(row[-1])
            dict_imp_cat_magnitude[imp_cat] = magnitude
    return dict_imp_cat_magnitude


def get_list_prod_order_cons():
    list_prod_order_cons = []
    with open(cfg.data_path+cfg.prod_order_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_prod_order_cons.append(row[0])
    list_prod_order_cons.reverse()
    return list_prod_order_cons


def get_list_reg_fd(reg_fd):
    if reg_fd == 'EU28':
        with open(cfg.data_path+cfg.eu28_file_name) as read_file:
            csv_file = csv.reader(read_file, delimiter='\t')
            list_reg_fd = []
            for row in csv_file:
                list_reg_fd.append(row[0])
    else:
        list_reg_fd = [reg_fd]
    return list_reg_fd


def make_result_dir():
    list_dir_name = []
    list_dir_name.append(cfg.priority_setting_dir_name)
    list_dir_name.append(cfg.shift_dir_name)
    list_dir_name.append(cfg.reduc_agg_dir_name)
    list_dir_name.append(cfg.reduc_dir_name)

    for dir_name in list_dir_name:
        if not os.path.exists(cfg.result_dir_path+dir_name):
            os.makedirs(cfg.result_dir_path+dir_name+cfg.pdf_dir_name)
            os.makedirs(cfg.result_dir_path+dir_name+cfg.png_dir_name)
