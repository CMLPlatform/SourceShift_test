# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:28:42 2018

@author: bfdeboer
"""

import csv

import cfg

def get_cf(file_path, df_cQ):
    list_imp = []
    with open(file_path) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_imp.append(tuple(row))
    return df_cQ.loc[list_imp]


def get_dict_cf():
    dict_cf = {}
    dict_cf['e'] = get_cf(cfg.data_path+cfg.e_fp_file_name,
                          dict_eb['cQe'])
    dict_cf['m'] = get_cf(cfg.data_path+cfg.m_fp_file_name,
                          dict_eb['cQm'])
    dict_cf['r'] = get_cf(cfg.data_path+cfg.r_fp_file_name,
                          dict_eb['cQr'])
    return dict_cf


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
