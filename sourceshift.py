# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:36:39 2018

@author: boerbfde
"""
from collections import OrderedDict
import csv
import math
import matplotlib as mpl
import matplotlib.collections as mpl_col
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pandas as pd
import pickle
import datetime

import eb

date_full = datetime.datetime.now()
date = '{}{:02}{:02}'.format(date_full.year, date_full.month, date_full.day)

#   Make result directories.
method = '_source_shift/'
result_dir_path = 'result/'+date+method
priority_setting_dir_name = '1_priority_setting/'
shift_dir_name = '2_shift/'
reduc_dir_name ='3_reduction/'
reduc_agg_dir_name='4_reduction_agg/'

list_dir_name = []
list_dir_name.append(priority_setting_dir_name)
list_dir_name.append(shift_dir_name)
list_dir_name.append(reduc_agg_dir_name)
list_dir_name.append(reduc_dir_name)

pdf_dir_name = 'pdf/'
png_dir_name = 'png/'

for dir_name in list_dir_name:
    dir_path = result_dir_path+dir_name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path+pdf_dir_name)
        os.makedirs(dir_path+png_dir_name)

mpl.rcParams['mathtext.default'] = 'regular'

data_path = 'data/'

q_e_file_name = 'q_emission.txt'
q_m_file_name = 'q_material.txt'
q_r_file_name = 'q_resource.txt'
q_e_file_path = data_path+q_e_file_name
q_m_file_path = data_path+q_m_file_name
q_r_file_path = data_path+q_r_file_name

eu28_file_name = 'EU28.txt'
eu28_file_path = data_path+eu28_file_name

list_e_fp_file_name = 'list_impact_emission.txt'
list_m_fp_file_name = 'list_impact_material.txt'
list_r_fp_file_name = 'list_impact_resource.txt'
list_e_fp_file_path = data_path+list_e_fp_file_name
list_m_fp_file_path = data_path+list_m_fp_file_name
list_r_fp_file_path = data_path+list_r_fp_file_name

country_code_file_name = 'country_codes.txt'
country_code_file_path = data_path+country_code_file_name

prod_long_file_name = 'prod_long.txt'
prod_long_file_path = data_path+prod_long_file_name

prod_short_file_name = 'prod_short.txt'
prod_short_file_path = data_path+prod_short_file_name

dict_eb_file_name = 'dict_eb_proc.pkl'
dict_eb_file_path = data_path+dict_eb_file_name

cf_long_footprint_file_name = 'cf_long_footprint.txt'
cf_long_footprint_file_path = data_path+cf_long_footprint_file_name

cf_magnitude_file_name = 'cf_magnitude.txt'
cf_magnitude_file_path = data_path+cf_magnitude_file_name

prod_order_file_name = 'prod_order.txt'
prod_order_file_path = data_path+prod_order_file_name

def cm2inch(tup_cm):
    inch = 2.54
    tup_inch = tuple(i/inch for i in tup_cm)
    return tup_inch


def get_reg_fd(reg_fd):
    if reg_fd == 'EU28':
        with open(eu28_file_path) as read_file:
            csv_file = csv.reader(read_file, delimiter='\t')
            list_reg_fd = []
            for row in csv_file:
                list_reg_fd.append(row[0])
    else:
        list_reg_fd = [reg_fd]
    dict_reg = {}
    dict_reg[reg_fd] = list_reg_fd
    return dict_reg


def get_dict_imp_prod_sort_cat(df_imp, df_imp_full, df_tY, imp_cum_lim):
    df_tY_prod = df_tY.sum(axis=0, level=1)
    df_imp_prod = df_imp.sum(axis=1, level=1)

    dict_imp_prod = df_imp_prod.T.to_dict()
    dict_imp_prod_sum = df_imp.sum(axis=1).to_dict()
    dict_imp_prod_sort = OrderedDict()
    for imp_cat in dict_imp_prod:
        dict_imp_prod_sort[imp_cat] = OrderedDict()
        list_imp_sort = sorted(dict_imp_prod[imp_cat].items(),
                               key=operator.itemgetter(1), reverse=True)
        imp_cum = 0
        bool_add = True
        for tup_prod_abs_id, tup_prod_abs in enumerate(list_imp_sort):
            (prod, imp_abs) = tup_prod_abs
            imp_rel = imp_abs/dict_imp_prod_sum[imp_cat]
            imp_cum = imp_cum + imp_rel
            y = df_tY_prod[prod]
            if imp_cum < imp_cum_lim:
                dict_imp_prod_sort[imp_cat][prod] = OrderedDict()
                dict_imp_prod_sort[imp_cat][prod]['imp_abs'] = imp_abs
                dict_imp_prod_sort[imp_cat][prod]['imp_rel'] = imp_rel
                dict_imp_prod_sort[imp_cat][prod]['imp_cum'] = imp_cum
                dict_imp_prod_sort[imp_cat][prod]['y'] = y
            elif bool_add:
                dict_imp_prod_sort[imp_cat][prod] = OrderedDict()
                dict_imp_prod_sort[imp_cat][prod]['imp_abs'] = imp_abs
                dict_imp_prod_sort[imp_cat][prod]['imp_rel'] = imp_rel
                dict_imp_prod_sort[imp_cat][prod]['imp_cum'] = imp_cum
                dict_imp_prod_sort[imp_cat][prod]['y'] = y
                bool_add = False
    return dict_imp_prod_sort


def get_cf(file_path, df_cQ):
    list_imp = []
    with open(file_path) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_imp.append(tuple(row))
    return df_cQ.loc[list_imp]


def get_dict_cf():
    dict_cf = {}
    dict_cf['e'] = get_cf(list_e_fp_file_path,
                          dict_eb['cQe'])
    dict_cf['m'] = get_cf(list_m_fp_file_path,
                          dict_eb['cQm'])
    dict_cf['r'] = get_cf(list_r_fp_file_path,
                          dict_eb['cQr'])
    return dict_cf


def get_dict_df_imp(dict_eb, df_tY, dict_cf):
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


def get_dict_imp_prod_sort(df_tY, dict_imp, dict_df_imp_full, imp_cum_lim):
    dict_imp_prod_sort = {}
    for cat in dict_imp:
        dict_imp_prod_sort_cat = (
                get_dict_imp_prod_sort_cat(dict_imp[cat],
                                           dict_df_imp_full[cat],
                                           df_tY,
                                           imp_cum_lim))
        for imp_cat in dict_imp_prod_sort_cat:
            dict_imp_prod_sort[imp_cat] = dict_imp_prod_sort_cat[imp_cat]

    return dict_imp_prod_sort


def get_dict_imp_prod_sort_cons(dict_imp_prod_sort, dict_imp_prod):
    list_prod_full = []
    for imp_cat in dict_imp_prod_sort:
        list_prod_imp_cat = list(dict_imp_prod_sort[imp_cat].keys())
        for prod in list_prod_imp_cat:
            if prod not in list_prod_full:
                list_prod_full.append(prod)

    dict_imp_prod_sort_cons = {}
    for cat in dict_df_imp:
        df_imp = dict_df_imp[cat]
        df_imp_prod = df_imp.sum(axis=1, level=1)
        dict_imp_prod = df_imp_prod.T.to_dict()
        for imp_cat in dict_imp_prod:
            dict_imp_prod_sort_cons[imp_cat] = {}
            for prod in list_prod_full:
                imp_abs = dict_imp_prod[imp_cat][prod]
                dict_imp_prod_sort_cons[imp_cat][prod] = {}
                dict_imp_prod_sort_cons[imp_cat][prod]['imp_abs'] = imp_abs
    return dict_imp_prod_sort_cons


def get_dict_imp_sel_reg(dict_imp_sel):
    reg_dom = 'domestic'
    reg_import = 'import'
    dict_imp_sel_reg = {}
    for cntr_fd in dict_imp_sel:
        for imp_cat_sel in dict_imp_sel[cntr_fd]:
            if imp_cat_sel not in dict_imp_sel_reg:
                dict_imp_sel_reg[imp_cat_sel] = {}
            for imp_cat_eff in dict_imp_sel[cntr_fd][imp_cat_sel]:
                if imp_cat_eff not in dict_imp_sel_reg[imp_cat_sel]:
                    dict_imp_sel_reg[imp_cat_sel][imp_cat_eff] = {}
                for prod_long in dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff]:
                    prod_short = dict_prod_long_short[prod_long]
                    if prod_short not in (
                            dict_imp_sel_reg[imp_cat_sel][imp_cat_eff]):
                        dict_imp_sel_reg[imp_cat_sel][imp_cat_eff][prod_short] = {}
                        dict_imp_sel_reg[imp_cat_sel][imp_cat_eff][prod_short][reg_dom] = 0
                        dict_imp_sel_reg[imp_cat_sel][imp_cat_eff][prod_short][reg_import] = 0
                    for cntr in dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod_long]:
                        imp_abs = dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod_long][cntr]
                        if cntr_fd == cntr:
                            reg = reg_dom
                        else:
                            reg = reg_import
                        dict_imp_sel_reg[imp_cat_sel][imp_cat_eff][prod_short][reg] += (
                                imp_abs)
    return dict_imp_sel_reg

def get_dict_imp_sel_reg_cons(dict_imp_sel_cons):
    dict_imp_sel_reg = get_dict_imp_sel_reg(dict_imp_sel_cons)

    dict_imp_sel_reg_cons = {}
    for imp_cat_sel in dict_imp_sel_reg:
        for imp_cat_eff in dict_imp_sel_reg[imp_cat_sel]:
            if imp_cat_eff not in dict_imp_sel_reg_cons:
                dict_imp_sel_reg_cons[imp_cat_eff] = {}
            for prod in dict_imp_sel_reg[imp_cat_sel][imp_cat_eff]:
                if prod not in dict_imp_sel_reg_cons[imp_cat_eff]:
                    dict_imp_sel_reg_cons[imp_cat_eff][prod] = (
                            dict_imp_sel_reg[imp_cat_sel][imp_cat_eff][prod])
    return dict_imp_sel_reg_cons



def get_dict_imp_prod_cntr_sort(dict_imp,
                                dict_imp_prod_sort,
                                df_tY, dict_imp_pME):
    dict_tY = df_tY.to_dict()
    dict_tY_prod_cntr = {}
    for tup_cntr_prod in dict_tY:
        cntr, prod = tup_cntr_prod
        if prod not in dict_tY_prod_cntr:
            dict_tY_prod_cntr[prod] = {}
        dict_tY_prod_cntr[prod][cntr] = dict_tY[tup_cntr_prod]

    dict_imp_prod_cntr_sort = {}
    for imp_cat in dict_imp_prod_sort:
        dict_imp_prod_cntr_sort[imp_cat] = OrderedDict()
        dict_imp_pME_prod_cntr = {}
        for prod in dict_imp_prod_sort[imp_cat]:
            dict_imp_prod_cntr_sort[imp_cat][prod] = OrderedDict()
            dict_imp_prod_cntr = dict_imp[imp_cat][prod]
            dict_imp_pME_prod_cntr[prod] = {}
            imp_abs_prod = dict_imp_prod_sort[imp_cat][prod]['imp_abs']

            for cntr in dict_imp_prod_cntr:
                imp_abs_prod_cntr = dict_imp_prod_cntr[cntr]
                y_prod_cntr = dict_tY_prod_cntr[prod][cntr]
                dict_imp_pME_prod_cntr[prod][cntr] = (
                        dict_imp_pME[imp_cat][prod][cntr])
            list_imp_pME_prod_cntr_sort = sorted(
                        dict_imp_pME_prod_cntr[prod].items(),
                        key=operator.itemgetter(1))
            imp_cum_prod_cntr = 0
            for tup_cntr_imp_pME in list_imp_pME_prod_cntr_sort:
                cntr, imp_pME_prod_cntr = tup_cntr_imp_pME
                if cntr not in dict_imp_prod_cntr_sort[imp_cat][prod]:
                    dict_imp_prod_cntr_sort[imp_cat][prod][cntr] = {}
                y_prod_cntr = df_tY.loc[(cntr, prod)]
                imp_abs_prod_cntr = dict_imp_prod_cntr[cntr]
                if imp_abs_prod > 0:
                    imp_rel_prod_cntr = imp_abs_prod_cntr/imp_abs_prod
                else:
                    imp_rel_prod_cntr = 0.0
                imp_cum_prod_cntr += imp_rel_prod_cntr
                dict_imp_prod_cntr_sort[imp_cat][prod][cntr]['imp_abs'] = (
                        imp_abs_prod_cntr)
                dict_imp_prod_cntr_sort[imp_cat][prod][cntr]['imp_rel'] = (
                        imp_rel_prod_cntr)
                dict_imp_prod_cntr_sort[imp_cat][prod][cntr]['imp_cum'] = (
                        imp_cum_prod_cntr)
                dict_imp_prod_cntr_sort[imp_cat][prod][cntr]['imp_pME'] = (
                        imp_pME_prod_cntr)
                dict_imp_prod_cntr_sort[imp_cat][prod][cntr]['y'] = y_prod_cntr
    return dict_imp_prod_cntr_sort


def get_dict_imp_prod_reg(dict_imp_prod_sel, dict_reg_fd, reg_fd):
    reg_dom = 'domestic'
    reg_import = 'import'
    dict_imp_prod_reg = {}
    for cntr_fd in dict_imp_prod_sel:
        for imp_cat in dict_imp_prod_sel[cntr_fd]:
            if imp_cat not in dict_imp_prod_reg:
                dict_imp_prod_reg[imp_cat] = {}
            for prod in dict_imp_prod_sel[cntr_fd][imp_cat]:
                if prod not in dict_imp_prod_reg[imp_cat]:
                    dict_imp_prod_reg[imp_cat][prod] = {}
                    dict_imp_prod_reg[imp_cat][prod][reg_dom] = 0
                    dict_imp_prod_reg[imp_cat][prod][reg_import] = 0
                for cntr in dict_imp_prod_sel[cntr_fd][imp_cat][prod]:
                    imp_abs = dict_imp_prod_sel[cntr_fd][imp_cat][prod][cntr]
                    if cntr_fd == cntr:
                        reg = reg_dom
                    else:
                        reg = reg_import
                    dict_imp_prod_reg[imp_cat][prod][reg] += imp_abs
    return dict_imp_prod_reg


def get_dict_xlim(dict_imp_sel_reg):
    dict_xlim = {}
    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_sel_reg):
        fig_y_size = len(dict_imp_sel_reg[imp_cat_sel][imp_cat_sel])
        plt.close('all')
        fig = plt.figure(figsize=cm2inch((16, 1+fig_y_size*0.4)))
        for imp_cat_eff_id, imp_cat_eff in (
                enumerate(dict_imp_sel_reg[imp_cat_sel])):
            plot_id = imp_cat_eff_id+1
            plot_loc = 140+plot_id
            ax = fig.add_subplot(plot_loc)
            df = pd.DataFrame(dict_imp_sel_reg[imp_cat_sel][imp_cat_eff])
            df.loc['import'].T.plot.barh(stacked=True,
                                         ax=ax,
                                         legend=False,
                                         color='C0')
            xlim = ax.get_xlim()
            xlim_max_magn = 10**np.floor(np.log10(xlim[1]))
            xlim_max_ceil = math.ceil(xlim[1]/xlim_max_magn)*xlim_max_magn
            tup_xlim_max_ceil = (int(xlim[0]), xlim_max_ceil)
            if imp_cat_eff not in dict_xlim:
                dict_xlim[imp_cat_eff] = tup_xlim_max_ceil
            if xlim[1] > dict_xlim[imp_cat_eff][1]:
                dict_xlim[imp_cat_eff] = tup_xlim_max_ceil
    return dict_xlim


def plot_priority_setting_individual_one_plot(dict_imp_sel_reg):
    analysis_name = 'priority_setting'
    dict_prod_order = {}
    for imp_cat_sel in dict_imp_sel_reg:
        df = pd.DataFrame(dict_imp_sel_reg[imp_cat_sel][imp_cat_sel])
        df_sum_sort = df.sum().sort_values()
        prod_order = list(df_sum_sort.index)
        dict_prod_order[imp_cat_sel] = prod_order

    plt.close('all')
    fig_y_size_max = 0
    for imp_cat_sel in dict_imp_sel_reg:
        fig_y_size = len(dict_imp_sel_reg[imp_cat_sel][imp_cat_sel])
        if fig_y_size > fig_y_size_max:
            fig_y_size_max = fig_y_size
    fig = plt.figure(figsize=cm2inch((16, 8)))
    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_sel_reg):
        fig_y_size = len(dict_imp_sel_reg[imp_cat_sel][imp_cat_sel])
        fp = dict_imp_cat_fp[imp_cat_sel]
        prod_order = dict_prod_order[imp_cat_sel]
        plot_id = imp_cat_sel_id+1
        plot_loc = 220+plot_id
        ax = fig.add_subplot(plot_loc)
        fp = dict_imp_cat_fp[imp_cat_sel]
        unit = dict_imp_cat_unit[imp_cat_sel[-1]]
        ax.set_xlabel('{} [{}]'.format(fp, unit), fontsize=font_size)
        df = pd.DataFrame(dict_imp_sel_reg[imp_cat_sel][imp_cat_sel])
        df = df.reindex(prod_order, axis=1)
        column_name_dummy = ''
        prod_order_dummy = prod_order
        while len(df.T) < 9:
            df[column_name_dummy] = 0
            prod_order_dummy.reverse()
            prod_order_dummy.append(column_name_dummy)
            prod_order_dummy.reverse()
            df = df.reindex(prod_order, axis=1)
            column_name_dummy += ' '


        df.loc['import'].T.plot.barh(stacked=True,
                                     ax=ax,
                                     legend=False,
                                     color='C0',
                                     width=0.8)
        yticklabels = ax.get_yticklabels()
        ax.set_yticklabels(yticklabels, fontsize=font_size)

        plt.locator_params(axis='x', nbins=1)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.set_xlim(dict_xlim[imp_cat_sel])
        xtick_magnitude = dict_imp_cat_magnitude[imp_cat_sel]

        list_xtick = [i/xtick_magnitude for i in dict_xlim[imp_cat_sel]]
        list_xtick[0] = int(list_xtick[0])
        ax.set_xticks(list(dict_xlim[imp_cat_sel]))
        ax.set_xticklabels(list_xtick, fontsize=font_size)

        xtick_objects = ax.xaxis.get_major_ticks()
        xtick_objects[0].label1.set_horizontalalignment('left')
        xtick_objects[-1].label1.set_horizontalalignment('right')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.yaxis.set_tick_params(size=0)
    fig.tight_layout()
    plt.subplots_adjust(wspace=1)

    priority_setting_dir_path = result_dir_path+priority_setting_dir_name
    fig_file_name = analysis_name+'.pdf'
    pdf_dir_path = priority_setting_dir_path+pdf_dir_name
    fig_file_path = pdf_dir_path+fig_file_name
    fig.savefig(fig_file_path)

    fig_file_name = analysis_name+'.png'
    png_dir_path = priority_setting_dir_path+png_dir_name
    fig_file_path = png_dir_path+fig_file_name
    fig.savefig(fig_file_path)

    return dict_prod_order


def get_dict_xlim_agg(dict_imp_sel_reg_cons):
    dict_xlim_agg = {}
    plt.close('all')
    fig = plt.figure(figsize=cm2inch((16, 2)))
    for imp_cat_id, imp_cat in enumerate(dict_imp_sel_reg_cons):
        plot_id = imp_cat_id+1
        plot_loc = 140+plot_id
        ax = fig.add_subplot(plot_loc)

        fp = dict_imp_cat_fp[imp_cat]
        ax.set_title(fp, fontsize=font_size)

        unit = dict_imp_cat_unit[imp_cat[-1]]
        ax.set_xlabel(unit, fontsize=font_size)

        df = pd.DataFrame(dict_imp_sel_reg_cons[imp_cat])
        df = df.sum(axis=1)

        pd.DataFrame(df).T.plot.barh(stacked=True, ax=ax, legend=False)
        if plot_id > 1:
            ax.set_yticklabels([])

        xlim = ax.get_xlim()
        xlim_max_magn = 10**np.floor(np.log10(xlim[1]))
        xlim_max_ceil = math.ceil(xlim[1]/xlim_max_magn)*xlim_max_magn
        if xlim_max_ceil >= 1.0:
            xlim_max_ceil = int(xlim_max_ceil)
        tup_xlim_max_ceil = (xlim[0], xlim_max_ceil)
        if imp_cat not in dict_xlim_agg:
            dict_xlim_agg[imp_cat] = tup_xlim_max_ceil
        if xlim > dict_xlim_agg[imp_cat]:
            dict_xlim_agg[imp_cat] = tup_xlim_max_ceil

    return dict_xlim_agg


def plot_improv(dict_imp_sel_reg_cons, dict_imp_new_sel_reg_flip, dict_xlim_improv):
    plt.close('all')
    dict_improv = {}

    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_sel_reg_flip):
        dict_improv[imp_cat_sel] = {}

        fig = plt.figure(
                figsize=cm2inch(
                        (16,
                         len(dict_imp_new_sel_reg_flip[imp_cat_sel][imp_cat_sel])*.4+2)))
        for imp_cat_eff_id, imp_cat_eff in (
                enumerate(dict_imp_new_sel_reg_flip[imp_cat_sel])):
            plot_id = imp_cat_eff_id+1
            plot_loc = 140+plot_id
            ax = fig.add_subplot(plot_loc)
            fp = dict_imp_cat_fp[imp_cat_eff]
            ax.set_title(fp, fontsize=font_size)
            unit = dict_imp_cat_unit[imp_cat_eff[-1]]
            ax.set_xlabel(unit, fontsize=font_size)
            ax.set_xlim(dict_xlim_improv[imp_cat_eff])
            df_old = pd.DataFrame(dict_imp_sel_reg_cons[imp_cat_eff])
            df_new = pd.DataFrame(
                    dict_imp_new_sel_reg_flip[imp_cat_sel][imp_cat_eff])
            df = df_new-df_old
            prod_order = dict_prod_order[imp_cat_eff]
            df = df.reindex(prod_order, axis=1)
            df_color = df.loc['import'] <= 0
            df.T.plot.barh(stacked=True,
                           ax=ax,
                           legend=False,
                           color=[df_color.map({True: 'g', False: 'r'})],
                           width=0.8)
            dict_improv[imp_cat_sel][imp_cat_eff] = df.sum(axis=0).to_dict()

            yticklabels = ax.get_yticklabels()

            if plot_id > 1:
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels(yticklabels, fontsize=font_size)

            plt.locator_params(axis='x', nbins=4)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            xtick_magnitude = dict_imp_cat_magnitude[imp_cat_eff]
            list_xtick = [i/xtick_magnitude for i in dict_xlim_improv[imp_cat_eff]]
            ax.set_xticks(list(dict_xlim_improv[imp_cat_eff]))
            ax.set_xticklabels(list_xtick, fontsize=font_size)

            xtick_objects = ax.xaxis.get_major_ticks()
            xtick_objects[0].label1.set_horizontalalignment('left')
            xtick_objects[-1].label1.set_horizontalalignment('right')

        fig.tight_layout(pad=0)
        plt.subplots_adjust(wspace=0.1)
        fp = dict_imp_cat_fp[imp_cat_sel]
        fp_lower = fp.lower()

        reduc_dir_path = result_dir_path+reduc_dir_name
        fig_file_name = fp_lower+'.pdf'
        pdf_dir_path = reduc_dir_path+pdf_dir_name
        fig_file_path = pdf_dir_path+fig_file_name
        fig.savefig(fig_file_path)

        fig_file_name = fp_lower+'.png'
        png_dir_path = reduc_dir_path+png_dir_name
        fig_file_path = png_dir_path+fig_file_name
        fig.savefig(fig_file_path)

    return dict_improv


def plot_improv_agg(dict_pot_imp_agg, dict_xlim_agg):
    df_improv_agg = pd.DataFrame(dict_pot_imp_agg)
    df_improv_agg = df_improv_agg.reindex(list(dict_pot_imp_agg.keys()), axis=1)
    list_imp_cat_order = list(df_improv_agg.columns.values)
    list_imp_cat_order.reverse()
    list_imp_cat_order.append('Prior')


    (dict_pot_imp_agg)
    plt.close('all')
    analysis_name = 'potential_improvement_agg'
    fig = plt.figure(figsize=cm2inch((16, 5*.4+2)))

    font_size = 8.0
    dict_improv_agg = {}
    for fp_id, fp in enumerate(df_improv_agg):
        imp_cat_eff = dict_fp_imp_cat[fp]
        plot_id = fp_id+1
        plot_loc = 140+plot_id
        ax = fig.add_subplot(plot_loc)

        df_improv_agg_imp_cat = df_improv_agg[fp]
        df_improv_agg_imp_cat = df_improv_agg_imp_cat.reindex(list_imp_cat_order, axis=1)

        dict_improv_agg[fp] = df_improv_agg_imp_cat.to_dict()
        df_improv_agg_imp_cat.plot.barh(stacked=True,
                        legend=False,
                        ax=ax,
                        color=['g', 'g', 'g', 'g', 'C0'],
                        width=0.8)

        ax.set_title(fp, fontsize=font_size)
        ax.set_xlim(list(dict_xlim_agg[imp_cat_eff]))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        unit = dict_imp_cat_unit[imp_cat_eff[-1]]
        ax.set_xlabel(unit, fontsize=font_size)
        xtick_magnitude = dict_imp_cat_magnitude[imp_cat_eff]

        list_xtick = [i/xtick_magnitude for i in dict_xlim_agg[imp_cat_eff]]
        list_xtick[0] = int(list_xtick[0])

        ax.set_xticks(list(dict_xlim_agg[imp_cat_eff]))
        ax.set_xticklabels(list_xtick, fontsize=font_size)

        xtick_objects = ax.xaxis.get_major_ticks()
        xtick_objects[0].label1.set_horizontalalignment('left')
        xtick_objects[-1].label1.set_horizontalalignment('right')

        yticklabels = ax.get_yticklabels()
        if plot_id > 1:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(yticklabels, fontsize=font_size)
            ax.set_ylabel('Optimized footprint', fontsize=font_size)

    fig.tight_layout(pad=0.1)
    plt.subplots_adjust(wspace=0.1)

    reduc_agg_dir_path = result_dir_path+reduc_agg_dir_name
    pdf_file_name = analysis_name+'.pdf'
    pdf_dir_path = reduc_agg_dir_path+pdf_dir_name
    pdf_file_path = pdf_dir_path+pdf_file_name
    fig.savefig(pdf_file_path)

    png_file_name = analysis_name+'.png'
    png_dir_path = reduc_agg_dir_path+png_dir_name
    png_file_path = png_dir_path+png_file_name
    fig.savefig(png_file_path)

###############################################################################
plt.close('all')


dict_cntr_short_long = {}
with open(country_code_file_path, 'r') as read_file:
    csv_file = csv.reader(read_file, delimiter='\t')
    for row in csv_file:
        cntr_short = row[0]
        cntr_long = row[1]
        dict_cntr_short_long[cntr_short] = cntr_long

#   Load shortened product names for plotting.
list_prod_long = []
with open(prod_long_file_path) as read_file:
    csv_file = csv.reader(read_file, delimiter='\t')
    for row in csv_file:
        list_prod_long.append(row[0])

list_prod_short = []
with open(prod_short_file_path) as read_file:
    csv_file = csv.reader(read_file, delimiter='\t')
    for row in csv_file:
        list_prod_short.append(row[0])

dict_prod_long_short = {}
for prod_id, prod_long in enumerate(list_prod_long):
    prod_long = list_prod_long[prod_id]
    prod_short = list_prod_short[prod_id]
    dict_prod_long_short[prod_long] = prod_short

dict_imp_cat_fp = {}
with open(cf_long_footprint_file_path, 'r') as read_file:
    csv_file = csv.reader(read_file, delimiter='\t')
    for row in csv_file:
        imp_cat = tuple(row[:-1])
        fp = row[-1]
        dict_imp_cat_fp[imp_cat] = fp

dict_imp_cat_magnitude = {}
with open(cf_magnitude_file_path, 'r') as read_file:
    csv_file = csv.reader(read_file, delimiter='\t')
    for row in csv_file:
        imp_cat = tuple(row[:-1])
        magnitude = int(row[-1])
        dict_imp_cat_magnitude[imp_cat] = magnitude

prod_order_cons = []
with open(prod_order_file_path, 'r') as read_file:
    csv_file = csv.reader(read_file, delimiter='\t')
    for row in csv_file:
        prod_order_cons.append(row[0])
prod_order_cons.reverse()


#   Load EXIOBASE.
if dict_eb_file_name in os.listdir(data_path):
    dict_eb = pickle.load(open(dict_eb_file_path, 'rb'))
else:
    dict_eb = eb.process(eb.parse())
    save_eb = True
    if save_eb:
        pickle.dump(dict_eb, open(dict_eb_file_path, 'wb'))

#   Get characterisation factors.
dict_cf = get_dict_cf()

#   Generate dictionary with list of EU28 countries.
reg_fd = 'EU28'
dict_reg_fd = get_reg_fd(reg_fd)
list_reg_fd = dict_reg_fd[reg_fd]

reg_dom = 'domestic'
reg_import = 'import'

font_size = 8.0
###############################################################################
# Contribution analysis
# Get dictionary with impacts.
# final demand matrix of EU28
df_tY_eu28 = dict_eb['tY'][list_reg_fd].copy()
# final demand vector of EU28
df_tY_eu28_full = df_tY_eu28.sum(axis=1)

for cntr in list_reg_fd:
    df_tY_eu28.loc[cntr, cntr] = 0
df_tY_eu28_fdsum = df_tY_eu28.sum(axis=1)
#   Calculate dataframes with absolute impact.
dict_df_imp_full = get_dict_df_imp(dict_eb, df_tY_eu28_full, dict_cf)
dict_df_imp = get_dict_df_imp(dict_eb, df_tY_eu28_fdsum, dict_cf)

dict_imp_cat_sum = {}
for imp_cat in dict_df_imp:
    dict_df_imp_cat_sum = dict_df_imp[imp_cat].sum(axis=1).to_dict()
    for key in dict_df_imp_cat_sum.keys():
        dict_imp_cat_sum[key] = dict_df_imp_cat_sum[key]
#   Cast dataframes with absolute impact to dict.
dict_imp = get_dict_imp(dict_df_imp)
#   Get imp_cat_gwp100 for debugging purposes
imp_cat_gwp100 = list(dict_imp.keys())[0]

#   Sort impact up to cumulative limit to get highest contributing products.
imp_cum_lim = 0.5
dict_imp_prod_sort = get_dict_imp_prod_sort(df_tY_eu28_fdsum,
                                            dict_df_imp,
                                            dict_df_imp_full,
                                            imp_cum_lim)

dict_imp_prod_sort_cons = get_dict_imp_prod_sort_cons(dict_imp_prod_sort,
                                                      dict_df_imp)

imp_cum_lim = 1.1
dict_imp_prod_sort_full = get_dict_imp_prod_sort(df_tY_eu28_fdsum,
                                            dict_df_imp,
                                            dict_df_imp_full,
                                            imp_cum_lim)

dict_imp_prod_sort_cons = get_dict_imp_prod_sort_cons(dict_imp_prod_sort_full,
                                                      dict_df_imp)

#   Disaggregate impacts by domestic production, imports from in- and exEU28.
dict_df_imp_cntr = {}
dict_imp_cntr = {}
dict_imp_prod_sel = {}
for cntr_fd in list_reg_fd:
    df_tY_cntr = dict_eb['tY'][cntr_fd].copy()
    df_tY_cntr.loc[cntr_fd] = 0
    df_tY_cntr_fdsum = df_tY_cntr.sum(axis=1)
    # For all countries in EU28, calculate dataframes with absolute impact.
    dict_df_imp_cntr[cntr_fd] = get_dict_df_imp(dict_eb,
                                                df_tY_cntr_fdsum,
                                                dict_cf)
    # For all countries in EU28, cast dataframes with absolute impact to dict.
    dict_imp_cntr[cntr_fd] = get_dict_imp(dict_df_imp_cntr[cntr_fd])

dict_imp_sel = {}
for cntr_fd in list_reg_fd:
    # For all countries in EU28, select highest contributing products.
    dict_imp_sel[cntr_fd] = {}
    for imp_cat_sel in dict_imp_prod_sort:
        dict_imp_sel[cntr_fd][imp_cat_sel] = {}
        for imp_cat_eff in dict_imp:
            dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff] = OrderedDict()
            for prod in dict_imp_prod_sort[imp_cat_sel]:
                dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod] = (
                        dict_imp_cntr[cntr_fd][imp_cat_eff][prod])

dict_imp_sel_reg = get_dict_imp_sel_reg(dict_imp_sel)
dict_xlim = get_dict_xlim(dict_imp_sel_reg)

dict_imp_sel_cons = {}
for cntr_fd in list_reg_fd:
    # For all countries in EU28, select highest contributing products..
    dict_imp_sel_cons[cntr_fd] = {}
    for imp_cat_sel in dict_imp_prod_sort_cons:
        dict_imp_sel_cons[cntr_fd][imp_cat_sel] = {}
        for imp_cat_eff in dict_imp:
            dict_imp_sel_cons[cntr_fd][imp_cat_sel][imp_cat_eff] = OrderedDict()
            for prod in dict_imp_prod_sort_cons[imp_cat_sel]:
                dict_imp_sel_cons[cntr_fd][imp_cat_sel][imp_cat_eff][prod] = (
                        dict_imp_cntr[cntr_fd][imp_cat_eff][prod])

dict_imp_sel_reg_cons = get_dict_imp_sel_reg_cons(dict_imp_sel_cons)

dict_imp_cat_unit = {}
dict_imp_cat_unit['kg CO2 eq.'] = r'$Pg\/CO_2\/eq.$'
dict_imp_cat_unit['kt'] = r'$Gt$'
dict_imp_cat_unit['Mm3'] = r'$Mm^3$'
dict_imp_cat_unit['km2'] = r'$Gm^2$'

###############################################################################
# Priority setting
dict_prod_order = plot_priority_setting_individual_one_plot(dict_imp_sel_reg)
dict_xlim_agg = get_dict_xlim_agg(dict_imp_sel_reg_cons)


###############################################################################
#   Optimize impact per ME for highest contributing products

#   Get dictionary of products sorted by absolute impact
#   and sorted by highest contributing country of origin.

#   Calculate absolute impact of imported products to EU28.
x_prod_cntr_min = 0.5

df_cQe = dict_cf['e']
df_cQm = dict_cf['m']
df_cQr = dict_cf['r']

df_cRe = dict_eb['cRe']
df_cRm = dict_eb['cRm']
df_cRr = dict_eb['cRr']

df_cL = dict_eb['cL']

dict_df_imp_pME = {}
dict_df_imp_pME['e'] = df_cQe.dot(df_cRe).dot(df_cL)
dict_df_imp_pME['m'] = df_cQm.dot(df_cRm).dot(df_cL)
dict_df_imp_pME['r'] = df_cQr.dot(df_cRr).dot(df_cL)

dict_imp_pME = get_dict_imp(dict_df_imp_pME)

dict_imp_prod_cntr_sort = get_dict_imp_prod_cntr_sort(dict_imp,
                                                      dict_imp_prod_sort_cons,
                                                      df_tY_eu28_fdsum,
                                                      dict_imp_pME)

#dict_imp_prod_cntr_sort = get_dict_imp_prod_cntr_sort(dict_imp,
#                                                      dict_imp_prod_sort_full,
#                                                      df_tY_eu28_fdsum,
#                                                      dict_imp_pME)


df_tY_eu28_cntr = df_tY_eu28.sum(axis=1, level=0)

dict_tY_eu28_cntr = df_tY_eu28_cntr.to_dict()
dict_tY_eu28_cntr_import = {}
dict_tY_eu28_cntr_dom = {}
for cntr_fd in dict_tY_eu28_cntr:
    for tup_cntr_prod in dict_tY_eu28_cntr[cntr_fd]:
        cntr, prod = tup_cntr_prod
        if prod not in dict_tY_eu28_cntr_import:
            dict_tY_eu28_cntr_import[prod] = {}

        if prod not in dict_tY_eu28_cntr_dom:
            dict_tY_eu28_cntr_dom[prod] = {}

        if cntr not in dict_tY_eu28_cntr_import[prod]:
            dict_tY_eu28_cntr_import[prod][cntr] = 0

        if cntr not in dict_tY_eu28_cntr_dom[prod]:
            dict_tY_eu28_cntr_dom[prod][cntr] = 0

        if cntr_fd != cntr:
            dict_tY_eu28_cntr_import[prod][cntr] += (
                    dict_tY_eu28_cntr[cntr_fd][tup_cntr_prod])
        else:
            dict_tY_eu28_cntr_dom[prod][cntr] += (
                    dict_tY_eu28_cntr[cntr_fd][tup_cntr_prod])

dict_tY_eu28_old_dom = {}
for prod in dict_tY_eu28_cntr_dom:
    for cntr in dict_tY_eu28_cntr_dom[prod]:
        tup_cntr_prod = (cntr, prod)
        dict_tY_eu28_old_dom[tup_cntr_prod] = dict_tY_eu28_cntr_dom[prod][cntr]
df_tY_eu28_old_dom = pd.DataFrame(dict_tY_eu28_old_dom, index=['domestic']).T
df_tY_eu28_old_dom = df_tY_eu28_old_dom.reindex(index=df_tY_eu28_fdsum.index)
dict_tY_eu28_old_imp = {}
for prod in dict_tY_eu28_cntr_import:
    for cntr in dict_tY_eu28_cntr_import[prod]:
        tup_cntr_prod = (cntr, prod)
        dict_tY_eu28_old_imp[tup_cntr_prod] = (
                dict_tY_eu28_cntr_import[prod][cntr])
df_tY_eu28_old_imp = pd.DataFrame(dict_tY_eu28_old_imp, index=['import']).T
df_tY_eu28_old_imp = df_tY_eu28_old_imp.reindex(index=df_tY_eu28_fdsum.index)
df_tY_eu28_old_reg = pd.concat([df_tY_eu28_old_dom, df_tY_eu28_old_imp],
                               axis=1)

#   Get dictionary with total output of products with highest impact
#   for all countries.
df_tY_world = dict_eb['tY'].copy()
df_tY_world_fdsum = df_tY_world.sum(axis=1)
df_tX_world = dict_eb['cL'].dot(df_tY_world_fdsum)
dict_tX_world = df_tX_world.to_dict()
dict_tX_world_prod_cntr = {}
for tup_cntr_prod in dict_tX_world:
    cntr, prod = tup_cntr_prod
    if prod not in dict_tX_world_prod_cntr:
        dict_tX_world_prod_cntr[prod] = {}
    dict_tX_world_prod_cntr[prod][cntr] = dict_tX_world[tup_cntr_prod]

# remove domestic use
#   Get dictionary with total output of products with highest impact
#   for all countries.
df_tY_world = dict_eb['tY'].copy()
list_cntr = list(df_tY_world.columns.get_level_values(0))
for cntr in list_cntr:
    df_tY_world.loc[cntr, cntr] = 0
df_tY_world_ex = df_tY_world.sum(axis=1)
dict_tY_world_ex = df_tY_world_ex.to_dict()
dict_tY_world_ex_prod_cntr = {}
for tup_cntr_prod in dict_tY_world_ex:
    cntr, prod = tup_cntr_prod
    if prod not in dict_tY_world_ex_prod_cntr:
        dict_tY_world_ex_prod_cntr[prod] = {}
    dict_tY_world_ex_prod_cntr[prod][cntr] = dict_tY_world_ex[tup_cntr_prod]

#       Plot current impact per ME against %imported by EU28 vs total output of
#       country per product.
y_sum_mg_old = 0
dict_imp_pME_old = {}
dict_y_old = {}
dict_lim = {}

dict_ax = {}


dict_imp_cat_prod_imp_abs = {}
dict_imp_cat_prod_imp_abs_sum = {}
for imp_cat in dict_imp_prod_cntr_sort:
    dict_imp_cat_prod_imp_abs[imp_cat] = {}
    dict_imp_cat_prod_imp_abs_sum[imp_cat] = {}
    for prod in dict_imp_prod_cntr_sort[imp_cat]:
        dict_imp_cat_prod_imp_abs[imp_cat][prod] ={}
        imp_abs_sum = 0
        for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
            imp_abs = dict_imp_prod_cntr_sort[imp_cat][prod][cntr]['imp_abs']
            dict_imp_cat_prod_imp_abs[imp_cat][prod][cntr] = imp_abs
            imp_abs_sum += imp_abs
        dict_imp_cat_prod_imp_abs_sum[imp_cat][prod] = imp_abs_sum


dict_imp_cat_prod_cntr_sort_trunc = {}
for imp_cat in dict_imp_cat_prod_imp_abs:
    dict_imp_cat_prod_cntr_sort_trunc[imp_cat] = {}
    for prod in dict_imp_cat_prod_imp_abs[imp_cat]:
        list_imp_cat_prod_sort = sorted(dict_imp_cat_prod_imp_abs[imp_cat][prod].items(),
                                        key=operator.itemgetter(1), reverse=True)
        list_imp_cat_prod_sort_trunc = []
        imp_abs_sum = dict_imp_cat_prod_imp_abs_sum[imp_cat][prod]
        if imp_abs_sum > 0:
            imp_cum = 0
            bool_add = True
            for tup_cntr_imp in list_imp_cat_prod_sort:
                cntr, imp_abs = tup_cntr_imp
                imp_rel = imp_abs/imp_abs_sum
                imp_cum += imp_rel
                if imp_cum <= 0.5:
                    list_imp_cat_prod_sort_trunc.append(cntr)

                elif bool_add:
                    list_imp_cat_prod_sort_trunc.append(cntr)
                    bool_add = False
        dict_imp_cat_prod_cntr_sort_trunc[imp_cat][prod] = list_imp_cat_prod_sort_trunc



for imp_cat in dict_imp_prod_cntr_sort:
    if imp_cat == imp_cat_gwp100:
        for prod in dict_imp_prod_cntr_sort[imp_cat]:
            if prod == 'Chemicals nec':
                dict_gwp100_chem_nec = {}
                imp_abs_sum = 0
                for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                    imp_abs = dict_imp_prod_cntr_sort[imp_cat][prod][cntr]['imp_abs']
                    dict_gwp100_chem_nec[cntr] = imp_abs
                    imp_abs_sum += imp_abs

list_gwp100_chem_nec_sort = sorted(dict_gwp100_chem_nec.items(),
                                   key=operator.itemgetter(1), reverse=True)
list_gwp100_chem_nec_sort_trunc = []
imp_cum = 0
bool_add = True
for tup_cntr_imp in list_gwp100_chem_nec_sort:
    cntr, imp_abs = tup_cntr_imp
    imp_rel = imp_abs/imp_abs_sum
    imp_cum += imp_rel
    if imp_cum <= 0.5:
        list_gwp100_chem_nec_sort_trunc.append(cntr)

    elif bool_add:
        list_gwp100_chem_nec_sort_trunc.append(cntr)
        bool_add = False



for imp_cat in dict_imp_prod_cntr_sort:
    dict_imp_pME_old[imp_cat] = {}
    dict_y_old[imp_cat] = {}
    dict_lim[imp_cat] = {}
    dict_ax[imp_cat] = {}
    for prod in dict_imp_prod_cntr_sort[imp_cat]:
        dict_imp_pME_old[imp_cat][prod] = {}
        dict_y_old[imp_cat][prod] = {}
        plt.close('all')
        x_start = 0
        y_start = 0
        list_rect_y = []
        list_rect_x = []
        fig = plt.figure(figsize=cm2inch((16, 8)))
        ax = plt.gca()
        y_prod_cntr_sum = 0
        imp_abs_old = 0
        for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
            imp_pME_prod_cntr = dict_imp_pME[imp_cat][prod][cntr]
            y_prod_cntr = dict_tY_eu28_cntr_import[prod][cntr]
            y_prod_cntr_sum += y_prod_cntr
            y_sum_mg_old += y_prod_cntr
            x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
            if x_prod_cntr >= x_prod_cntr_min:
                if cntr in dict_imp_cat_prod_cntr_sort_trunc[imp_cat][prod]:
                    cntr_long = dict_cntr_short_long[cntr]
                    plt.text(x_start+y_prod_cntr/2,
                             y_start+imp_pME_prod_cntr,
                             ' '+cntr_long,
                             rotation=90,
                             verticalalignment='bottom',
                             horizontalalignment='center',
                             fontsize=font_size,
                             color='C0')
                dict_imp_pME_old[imp_cat][prod][cntr] = imp_pME_prod_cntr
                dict_y_old[imp_cat][prod][cntr] = y_prod_cntr
                rect_y = patches.Rectangle((x_start, y_start),
                                           y_prod_cntr,
                                           imp_pME_prod_cntr)
                rect_x = patches.Rectangle((x_start, y_start),
                                           x_prod_cntr,
                                           imp_pME_prod_cntr)
                list_rect_y.append(rect_y)
                list_rect_x.append(rect_x)
                imp_abs_old += y_prod_cntr*imp_pME_prod_cntr
                x_max = x_start+x_prod_cntr
                y_max = y_start+imp_pME_prod_cntr
                x_start += x_prod_cntr

        col_rect_y = mpl_col.PatchCollection(list_rect_y, facecolor='C0')
        col_rect_x = mpl_col.PatchCollection(list_rect_x, facecolor='gray')
        ax.add_collection(col_rect_x)
        ax.add_collection(col_rect_y)
        ax.autoscale()
        fig = ax.get_figure()
        unit = imp_cat[-1]
        ax.set_ylabel('{}/M€'.format(unit), fontsize=font_size)
        ax.set_xlabel('M€', fontsize=font_size)
        fp = dict_imp_cat_fp[imp_cat]
        fp_lower = fp.lower()
        prod_short = dict_prod_long_short[prod]
        prod_short_lower = prod_short.lower()
        prod_short_lower_strip = prod_short_lower.strip()
        fig.tight_layout(pad=0)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        dict_lim[imp_cat][prod] = {}
        dict_lim[imp_cat][prod]['x'] = (0, x_max)
        dict_lim[imp_cat][prod]['y'] = (0, y_max)
        plt.locator_params(axis='both', nbins=4, tight=True)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        dict_ax[imp_cat][prod] = ax
        fig.tight_layout(pad=0.1)



dict_tY_prod = {}
for prod in dict_tY_eu28_cntr_import:
    dict_tY_prod[prod] = 0
    for cntr in dict_tY_eu28_cntr_import[prod]:
        dict_tY_prod[prod] += dict_tY_eu28_cntr_import[prod][cntr]

dict_imp_pME_new = {}
dict_y_new = {}
#   Plot optimized impact per ME against %imported by EU28 vs
#   total output of country per product
y_sum_mg_new = 0

for imp_cat in dict_imp_prod_cntr_sort:
    dict_imp_pME_new[imp_cat] = {}
    dict_y_new[imp_cat] = {}
    for prod in dict_imp_prod_cntr_sort[imp_cat]:
        plt.close('all')
        x_start = 0
        x_end = 0
        y_start = 0
        list_rect_y = []
        list_rect_x = []
        ax = dict_ax[imp_cat][prod]
        y_prod = dict_tY_prod[prod]
        imp_abs_new = 0
        dict_imp_pME_new[imp_cat][prod] = {}
        dict_y_new[imp_cat][prod] = {}
        for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
            imp_pME_prod_cntr = dict_imp_pME[imp_cat][prod][cntr]
            y_prod_cntr = dict_tY_eu28_cntr_import[prod][cntr]
            x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
            if x_prod_cntr >= x_prod_cntr_min:
                if x_prod_cntr < y_prod:
                    y_sum_mg_new += x_prod_cntr
                    rect_y = patches.Rectangle((x_start, y_start),
                                               x_prod_cntr,
                                               imp_pME_prod_cntr)
                    y_prod -= x_prod_cntr
                    imp_abs_new += x_prod_cntr*imp_pME_prod_cntr
                    dict_y_new[imp_cat][prod][cntr] = x_prod_cntr
                    dict_imp_pME_new[imp_cat][prod][cntr] = (
                            imp_pME_prod_cntr)
                elif y_prod > 0:
                    y_sum_mg_new += y_prod
                    rect_y = patches.Rectangle((x_start, y_start),
                                               y_prod,
                                               imp_pME_prod_cntr)
                    imp_abs_new += y_prod*imp_pME_prod_cntr
                    dict_y_new[imp_cat][prod][cntr] = y_prod
                    dict_imp_pME_new[imp_cat][prod][cntr] = (
                            imp_pME_prod_cntr)
                    y_prod -= y_prod
                rect_x = patches.Rectangle((x_start, y_start),
                                           x_prod_cntr,
                                           imp_pME_prod_cntr)
                list_rect_y.append(rect_y)
                list_rect_x.append(rect_x)
                x_start += x_prod_cntr

                rect_y = patches.Rectangle((0, 0), 0, 0)
                list_rect_y.append(rect_y)

        col_rect_y = mpl_col.PatchCollection(list_rect_y, facecolor='green')
        col_rect_x = mpl_col.PatchCollection(list_rect_x, facecolor='gray')
        col_rect_y.set_alpha(0.5)

dict_imp_cat_prod_imp_abs_new = {}
dict_imp_cat_prod_imp_abs_new_sum = {}
for imp_cat in dict_imp_pME_new:
    dict_imp_cat_prod_imp_abs_new[imp_cat] = {}
    dict_imp_cat_prod_imp_abs_new_sum[imp_cat] = {}
    for prod in dict_imp_pME_new[imp_cat]:
        dict_imp_cat_prod_imp_abs_new[imp_cat][prod] ={}
        imp_abs_sum = 0
        for cntr in dict_imp_pME_new[imp_cat][prod]:
            imp_pME = dict_imp_pME_new[imp_cat][prod][cntr]
            y_new = dict_y_new[imp_cat][prod][cntr]
            imp_abs = imp_pME*y_new
            imp_abs_sum += imp_abs
            dict_imp_cat_prod_imp_abs_new[imp_cat][prod][cntr] = imp_abs
        dict_imp_cat_prod_imp_abs_new_sum[imp_cat][prod] = imp_abs_sum


dict_imp_cat_prod_cntr_sort_new_trunc = {}
for imp_cat in dict_imp_cat_prod_imp_abs_new:
    dict_imp_cat_prod_cntr_sort_new_trunc[imp_cat] = {}
    for prod in dict_imp_cat_prod_imp_abs_new[imp_cat]:
        list_imp_cat_prod_sort = sorted(dict_imp_cat_prod_imp_abs_new[imp_cat][prod].items(),
                                        key=operator.itemgetter(1), reverse=True)
        list_imp_cat_prod_sort_trunc = []
        imp_abs_sum = dict_imp_cat_prod_imp_abs_new_sum[imp_cat][prod]
        if imp_abs_sum > 0:
            imp_cum = 0
            bool_add = True
            for tup_cntr_imp in list_imp_cat_prod_sort:
                cntr, imp_abs = tup_cntr_imp
                imp_rel = imp_abs/imp_abs_sum
                imp_cum += imp_rel
                if imp_cum <= 0.5:
                    list_imp_cat_prod_sort_trunc.append(cntr)

                elif bool_add:
                    list_imp_cat_prod_sort_trunc.append(cntr)
                    bool_add = False
        dict_imp_cat_prod_cntr_sort_new_trunc[imp_cat][prod] = list_imp_cat_prod_sort_trunc



dict_imp_gwp100_chem_new = {}
imp_abs_sum = 0
for cntr in dict_imp_pME_new[imp_cat_gwp100]['Chemicals nec']:
    imp_pME = dict_imp_pME_new[imp_cat_gwp100]['Chemicals nec'][cntr]
    y_new = dict_y_new[imp_cat_gwp100]['Chemicals nec'][cntr]
    imp_abs = imp_pME*y_new
    imp_abs_sum += imp_abs
    dict_imp_gwp100_chem_new[cntr] = imp_abs

list_imp_gwp100_chem_new = sorted(dict_imp_gwp100_chem_new.items(),
                                  key=operator.itemgetter(1), reverse=True)

imp_cum = 0
list_imp_gwp100_chem_new_trunc = []
bool_add = True
for tup_cntr_imp in list_imp_gwp100_chem_new:
    cntr, imp_abs = tup_cntr_imp
    imp_rel = imp_abs/imp_abs_sum
    imp_cum += imp_rel
    if imp_cum <= 0.5:
        list_imp_gwp100_chem_new_trunc.append(cntr)
    elif bool_add:
        list_imp_gwp100_chem_new_trunc.append(cntr)
        bool_add = False

dict_tY_prod = {}
for prod in dict_tY_eu28_cntr_import:
    dict_tY_prod[prod] = 0
    for cntr in dict_tY_eu28_cntr_import[prod]:
        dict_tY_prod[prod] += dict_tY_eu28_cntr_import[prod][cntr]

dict_cntr = {}
for imp_cat in dict_imp_prod_cntr_sort:
    dict_imp_pME_new[imp_cat] = {}
    dict_y_new[imp_cat] = {}
    for prod in dict_imp_prod_cntr_sort[imp_cat]:
        plt.close('all')
        x_start = 0
        x_end = 0
        y_start = 0
        list_rect_y = []
        list_rect_x = []
        ax = dict_ax[imp_cat][prod]
        y_prod = dict_tY_prod[prod]
        imp_abs_new = 0
        dict_imp_pME_new[imp_cat][prod] = {}
        dict_y_new[imp_cat][prod] = {}
        for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
            imp_pME_prod_cntr = dict_imp_pME[imp_cat][prod][cntr]
            y_prod_cntr = dict_tY_eu28_cntr_import[prod][cntr]
            x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
            if x_prod_cntr >= x_prod_cntr_min:
                if x_prod_cntr < y_prod:
                    if cntr in dict_imp_cat_prod_cntr_sort_new_trunc[imp_cat][prod]:
                        cntr_long = dict_cntr_short_long[cntr]
                        x_text = x_start+x_prod_cntr/2
                        y_text = y_start+imp_pME_prod_cntr
                        dict_cntr[cntr] = (x_text, y_text)
                        ax.text(x_text,
                                y_text,
                                ' '+cntr_long,
                                rotation=90,
                                verticalalignment='bottom',
                                horizontalalignment='center',
                                fontsize=font_size,
                                color='green')
                    y_sum_mg_new += x_prod_cntr
                    rect_y = patches.Rectangle((x_start, y_start),
                                               x_prod_cntr,
                                               imp_pME_prod_cntr)
                    y_prod -= x_prod_cntr
                    imp_abs_new += x_prod_cntr*imp_pME_prod_cntr
                    dict_y_new[imp_cat][prod][cntr] = x_prod_cntr
                    dict_imp_pME_new[imp_cat][prod][cntr] = (
                            imp_pME_prod_cntr)
                elif y_prod > 0:
                    if cntr in dict_imp_cat_prod_cntr_sort_new_trunc[imp_cat][prod]:
                        cntr_long = dict_cntr_short_long[cntr]
                        x_text = x_start+x_prod_cntr/2
                        y_text = y_start+imp_pME_prod_cntr
                        dict_cntr[cntr] = (x_text, y_text)
                        ax.text(x_text,
                                y_text,
                                ' '+cntr_long,
                                rotation=90,
                                verticalalignment='bottom',
                                horizontalalignment='center',
                                fontsize=font_size,
                                color='green')
                    y_sum_mg_new += y_prod
                    rect_y = patches.Rectangle((x_start, y_start),
                                               y_prod,
                                               imp_pME_prod_cntr)
                    imp_abs_new += y_prod*imp_pME_prod_cntr
                    dict_y_new[imp_cat][prod][cntr] = y_prod
                    dict_imp_pME_new[imp_cat][prod][cntr] = (
                            imp_pME_prod_cntr)
                    y_prod -= y_prod

                rect_x = patches.Rectangle((x_start, y_start),
                                           x_prod_cntr,
                                           imp_pME_prod_cntr)
                list_rect_y.append(rect_y)
                list_rect_x.append(rect_x)
                x_start += x_prod_cntr

                rect_y = patches.Rectangle((0, 0), 0, 0)
                list_rect_y.append(rect_y)

        col_rect_y = mpl_col.PatchCollection(list_rect_y, facecolor='green')
        col_rect_x = mpl_col.PatchCollection(list_rect_x, facecolor='gray')
        col_rect_y.set_alpha(0.5)
        ax.add_collection(col_rect_y)
        ax.autoscale()
        fig = ax.get_figure()
        unit = imp_cat[-1]
        ax.set_ylabel('{}/M€'.format(unit), fontsize=font_size)
        ax.set_xlabel('M€', fontsize=font_size)
        ax.set_xlim(dict_lim[imp_cat][prod]['x'])
        ax.set_ylim(dict_lim[imp_cat][prod]['y'])

        plt.locator_params(axis='both', nbins=4, tight=True)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(font_size)
        ax.xaxis.offsetText.set_fontsize(font_size)

        prod_short = dict_prod_long_short[prod]
        prod_short_lower = prod_short.lower()
        prod_short_lower_strip = prod_short_lower.strip()
        prod_short_lower_strip = prod_short_lower_strip.replace(':','')
        prod_short_lower_strip_us = prod_short_lower_strip.replace(' ','_')
        fp = dict_imp_cat_fp[imp_cat]
        fp_lower = fp.lower()
        fp_prod = fp_lower+prod_short_lower_strip_us
        fig.tight_layout(pad=0.1)

        shift_dir_path = result_dir_path+shift_dir_name

        pdf_file_name = fp_prod+'.pdf'
        pdf_dir_path = shift_dir_path+pdf_dir_name
        pdf_file_path = pdf_dir_path+pdf_file_name
        fig.savefig(pdf_file_path)

        png_file_name = fp_prod+'.png'
        png_dir_path = shift_dir_path+png_dir_name
        png_file_path = png_dir_path+png_file_name
        fig.savefig(png_file_path)

dict_fp_imp_cat = {}
for imp_cat in dict_imp_cat_fp:
    fp = dict_imp_cat_fp[imp_cat]
    dict_fp_imp_cat[fp] = imp_cat


###############################################################################
#   Plot potential improvement per impact category and product
#   Reshape dictionary with new final demand from imports
#   from [cntr][prod] to MultiIndex Dataframe format with tuple [(cntr, prod)]
dict_df_tY_import_new = {}
for imp_cat_id, imp_cat in enumerate(dict_y_new):
    dict_df_tY_import_new[imp_cat] = {}
    for prod in dict_y_new[imp_cat]:
        for cntr in dict_y_new[imp_cat][prod]:
            dict_df_tY_import_new[imp_cat][(cntr, prod)] = (
                    dict_y_new[imp_cat][prod][cntr])

#   For each impact category make new final demand DataFrames
dict_imp_new = {}
dict_imp_prod_reg_new = {}

dict_df_tY_new = {}
for imp_cat_id, imp_cat in enumerate(dict_df_tY_import_new):
    #    Fill import column
    df_tY_eu28 = dict_eb['tY'][list_reg_fd].copy()
    df_tY_eu28_fdsum = df_tY_eu28.sum(axis=1)
    df_tY_eu28_import = df_tY_eu28_fdsum.copy()
    df_tY_eu28_import[:] = 0
    df_tY_eu28_import[list(dict_df_tY_import_new[imp_cat].keys())] = (
            list(dict_df_tY_import_new[imp_cat].values()))
    df_tY_eu28_import.columns = ['import']

    #   Fill domestic column
    dict_tY_eu28_dom = {}
    for prod in dict_tY_eu28_cntr_dom:
        for cntr in dict_tY_eu28_cntr_dom[prod]:
            dict_tY_eu28_dom[(cntr, prod)] = (
                    dict_tY_eu28_cntr_dom[prod][cntr])
    df_tY_eu28_dom = df_tY_eu28_fdsum.copy()
    df_tY_eu28_dom[:] = 0
    df_tY_eu28_dom[list(dict_tY_eu28_dom.keys())] = (
            list(dict_tY_eu28_dom.values()))
    df_tY_eu28_dom.columns = ['domestic']
    df_tY_eu28_new = pd.concat([df_tY_eu28_dom, df_tY_eu28_import], axis=1)
    df_tY_eu28_new.columns = ['domestic', 'import']
    dict_df_tY_new[imp_cat] = df_tY_eu28_new

#   Calculate new impact
dict_imp_new = {}
for imp_cat in dict_df_tY_new:
    dict_imp_new[imp_cat] = {}
    for reg_fd in dict_df_tY_new[imp_cat]:
        df_tY_new = dict_df_tY_new[imp_cat][reg_fd]
        dict_df_imp_new = get_dict_df_imp(dict_eb,
                                          df_tY_new,
                                          dict_cf)
        # Cast dataframes with absolute impact to dict.
        dict_imp_new[imp_cat][reg_fd] = get_dict_imp(dict_df_imp_new)

dict_imp_new_sel = {}
for imp_cat_sel in dict_imp_new:
    dict_imp_new_sel[imp_cat_sel] = {}
    for reg_fd in dict_imp_new[imp_cat_sel]:
        dict_imp_new_sel[imp_cat_sel][reg_fd] = {}
        for imp_cat_eff in dict_imp_new[imp_cat_sel][reg_fd]:
            dict_imp_new_sel[imp_cat_sel][reg_fd][imp_cat_eff] = {}
            for prod in dict_imp_prod_sort_cons[imp_cat_sel]:
                dict_imp_new_sel[imp_cat_sel][reg_fd][imp_cat_eff][prod] = (
                        dict_imp_new[imp_cat_sel][reg_fd][imp_cat_eff][prod])

dict_imp_new_sel_reg = {}
for imp_cat_sel in dict_imp_new_sel:
    dict_imp_new_sel_reg[imp_cat_sel] = {}
    for reg_fd in dict_imp_new_sel[imp_cat_sel]:
        dict_imp_new_sel_reg[imp_cat_sel][reg_fd] = {}
        for imp_cat_eff in dict_imp_new_sel[imp_cat_sel][reg_fd]:
            dict_imp_new_sel_reg[imp_cat_sel][reg_fd][imp_cat_eff] = {}
            for prod in dict_imp_new_sel[imp_cat_sel][reg_fd][imp_cat_eff]:
                if prod not in dict_imp_new_sel_reg[imp_cat_sel][reg_fd][imp_cat_eff]:
                    dict_imp_new_sel_reg[imp_cat_sel][reg_fd][imp_cat_eff][prod] = 0
                for cntr in dict_imp_new_sel[imp_cat_sel][reg_fd][imp_cat_eff][prod]:
                    dict_imp_new_sel_reg[imp_cat_sel][reg_fd][imp_cat_eff][prod] += (
                            dict_imp_new_sel[imp_cat_sel][reg_fd][imp_cat_eff][prod][cntr])

dict_imp_new_sel_reg_flip = {}
for imp_cat_sel in dict_imp_new_sel_reg:
    if imp_cat_sel not in dict_imp_new_sel_reg_flip:
        dict_imp_new_sel_reg_flip[imp_cat_sel] = {}
    for reg_fd in dict_imp_new_sel_reg[imp_cat_sel]:
        for imp_cat_eff in dict_imp_new_sel_reg[imp_cat_sel][reg_fd]:
            if imp_cat_eff not in dict_imp_new_sel_reg_flip[imp_cat_sel]:
                dict_imp_new_sel_reg_flip[imp_cat_sel][imp_cat_eff] = {}
            for prod_long in (
                    dict_imp_new_sel_reg[imp_cat_sel][reg_fd][imp_cat_eff]):
                prod_short = dict_prod_long_short[prod_long]
                if prod_short not in (
                        dict_imp_new_sel_reg_flip[imp_cat_sel][imp_cat_eff]):
                    dict_imp_new_sel_reg_flip[imp_cat_sel][imp_cat_eff][prod_short] = {}
                dict_imp_new_sel_reg_flip[imp_cat_sel][imp_cat_eff][prod_short][reg_fd] = (
                        dict_imp_new_sel_reg[imp_cat_sel][reg_fd][imp_cat_eff][prod_long])


dict_df_post = {}

plt.close('all')
fig = plt.figure(figsize=cm2inch((16, 1+13*0.4)))

dict_xlim_improv = {}
for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_sel_reg_flip):
    for imp_cat_eff_id, imp_cat_eff in (
            enumerate(dict_imp_new_sel_reg_flip[imp_cat_sel])):
        plot_id = imp_cat_eff_id+1
        plot_loc = 140+plot_id
        ax = fig.add_subplot(plot_loc)
        df_old = pd.DataFrame(dict_imp_sel_reg_cons[imp_cat_eff])
        df_new = pd.DataFrame(
                dict_imp_new_sel_reg_flip[imp_cat_sel][imp_cat_eff])
        df = df_new-df_old
        df = df.reindex(prod_order_cons, axis=1)
        df_color = df.loc['import'] <= 0
        df.T.plot.barh(stacked=True,
                       ax=ax,
                       legend=False,
                       color=[df_color.map({True: 'g', False: 'r'})])
        if plot_id > 1:
            ax.set_yticklabels([])
        plt.locator_params(axis='x', nbins=4)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        xlim = ax.get_xlim()
        xlim_min_magn = 10**np.floor(np.log10(abs(xlim[0])))
        xlim_min_floor = math.floor(xlim[0]/xlim_min_magn)*xlim_min_magn

        if xlim[1] > 0.0:
            xlim_max_magn = 10**np.floor(np.log10(xlim[1]))
            xlim_max_ceil = math.ceil(xlim[1]/xlim_max_magn)*xlim_max_magn
        else:
            xlim_max_ceil = int(xlim[1])
        tup_xlim_min_floor_max_ceil = (xlim_min_floor, xlim_max_ceil)

        if imp_cat_eff not in dict_xlim_improv:
            dict_xlim_improv[imp_cat_eff] = tup_xlim_min_floor_max_ceil
        if xlim_min_floor < dict_xlim_improv[imp_cat_eff][0]:
            xlim_new = tuple([xlim_min_floor,
                              dict_xlim_improv[imp_cat_eff][1]])
            dict_xlim_improv[imp_cat_eff] = xlim_new
        if xlim_max_ceil > dict_xlim_improv[imp_cat_eff][1]:
            xlim_new = tuple([dict_xlim_improv[imp_cat_eff][0], xlim_max_ceil])
            dict_xlim_improv[imp_cat_eff] = xlim_new

plt.close('all')
dict_improv = plot_improv(dict_imp_sel_reg_cons, dict_imp_new_sel_reg_flip, dict_xlim_improv)

for imp_cat_sel in dict_improv:
    fp_sel = dict_imp_cat_fp[imp_cat_sel]
    for imp_cat_eff in dict_improv[imp_cat_sel]:
        fp_eff = dict_imp_cat_fp[imp_cat_eff]
        for prod in dict_improv[imp_cat_sel][imp_cat_eff]:
            imp_ante = dict_imp_sel_reg_cons[imp_cat_eff][prod]['import']
            imp_reduc = dict_improv[imp_cat_sel][imp_cat_eff][prod]
            prod_strip = prod.strip()
            prod_strip_us = prod_strip.replace(' ','_')
            print('{} {} {} {:.2E} {:.2E} {:.0%}'.format(fp_sel, fp_eff, prod_strip_us, imp_ante, imp_reduc, imp_reduc/imp_ante))


dict_pot_imp_agg = {}
for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_sel_reg_flip):
    fp_sel = dict_imp_cat_fp[imp_cat_sel]
    for imp_cat_eff_id, imp_cat_eff in enumerate(
            dict_imp_new_sel_reg_flip[imp_cat_sel]):

        fp_eff = dict_imp_cat_fp[imp_cat_eff]
        if fp_eff not in dict_pot_imp_agg:
            dict_pot_imp_agg[fp_eff] = {}
        if 'Ante' not in dict_pot_imp_agg[fp_eff]:
            df_old = pd.DataFrame(dict_imp_sel_reg_cons[imp_cat_eff])
            df_old_sum = df_old.sum(axis=1)
            dict_pot_imp_agg[fp_eff]['Prior'] = float(
                    df_old_sum['import'])
        df_new = pd.DataFrame(
                dict_imp_new_sel_reg_flip[imp_cat_sel][imp_cat_eff])
        df_new_sum = df_new.sum(axis=1)
        dict_pot_imp_agg[fp_eff][fp_sel] = float(
                df_new_sum['import'])


plot_improv_agg(dict_pot_imp_agg, dict_xlim_agg)

dict_fp_color = {}
dict_fp_color['Carbon'] = 'C3'
dict_fp_color['Material'] = 'C1'
dict_fp_color['Water'] = 'C0'
dict_fp_color['Land'] = 'C2'

dict_pot_imp_agg_rel = {}
for fp_eff_id, fp_eff in enumerate(dict_pot_imp_agg):
    list_imp_rel = []
    list_xticklabel = []
    for fp_sel in dict_pot_imp_agg[fp_eff]:
        imp_abs = dict_pot_imp_agg[fp_eff][fp_sel]
        if fp_sel == 'Prior':
            imp_abs_prior = imp_abs
        else:
            if fp_sel not in dict_pot_imp_agg_rel:
                dict_pot_imp_agg_rel[fp_sel] = {}
            imp_rel = imp_abs/imp_abs_prior
            imp_diff = imp_abs_prior - imp_abs
            dict_pot_imp_agg_rel[fp_sel][fp_eff] = 1-imp_rel

for fp_sel in dict_pot_imp_agg_rel:
    for fp_eff in dict_pot_imp_agg_rel[fp_sel]:
        imp_rel = dict_pot_imp_agg_rel[fp_sel][fp_eff]

plt.close('all')
fig = plt.figure(figsize=cm2inch((16, 16)))


fp_order = ['Carbon', 'Land', 'Water', 'Material']

dict_fp_abr = {}
dict_fp_abr['Carbon'] = 'CF'
dict_fp_abr['Material'] = 'MF'
dict_fp_abr['Water'] = 'WF'
dict_fp_abr['Land'] = 'LF'


for fp_sel_id, fp_sel in enumerate(dict_pot_imp_agg_rel):
    list_imp_rel = []
    list_xticklabel = []
    for fp_eff in fp_order:
        imp_rel = dict_pot_imp_agg_rel[fp_sel][fp_eff]
        print('{} {} {:.0%}'.format(fp_sel, fp_eff, 1-imp_rel))

        list_imp_rel.append(imp_rel)
        list_xticklabel.append(fp_eff)

    plot_id = 220+fp_sel_id+1
    print(plot_id)
    ax = fig.add_subplot(plot_id, projection='polar')
    ax.set_rticks([0.50, 1])  # less radial ticks
    ax.yaxis.set_ticklabels(['50%', '100%'], fontsize=font_size)  # less radial ticks
    ax.set_rlabel_position(0)  # get radial labels away from plotted line
    xtick_count = np.arange(2*math.pi/8, 2*math.pi+2*math.pi/8, 2*math.pi/4)
    ax.set_xticks(xtick_count)
    ax.set_xticklabels(list_xticklabel, fontsize=font_size)
    ax.set_ylim([0,1.0])
    y_val = list_imp_rel
    y_val.append(y_val[0])
    x_val = list(xtick_count)
    x_val.append(x_val[0])
    ax.plot(x_val, y_val, color = 'C2')
    ax_title = 'Optimized {} footprint'.format(fp_sel.lower())
    ax.set_title(ax_title, fontsize=font_size)
plt.tight_layout(pad=3)
fig.savefig('spider_plot.png')