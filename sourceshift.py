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

def get_date():
    date_full = datetime.datetime.now()
    return '{}{:02}{:02}'.format(date_full.year, date_full.month, date_full.day)
date = get_date()

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
    if not os.path.exists(result_dir_path+dir_name):
        os.makedirs(result_dir_path+dir_name+pdf_dir_name)
        os.makedirs(result_dir_path+dir_name+png_dir_name)

mpl.rcParams['mathtext.default'] = 'regular'

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


def cm2inch(tup_cm):
    inch = 2.54
    tup_inch = tuple(i/inch for i in tup_cm)
    return tup_inch


def get_cf(file_path, df_cQ):
    list_imp = []
    with open(file_path) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_imp.append(tuple(row))
    return df_cQ.loc[list_imp]


def get_dict_cf():
    dict_cf = {}
    dict_cf['e'] = get_cf(data_path+e_fp_file_name,
                          dict_eb['cQe'])
    dict_cf['m'] = get_cf(data_path+m_fp_file_name,
                          dict_eb['cQm'])
    dict_cf['r'] = get_cf(data_path+r_fp_file_name,
                          dict_eb['cQr'])
    return dict_cf


def get_dict_df_imp(df_tY):
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


def get_dict_imp_prod_sort(imp_cum_lim):
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


def set_priority():
    print('priority setting')
    dict_xlim = {}


    #   Disaggregate impacts by domestic production, imports from in- and exEU28.
    dict_df_imp_cntr = {}
    dict_imp_cntr = {}
    for cntr_fd in list_reg_fd:
        df_tY_cntr = dict_eb['tY'][cntr_fd].copy()
        df_tY_cntr.loc[cntr_fd] = 0
        df_tY_cntr_fdsum = df_tY_cntr.sum(axis=1)
        # For all countries in EU28, calculate dataframes with absolute impact.
    #    dict_df_imp_cntr[cntr_fd] = get_dict_df_imp(dict_eb,
    #                                                df_tY_cntr_fdsum,
    #                                                dict_cf)

        dict_df_imp_cntr[cntr_fd] = get_dict_df_imp(df_tY_cntr_fdsum)

        # For all countries in EU28, cast dataframes with absolute impact to dict.
        dict_imp_cntr[cntr_fd] = get_dict_imp(dict_df_imp_cntr[cntr_fd])


    dict_imp_prod_sort = get_dict_imp_prod_sort(0.5)

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


    dict_imp_sel_reg = {}
    for cntr_fd in dict_imp_sel:
        for imp_cat_sel in dict_imp_sel[cntr_fd]:
            if imp_cat_sel not in dict_imp_sel_reg:
                dict_imp_sel_reg[imp_cat_sel] = {}
            for imp_cat_eff in dict_imp_sel[cntr_fd][imp_cat_sel]:
                if imp_cat_eff not in dict_imp_sel_reg[imp_cat_sel]:
                    dict_imp_sel_reg[imp_cat_sel][imp_cat_eff] = {}
                for prod in dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff]:
                    if prod not in (
                            dict_imp_sel_reg[imp_cat_sel][imp_cat_eff]):
                        dict_imp_sel_reg[imp_cat_sel][imp_cat_eff][prod] = 0
                    for cntr in dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod]:
                        imp_abs = dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod][cntr]
                        if cntr_fd is not cntr:
                            dict_imp_sel_reg[imp_cat_sel][imp_cat_eff][prod] += imp_abs

    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_sel_reg):
        fig_y_size = len(dict_imp_sel_reg[imp_cat_sel][imp_cat_sel])
        plt.close('all')
        fig = plt.figure(figsize=cm2inch((16, 1+fig_y_size*0.4)))
        for imp_cat_eff_id, imp_cat_eff in (
                enumerate(dict_imp_sel_reg[imp_cat_sel])):
            plot_id = imp_cat_eff_id+1
            plot_loc = 140+plot_id
            ax = fig.add_subplot(plot_loc)
            df = pd.DataFrame(dict_imp_sel_reg[imp_cat_sel][imp_cat_eff], index=['import'])
            df.rename(columns=dict_prod_long_short, inplace=True)
            df.T.plot.barh(stacked=True,
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


    analysis_name = 'priority_setting'
    dict_prod_order = {}
    for imp_cat_sel in dict_imp_sel_reg:
        df = pd.DataFrame(dict_imp_sel_reg[imp_cat_sel][imp_cat_sel], index=['import'])
        df.rename(columns=dict_prod_long_short, inplace=True)
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
        prod_order = dict_prod_order[imp_cat_sel]
        plot_id = imp_cat_sel_id+1
        plot_loc = 220+plot_id
        ax = fig.add_subplot(plot_loc)
        fp = dict_imp_cat_fp[imp_cat_sel]
        unit = dict_imp_cat_unit[imp_cat_sel[-1]]
        ax.set_xlabel('{} [{}]'.format(fp, unit), fontsize=font_size)
        df = pd.DataFrame(dict_imp_sel_reg[imp_cat_sel][imp_cat_sel], index=['import'])
        df.rename(columns=dict_prod_long_short, inplace=True)
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

        df.T.plot.barh(stacked=True,
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


def calc_improv():
    dict_imp_new_reg = calc_new_fp()

    prod_order_cons = []
    with open(data_path+prod_order_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            prod_order_cons.append(row[0])
    prod_order_cons.reverse()


    print('plot_improv')
    plt.close('all')
    fig = plt.figure(figsize=cm2inch((16, 1+13*0.4)))
    dict_imp_prod_sort_full = get_dict_imp_prod_sort(1.1)

    dict_xlim_improv = {}
    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_reg):
        for imp_cat_eff_id, imp_cat_eff in (
                enumerate(dict_imp_new_reg[imp_cat_sel])):
            plot_id = imp_cat_eff_id+1
            plot_loc = 140+plot_id
            ax = fig.add_subplot(plot_loc)
            df_old = pd.DataFrame(dict_imp_prod_sort_full[imp_cat_eff], index=['import'])
            df_new = pd.DataFrame(
                    dict_imp_new_reg[imp_cat_sel][imp_cat_eff], index=['import'])
            df = df_new-df_old
            df.rename(columns = dict_prod_long_short, inplace=True)
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

    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_reg):

        fig = plt.figure(
                figsize=cm2inch(
                        (16,
                         len(prod_order_cons)*.4+2)))
        for imp_cat_eff_id, imp_cat_eff in (
                enumerate(dict_imp_new_reg[imp_cat_sel])):
            plot_id = imp_cat_eff_id+1
            plot_loc = 140+plot_id
            ax = fig.add_subplot(plot_loc)
            fp = dict_imp_cat_fp[imp_cat_eff]
            ax.set_title(fp, fontsize=font_size)
            unit = dict_imp_cat_unit[imp_cat_eff[-1]]
            ax.set_xlabel(unit, fontsize=font_size)
            ax.set_xlim(dict_xlim_improv[imp_cat_eff])
            df_old = pd.DataFrame(dict_imp_prod_sort_full[imp_cat_eff], index=['import'])

            df_new = pd.DataFrame(
                    dict_imp_new_reg[imp_cat_sel][imp_cat_eff], index=['import'])
            df = df_new-df_old
            df.rename(columns = dict_prod_long_short, inplace=True)
            df = df.reindex(prod_order_cons, axis=1)
            df_color = df.loc['import'] <= 0
            df.T.plot.barh(stacked=True,
                           ax=ax,
                           legend=False,
                           color=[df_color.map({True: 'g', False: 'r'})],
                           width=0.8)

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



def calc_improv_agg():
    dict_imp_new_reg = calc_new_fp()

    print('plot_improv_agg')
    dict_imp_prod_sort_full = get_dict_imp_prod_sort(1.1)

    dict_pot_imp_agg = {}
    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_reg):
        fp_sel = dict_imp_cat_fp[imp_cat_sel]
        for imp_cat_eff_id, imp_cat_eff in enumerate(
                dict_imp_new_reg[imp_cat_sel]):

            fp_eff = dict_imp_cat_fp[imp_cat_eff]
            if fp_eff not in dict_pot_imp_agg:
                dict_pot_imp_agg[fp_eff] = {}
            if 'Ante' not in dict_pot_imp_agg[fp_eff]:
                df_old = pd.DataFrame(dict_imp_prod_sort_full[imp_cat_eff], index=['import'])
                df_old_sum = df_old.sum(axis=1)
                dict_pot_imp_agg[fp_eff]['Prior'] = float(
                        df_old_sum['import'])
            df_new = pd.DataFrame(
                    dict_imp_new_reg[imp_cat_sel][imp_cat_eff], index=['import'])
            df_new_sum = df_new.sum(axis=1)
            dict_pot_imp_agg[fp_eff][fp_sel] = float(
                    df_new_sum['import'])


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
                dict_pot_imp_agg_rel[fp_sel][fp_eff] = 1-imp_rel

    for fp_sel in dict_pot_imp_agg_rel:
        for fp_eff in dict_pot_imp_agg_rel[fp_sel]:
            imp_rel = dict_pot_imp_agg_rel[fp_sel][fp_eff]

    plt.close('all')
    fig = plt.figure(figsize=cm2inch((16, 16)))

    fp_order = ['Carbon', 'Land', 'Water', 'Material']

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
        ax.set_rticks([0.50, 1])
        ax.yaxis.set_ticklabels(['50%', '100%'], fontsize=font_size)
        ax.set_rlabel_position(0)
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
    plot_name = 'spider_plot'
    reduc_agg_dir_path = result_dir_path+reduc_agg_dir_name
    pdf_file_name = plot_name+'.pdf'
    pdf_dir_path = reduc_agg_dir_path+pdf_dir_name
    pdf_file_path = pdf_dir_path+pdf_file_name
    fig.savefig(pdf_file_path)
    png_file_name = plot_name+'.png'
    png_dir_path = reduc_agg_dir_path+png_dir_name
    png_file_path = png_dir_path+png_file_name
    fig.savefig(png_file_path)

def shift_source():
    print('sourceshift')

    dict_cntr_short_long = {}
    with open(data_path+country_code_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            cntr_short = row[0]
            cntr_long = row[1]
            dict_cntr_short_long[cntr_short] = cntr_long

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


    dict_imp_prod_sort_full = get_dict_imp_prod_sort(1.1)
    dict_tY = df_tY_eu28_fdsum.to_dict()
    dict_tY_prod_cntr = {}
    for tup_cntr_prod in dict_tY:
        cntr, prod = tup_cntr_prod
        if prod not in dict_tY_prod_cntr:
            dict_tY_prod_cntr[prod] = {}
        dict_tY_prod_cntr[prod][cntr] = dict_tY[tup_cntr_prod]

    dict_imp_prod_cntr_sort = {}
    for imp_cat in dict_imp_prod_sort_full:
        dict_imp_prod_cntr_sort[imp_cat] = OrderedDict()
        dict_imp_pME_prod_cntr = {}
        for prod in dict_imp_prod_sort_full[imp_cat]:
            dict_imp_prod_cntr_sort[imp_cat][prod] = OrderedDict()
            dict_imp_prod_cntr = dict_imp[imp_cat][prod]
            dict_imp_pME_prod_cntr[prod] = {}
            imp_abs_prod = dict_imp_prod_sort_full[imp_cat][prod]

            for cntr in dict_imp_prod_cntr:
                imp_abs_prod_cntr = dict_imp_prod_cntr[cntr]
                dict_imp_pME_prod_cntr[prod][cntr] = (
                        dict_imp_pME[imp_cat][prod][cntr])
            list_imp_pME_prod_cntr_sort = sorted(
                        dict_imp_pME_prod_cntr[prod].items(),
                        key=operator.itemgetter(1))
            imp_cum_prod_cntr = 0
            for tup_cntr_imp_pME in list_imp_pME_prod_cntr_sort:
                cntr, imp_pME_prod_cntr = tup_cntr_imp_pME
                imp_abs_prod_cntr = dict_imp_prod_cntr[cntr]
                if imp_abs_prod > 0:
                    imp_rel_prod_cntr = imp_abs_prod_cntr/imp_abs_prod
                else:
                    imp_rel_prod_cntr = 0.0
                imp_cum_prod_cntr += imp_rel_prod_cntr
                dict_imp_prod_cntr_sort[imp_cat][prod][cntr] = imp_abs_prod_cntr


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
                imp_abs = dict_imp_prod_cntr_sort[imp_cat][prod][cntr]
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
            fig.tight_layout(pad=0)
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
            fp_prod = fp_lower+'_'+prod_short_lower_strip_us
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
    return dict_y_new
#   Calculate new impact
def calc_new_fp():
    print('calc_new_fp')
    dict_df_tY_import_new = {}
    for imp_cat_id, imp_cat in enumerate(dict_y_new):
        dict_df_tY_import_new[imp_cat] = {}
        for prod in dict_y_new[imp_cat]:
            for cntr in dict_y_new[imp_cat][prod]:
                dict_df_tY_import_new[imp_cat][(cntr, prod)] = (
                        dict_y_new[imp_cat][prod][cntr])

    #   For each impact category make new final demand DataFrames
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
        dict_df_tY_new[imp_cat] = df_tY_eu28_import

    dict_imp_new = {}
    for imp_cat in dict_df_tY_new:
        dict_imp_new[imp_cat] = {}
        df_tY_new = dict_df_tY_new[imp_cat]
#        dict_df_imp_new = get_dict_df_imp(dict_eb,
#                                          df_tY_new,
#                                          dict_cf)

        dict_df_imp_new = get_dict_df_imp(df_tY_new)

        # Cast dataframes with absolute impact to dict.
        dict_imp_new[imp_cat]= get_dict_imp(dict_df_imp_new)

    dict_imp_new_reg = {}
    for imp_cat_sel in dict_imp_new:
        dict_imp_new_reg[imp_cat_sel] = {}
        for imp_cat_eff in dict_imp_new[imp_cat_sel]:
            if imp_cat_eff not in dict_imp_new_reg[imp_cat_sel]:
                dict_imp_new_reg[imp_cat_sel][imp_cat_eff] = {}
            for prod in dict_imp_new[imp_cat_sel][imp_cat_eff]:
                if prod not in dict_imp_new_reg[imp_cat_sel][imp_cat_eff]:
                    dict_imp_new_reg[imp_cat_sel][imp_cat_eff][prod] = 0
                for cntr in dict_imp_new[imp_cat_sel][imp_cat_eff][prod]:
                    dict_imp_new_reg[imp_cat_sel][imp_cat_eff][prod] += (
                            dict_imp_new[imp_cat_sel][imp_cat_eff][prod][cntr])

    return dict_imp_new_reg
###############################################################################
plt.close('all')


#   Load shortened product names for plotting.
def get_dict_prod_long_short():
    list_prod_long = []
    with open(data_path+prod_long_file_name) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_prod_long.append(row[0])

    list_prod_short = []
    with open(data_path+prod_short_file_name) as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            list_prod_short.append(row[0])

    dict_prod_long_short = {}
    for prod_id, prod_long in enumerate(list_prod_long):
        prod_long = list_prod_long[prod_id]
        prod_short = list_prod_short[prod_id]
        dict_prod_long_short[prod_long] = prod_short
    return dict_prod_long_short
dict_prod_long_short = get_dict_prod_long_short()

def get_dict_imp_cat_fp():
    dict_imp_cat_fp = {}
    with open(data_path+cf_long_footprint_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            imp_cat = tuple(row[:-1])
            fp = row[-1]
            dict_imp_cat_fp[imp_cat] = fp
    return dict_imp_cat_fp
dict_imp_cat_fp = get_dict_imp_cat_fp()

def get_dict_imp_cat_magnitude():
    dict_imp_cat_magnitude = {}
    with open(data_path+cf_magnitude_file_name, 'r') as read_file:
        csv_file = csv.reader(read_file, delimiter='\t')
        for row in csv_file:
            imp_cat = tuple(row[:-1])
            magnitude = int(row[-1])
            dict_imp_cat_magnitude[imp_cat] = magnitude
    return dict_imp_cat_magnitude
dict_imp_cat_magnitude = get_dict_imp_cat_magnitude()

#   Load EXIOBASE.
if dict_eb_file_name in os.listdir(data_path):
    dict_eb = pickle.load(open(data_path+dict_eb_file_name, 'rb'))
else:
    dict_eb = eb.process(eb.parse())
    save_eb = True
    if save_eb:
        pickle.dump(dict_eb, open(data_path+dict_eb_file_name, 'wb'))

#   Get characterisation factors.
dict_cf = get_dict_cf()

#   Generate dictionary with list of EU28 countries.
def get_list_reg_fd(reg_fd):
    if reg_fd == 'EU28':
        with open(data_path+eu28_file_name) as read_file:
            csv_file = csv.reader(read_file, delimiter='\t')
            list_reg_fd = []
            for row in csv_file:
                list_reg_fd.append(row[0])
    else:
        list_reg_fd = [reg_fd]
    return list_reg_fd


font_size = 8.0
###############################################################################
# Priority setting
# Get dictionary with footprints.
# final demand matrix of EU28
list_reg_fd = get_list_reg_fd('EU28')
df_tY_eu28 = dict_eb['tY'][list_reg_fd].copy()
# final demand vector of EU28
#df_tY_eu28_full = df_tY_eu28.sum(axis=1)

for cntr in list_reg_fd:
    df_tY_eu28.loc[cntr, cntr] = 0
df_tY_eu28_fdsum = df_tY_eu28.sum(axis=1)
#dict_df_imp_full = get_dict_df_imp(df_tY_eu28_full)
dict_df_imp = get_dict_df_imp(df_tY_eu28_fdsum)



dict_imp_cat_sum = {}
for imp_cat in dict_df_imp:
    dict_df_imp_cat_sum = dict_df_imp[imp_cat].sum(axis=1).to_dict()
    for key in dict_df_imp_cat_sum.keys():
        dict_imp_cat_sum[key] = dict_df_imp_cat_sum[key]
#   Cast dataframes with absolute impact to dict.
dict_imp = get_dict_imp(dict_df_imp)


dict_imp_cat_unit = {}
dict_imp_cat_unit['kg CO2 eq.'] = r'$Pg\/CO_2\/eq.$'
dict_imp_cat_unit['kt'] = r'$Gt$'
dict_imp_cat_unit['Mm3'] = r'$Mm^3$'
dict_imp_cat_unit['km2'] = r'$Gm^2$'

###############################################################################
# Plot priority setting
set_priority()
dict_y_new = shift_source()
calc_improv()
calc_improv_agg()