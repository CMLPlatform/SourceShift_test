# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:18:15 2018

@author: bfdeboer
"""

from collections import OrderedDict

import csv
import math
import matplotlib.collections as mpl_col
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pandas as pd
import pickle

import cfg
import exiobase as eb


def calc_priority(dict_df_imp, imp_cum_lim):
    dict_priority = OrderedDict()
    for cat in dict_df_imp:
        df_imp_prod = dict_df_imp[cat].sum(axis=1, level=1)
        dict_imp_prod = df_imp_prod.T.to_dict()
        dict_imp_prod_sum = dict_df_imp[cat].sum(axis=1).to_dict()
        for imp_cat in dict_imp_prod:
            dict_priority[imp_cat] = OrderedDict()
            list_imp_sort = sorted(dict_imp_prod[imp_cat].items(),
                                   key=operator.itemgetter(1), reverse=True)
            imp_cum = 0
            bool_add = True
            for tup_prod_abs_id, tup_prod_abs in enumerate(list_imp_sort):
                (prod, imp_abs) = tup_prod_abs
                imp_rel = imp_abs/dict_imp_prod_sum[imp_cat]
                imp_cum = imp_cum + imp_rel
                if imp_cum < imp_cum_lim:
                    dict_priority[imp_cat][prod] = imp_abs
                elif bool_add:
                    dict_priority[imp_cat][prod] = imp_abs
                    bool_add = False
    return dict_priority


def plot_priority(dict_priority):
    """ Find highest contributing product for each footprint



    """
    # For each footprint, select highest contributing products
#    dict_imp_prod_sort = get_dict_imp_prod_sort(dict_df_imp,
#                                                    cfg.imp_cum_lim_priority)

    dict_imp_cat_fp = get_dict_imp_cat_fp()
    dict_prod_long_short = get_dict_prod_long_short()
    dict_imp_cat_magnitude = get_dict_imp_cat_magnitude()


    analysis_name = 'priority_setting'

    plt.close('all')
    dict_imp_cat_unit = get_dict_imp_cat_unit()
#    fig_y_size_max = 0
#    for imp_cat_sel in dict_imp_prod_sort:
#        fig_y_size = len(dict_imp_prod_sort[imp_cat_sel])
#        if fig_y_size > fig_y_size_max:
#            fig_y_size_max = fig_y_size
    fig = plt.figure(figsize=cm2inch((16, 8)))
    for imp_cat_id, imp_cat in enumerate(dict_priority):
#        fig_y_size = len(dict_imp_prod_sort[imp_cat_sel])
        plot_id = imp_cat_id+1
        plot_loc = 220+plot_id
        ax = fig.add_subplot(plot_loc)
        fp = dict_imp_cat_fp[imp_cat]
        unit = dict_imp_cat_unit[imp_cat[-1]]
        ax.set_xlabel('{} [{}]'.format(fp, unit))
        df = pd.DataFrame(dict_priority[imp_cat],
                          index=['import'])
        df.rename(columns=dict_prod_long_short, inplace=True)
        df_column_order = list(df.columns)
        df_column_order.reverse()
        df = df.reindex(df_column_order, axis=1)
        column_name_dummy = ''
        prod_order_dummy = df_column_order
        while len(df.T) < 9:
            df[column_name_dummy] = 0
            prod_order_dummy.reverse()
            prod_order_dummy.append(column_name_dummy)
            prod_order_dummy.reverse()
            df = df.reindex(df_column_order, axis=1)
            column_name_dummy += ' '

        df.T.plot.barh(stacked=True,
                       ax=ax,
                       legend=False,
                       color='C0',
                       width=0.8)

        yticklabels = ax.get_yticklabels()
        ax.set_yticklabels(yticklabels)

        plt.locator_params(axis='x', nbins=1)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        xlim = ax.get_xlim()
        xlim_max_magn = 10**np.floor(np.log10(xlim[1]))
        xlim_max_ceil = math.ceil(xlim[1]/xlim_max_magn)*xlim_max_magn
        tup_xlim_max_ceil = (int(xlim[0]), xlim_max_ceil)

#        ax.set_xlim(dict_xlim[imp_cat_sel])
        ax.set_xlim(tup_xlim_max_ceil)
        xtick_magnitude = dict_imp_cat_magnitude[imp_cat]

#        list_xtick = [i/xtick_magnitude for i in dict_xlim[imp_cat_sel]]
        list_xtick = [i/xtick_magnitude for i in tup_xlim_max_ceil]
        list_xtick[0] = int(list_xtick[0])
#        ax.set_xticks(list(dict_xlim[imp_cat_sel]))
        ax.set_xticks(list(tup_xlim_max_ceil))
        ax.set_xticklabels(list_xtick)

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

    priority_setting_dir_path = cfg.result_dir_path+cfg.priority_setting_dir_name
    fig_file_name = analysis_name+'.pdf'
    pdf_dir_path = priority_setting_dir_path+cfg.pdf_dir_name
    fig_file_path = pdf_dir_path+fig_file_name
    fig.savefig(fig_file_path)

    fig_file_name = analysis_name+'.png'
    png_dir_path = priority_setting_dir_path+cfg.png_dir_name
    fig_file_path = png_dir_path+fig_file_name
    fig.savefig(fig_file_path)


def cm2inch(tup_cm):
    inch = 2.54
    tup_inch = tuple(i/inch for i in tup_cm)
    return tup_inch


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
#    list_dir_name.append(cfg.priority_setting_dir_name)
    list_dir_name.append(cfg.shift_dir_name)
    list_dir_name.append(cfg.reduc_agg_dir_name)
    list_dir_name.append(cfg.reduc_dir_name)

    for dir_name in list_dir_name:
        if not os.path.exists(cfg.result_dir_path+dir_name):
            os.makedirs(cfg.result_dir_path+dir_name+cfg.pdf_dir_name)
            os.makedirs(cfg.result_dir_path+dir_name+cfg.png_dir_name)



class Priority:

    dict_priority = OrderedDict()
    dict_imp_cat_fp = get_dict_imp_cat_fp()
    dict_prod_long_short = get_dict_prod_long_short()

    def __init__(self):
        print('Creating instance of Priority class.')
        self.makedirs()

    def calc(self, dict_df_imp, imp_cum_lim):
        for cat in dict_df_imp:
            df_imp_prod = dict_df_imp[cat].sum(axis=1, level=1)
            dict_imp_prod = df_imp_prod.T.to_dict()
            dict_imp_prod_sum = dict_df_imp[cat].sum(axis=1).to_dict()
            for imp_cat in dict_imp_prod:
                self.dict_priority[imp_cat] = OrderedDict()
                list_imp_sort = sorted(dict_imp_prod[imp_cat].items(),
                                       key=operator.itemgetter(1),
                                       reverse=True)
                imp_cum = 0
                bool_add = True
                for tup_prod_abs_id, tup_prod_abs in enumerate(list_imp_sort):
                    (prod, imp_abs) = tup_prod_abs
                    imp_rel = imp_abs/dict_imp_prod_sum[imp_cat]
                    imp_cum = imp_cum + imp_rel
                    if imp_cum < imp_cum_lim:
                        self.dict_priority[imp_cat][prod] = imp_abs
                    elif bool_add:
                        self.dict_priority[imp_cat][prod] = imp_abs
                        bool_add = False

    def log(self):
        with open('log.txt', 'w') as write_file:
            csv_file = csv.writer(write_file,
                                  delimiter = '\t',
                                  lineterminator = '\n')
            csv_file.writerow(['Footprint','Product'])
            for imp_cat in self.dict_priority:
                csv_file.writerow([])
                fp = self.dict_imp_cat_fp[imp_cat]
                for prod in self.dict_priority[imp_cat]:
                    csv_file.writerow([fp, prod])


    def plot(self):
        """ Find highest contributing product for each footprint



        """
        # For each footprint, select highest contributing products
        dict_imp_cat_magnitude = get_dict_imp_cat_magnitude()


        analysis_name = 'priority_setting'

        plt.close('all')
        dict_imp_cat_unit = get_dict_imp_cat_unit()
        fig = plt.figure(figsize=cm2inch((16, 8)))
        for imp_cat_id, imp_cat in enumerate(self.dict_priority):
            plot_id = imp_cat_id+1
            plot_loc = 220+plot_id
            ax = fig.add_subplot(plot_loc)
            fp = self.dict_imp_cat_fp[imp_cat]
            unit = dict_imp_cat_unit[imp_cat[-1]]
            ax.set_xlabel('{} [{}]'.format(fp, unit))
            df = pd.DataFrame(self.dict_priority[imp_cat],
                              index=['import'])
            df.rename(columns=self.dict_prod_long_short, inplace=True)
            df_column_order = list(df.columns)
            df_column_order.reverse()
            df = df.reindex(df_column_order, axis=1)
            column_name_dummy = ''
            prod_order_dummy = df_column_order
            while len(df.T) < 9:
                df[column_name_dummy] = 0
                prod_order_dummy.reverse()
                prod_order_dummy.append(column_name_dummy)
                prod_order_dummy.reverse()
                df = df.reindex(df_column_order, axis=1)
                column_name_dummy += ' '

            df.T.plot.barh(stacked=True,
                           ax=ax,
                           legend=False,
                           color='C0',
                           width=0.8)

            yticklabels = ax.get_yticklabels()
            ax.set_yticklabels(yticklabels)

            plt.locator_params(axis='x', nbins=1)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

            xlim = ax.get_xlim()
            xlim_max_magn = 10**np.floor(np.log10(xlim[1]))
            xlim_max_ceil = math.ceil(xlim[1]/xlim_max_magn)*xlim_max_magn
            tup_xlim_max_ceil = (int(xlim[0]), xlim_max_ceil)

            ax.set_xlim(tup_xlim_max_ceil)
            xtick_magnitude = dict_imp_cat_magnitude[imp_cat]

            list_xtick = [i/xtick_magnitude for i in tup_xlim_max_ceil]
            list_xtick[0] = int(list_xtick[0])
            ax.set_xticks(list(tup_xlim_max_ceil))
            ax.set_xticklabels(list_xtick)

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

        priority_setting_dir_path = (cfg.result_dir_path
                                     +cfg.priority_setting_dir_name)
        fig_file_name = analysis_name+'.pdf'
        pdf_dir_path = priority_setting_dir_path+cfg.pdf_dir_name
        fig_file_path = pdf_dir_path+fig_file_name
        fig.savefig(fig_file_path)

        fig_file_name = analysis_name+'.png'
        png_dir_path = priority_setting_dir_path+cfg.png_dir_name
        fig_file_path = png_dir_path+fig_file_name
        fig.savefig(fig_file_path)


    def makedirs(self):
        print('Making output directories.')
        for output_dir_name in cfg.list_output_dir_name:
            try:
                output_dir_path = (cfg.result_dir_path
                                   +cfg.priority_setting_dir_name
                                   +output_dir_name)
                os.makedirs(output_dir_path)
            except FileExistsError as e:
                print('\tOutput directory already exists:\n'
                      '\t{}\n'
                      '\tThis run will overwrite previous output.'.format(
                              output_dir_path))


class SourceShift():

    dict_imp_cat_fp = get_dict_imp_cat_fp()
    dict_prod_long_short = get_dict_prod_long_short()
    dict_y_new = {}
    dict_source_shift = {}
    def calc(self, dict_cf, dict_eb, dict_imp, df_tY_eu28):
        print('calc')
        dict_df_imp_pME = {}
        dict_df_imp_pME['e'] = dict_cf['e'].dot(dict_eb['cRe']).dot(dict_eb['cL'])
        dict_df_imp_pME['m'] = dict_cf['m'].dot(dict_eb['cRm']).dot(dict_eb['cL'])
        dict_df_imp_pME['r'] = dict_cf['r'].dot(dict_eb['cRr']).dot(dict_eb['cL'])
        dict_imp_pME = get_dict_imp(dict_df_imp_pME)

        dict_imp_prod_cntr_sort = {}
        for imp_cat in dict_imp:
            dict_imp_prod_cntr_sort[imp_cat] = OrderedDict()
            for prod in dict_imp[imp_cat]:
                dict_imp_prod_cntr_sort[imp_cat][prod] = OrderedDict()
                list_imp_pME_prod_cntr_sort = sorted(
                            dict_imp_pME[imp_cat][prod].items(),
                            key=operator.itemgetter(1))
                for tup_cntr_imp_pME in list_imp_pME_prod_cntr_sort:
                    cntr, imp_pME_prod_cntr = tup_cntr_imp_pME
                    dict_imp_prod_cntr_sort[imp_cat][prod][cntr] = (
                            dict_imp[imp_cat][prod][cntr])

        df_tY_eu28_cntr = df_tY_eu28.sum(axis=1, level=0)
        dict_tY_eu28_cntr = df_tY_eu28_cntr.to_dict()
        dict_tY_eu28_cntr_import = {}
        for cntr_fd in dict_tY_eu28_cntr:
            for tup_cntr_prod in dict_tY_eu28_cntr[cntr_fd]:
                cntr, prod = tup_cntr_prod
                if prod not in dict_tY_eu28_cntr_import:
                    dict_tY_eu28_cntr_import[prod] = {}

                if cntr not in dict_tY_eu28_cntr_import[prod]:
                    dict_tY_eu28_cntr_import[prod][cntr] = 0

                if cntr_fd is not cntr:
                    dict_tY_eu28_cntr_import[prod][cntr] += (
                            dict_tY_eu28_cntr[cntr_fd][tup_cntr_prod])

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
            dict_tY_world_ex_prod_cntr[prod][cntr] = (
                    dict_tY_world_ex[tup_cntr_prod])

        for imp_cat in dict_imp_prod_cntr_sort:
            self.dict_source_shift[imp_cat] = {}
            for prod in dict_imp_prod_cntr_sort[imp_cat]:
                self.dict_source_shift[imp_cat]
                self.dict_source_shift[imp_cat][prod] = {}
                for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                    imp_pME_prod_cntr = dict_imp_pME[imp_cat][prod][cntr]
                    y_prod_cntr = dict_tY_eu28_cntr_import[prod][cntr]
                    x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
                    if x_prod_cntr >= cfg.x_prod_cntr_min:
                        self.dict_source_shift[imp_cat][prod][cntr] = {}
                        self.dict_source_shift[imp_cat][prod][cntr]['imp_pME'] = (
                                imp_pME_prod_cntr)
                        self.dict_source_shift[imp_cat][prod][cntr]['export'] = (
                                x_prod_cntr)
                        self.dict_source_shift[imp_cat][prod][cntr]['EU_import_old'] = (
                                y_prod_cntr)
                        self.dict_source_shift[imp_cat][prod][cntr]['EU_import_new'] = (
                                                        0)

        dict_tY_prod = {}
        for prod in dict_tY_eu28_cntr_import:
            dict_tY_prod[prod] = 0
            for cntr in dict_tY_eu28_cntr_import[prod]:
                dict_tY_prod[prod] += dict_tY_eu28_cntr_import[prod][cntr]

        for imp_cat in dict_imp_prod_cntr_sort:
            self.dict_y_new[imp_cat] = {}
            for prod in dict_imp_prod_cntr_sort[imp_cat]:
                y_prod = dict_tY_prod[prod]
                self.dict_y_new[imp_cat][prod] = {}
                for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                    imp_pME_prod_cntr = dict_imp_pME[imp_cat][prod][cntr]
                    y_prod_cntr = dict_tY_eu28_cntr_import[prod][cntr]
                    x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
                    if x_prod_cntr >= cfg.x_prod_cntr_min:
                        if x_prod_cntr < y_prod:
                            self.dict_y_new[imp_cat][prod][cntr] = x_prod_cntr
                            self.dict_source_shift[imp_cat][prod][cntr]['EU_import_new'] = (
                                                            x_prod_cntr)
                            y_prod -= x_prod_cntr
                        elif y_prod > 0:
                            self.dict_source_shift[imp_cat][prod][cntr]['EU_import_new'] = (
                                                            y_prod)
                            self.dict_y_new[imp_cat][prod][cntr] = y_prod
                            y_prod -= y_prod


    def plot(self, dict_cf, dict_eb, dict_imp, df_tY_eu28):
        print('plot')
        print('sourceshift')
        dict_cntr_short_long = get_dict_cntr_short_long()

        dict_df_imp_pME = {}
        dict_df_imp_pME['e'] = dict_cf['e'].dot(dict_eb['cRe']).dot(dict_eb['cL'])
        dict_df_imp_pME['m'] = dict_cf['m'].dot(dict_eb['cRm']).dot(dict_eb['cL'])
        dict_df_imp_pME['r'] = dict_cf['r'].dot(dict_eb['cRr']).dot(dict_eb['cL'])
        dict_imp_pME = get_dict_imp(dict_df_imp_pME)

        dict_imp_prod_cntr_sort = {}
        for imp_cat in dict_imp:
            dict_imp_prod_cntr_sort[imp_cat] = OrderedDict()
            for prod in dict_imp[imp_cat]:
                dict_imp_prod_cntr_sort[imp_cat][prod] = OrderedDict()
                list_imp_pME_prod_cntr_sort = sorted(
                            dict_imp_pME[imp_cat][prod].items(),
                            key=operator.itemgetter(1))
                for tup_cntr_imp_pME in list_imp_pME_prod_cntr_sort:
                    cntr, imp_pME_prod_cntr = tup_cntr_imp_pME
                    dict_imp_prod_cntr_sort[imp_cat][prod][cntr] = (
                            dict_imp[imp_cat][prod][cntr])

#        df_tY_eu28_cntr = df_tY_eu28.sum(axis=1, level=0)
#        dict_tY_eu28_cntr = df_tY_eu28_cntr.to_dict()
#        dict_tY_eu28_cntr_import = {}
#        for cntr_fd in dict_tY_eu28_cntr:
#            for tup_cntr_prod in dict_tY_eu28_cntr[cntr_fd]:
#                cntr, prod = tup_cntr_prod
#                if prod not in dict_tY_eu28_cntr_import:
#                    dict_tY_eu28_cntr_import[prod] = {}
#
#                if cntr not in dict_tY_eu28_cntr_import[prod]:
#                    dict_tY_eu28_cntr_import[prod][cntr] = 0
#
#                if cntr_fd is not cntr:
#                    dict_tY_eu28_cntr_import[prod][cntr] += (
#                            dict_tY_eu28_cntr[cntr_fd][tup_cntr_prod])

#        df_tY_world = dict_eb['tY'].copy()
#        list_cntr = list(df_tY_world.columns.get_level_values(0))
#        for cntr in list_cntr:
#            df_tY_world.loc[cntr, cntr] = 0
#        df_tY_world_ex = df_tY_world.sum(axis=1)
#        dict_tY_world_ex = df_tY_world_ex.to_dict()
#        dict_tY_world_ex_prod_cntr = {}
#        for tup_cntr_prod in dict_tY_world_ex:
#            cntr, prod = tup_cntr_prod
#            if prod not in dict_tY_world_ex_prod_cntr:
#                dict_tY_world_ex_prod_cntr[prod] = {}
#            dict_tY_world_ex_prod_cntr[prod][cntr] = (
#                    dict_tY_world_ex[tup_cntr_prod])

#        dict_imp_cat_prod_imp_abs_sum = {}
        dict_imp_cat_prod_cntr_sort_trunc = {}
        for imp_cat in dict_imp_prod_cntr_sort:
#            dict_imp_cat_prod_imp_abs_sum[imp_cat] = {}
            dict_imp_cat_prod_cntr_sort_trunc[imp_cat] = {}
            for prod in dict_imp_prod_cntr_sort[imp_cat]:
                imp_abs_sum = 0
                for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                    imp_abs = dict_imp_prod_cntr_sort[imp_cat][prod][cntr]
                    imp_abs_sum += imp_abs
#                dict_imp_cat_prod_imp_abs_sum[imp_cat][prod] = imp_abs_sum
                list_imp_cat_prod_sort = sorted(
                        dict_imp_prod_cntr_sort[imp_cat][prod].items(),
                        key=operator.itemgetter(1), reverse=True)
                list_imp_cat_prod_sort_trunc = []
#                imp_abs_sum = dict_imp_cat_prod_imp_abs_sum[imp_cat][prod]
                if imp_abs_sum > 0:
                    imp_cum = 0
                    bool_add = True
                    for tup_cntr_imp in list_imp_cat_prod_sort:
                        cntr, imp_abs = tup_cntr_imp
                        imp_rel = imp_abs/imp_abs_sum
                        imp_cum += imp_rel
                        if imp_cum <= cfg.imp_cum_lim_source_shift:
                            list_imp_cat_prod_sort_trunc.append(cntr)
                        elif bool_add:
                            list_imp_cat_prod_sort_trunc.append(cntr)
                            bool_add = False
                dict_imp_cat_prod_cntr_sort_trunc[imp_cat][prod] = (
                        list_imp_cat_prod_sort_trunc)

        dict_lim = {}
        dict_ax = {}
        for imp_cat in self.dict_source_shift:
            dict_lim[imp_cat] = {}
            dict_ax[imp_cat] = {}
            for prod in self.dict_source_shift[imp_cat]:
                plt.close('all')
                x_start = 0
                y_start = 0
                list_rect_y = []
                list_rect_x = []
                fig = plt.figure(figsize=cm2inch((16, 8)))
                ax = plt.gca()
                for cntr in self.dict_source_shift[imp_cat][prod]:
                    imp_pME_prod_cntr = self.dict_source_shift[imp_cat][prod][cntr]['imp_pME']
                    y_prod_cntr = self.dict_source_shift[imp_cat][prod][cntr]['EU_import_old']
                    x_prod_cntr = self.dict_source_shift[imp_cat][prod][cntr]['export']

                    if cntr in (
                            dict_imp_cat_prod_cntr_sort_trunc[imp_cat][prod]):
                        cntr_long = dict_cntr_short_long[cntr]
                        plt.text(x_start+y_prod_cntr/2,
                                 y_start+imp_pME_prod_cntr,
                                 ' '+cntr_long,
                                 rotation=90,
                                 verticalalignment='bottom',
                                 horizontalalignment='center',
                                 color='C0')
                    rect_y = patches.Rectangle((x_start, y_start),
                                               y_prod_cntr,
                                               imp_pME_prod_cntr)
                    rect_x = patches.Rectangle((x_start, y_start),
                                               x_prod_cntr,
                                               imp_pME_prod_cntr)
                    list_rect_y.append(rect_y)
                    list_rect_x.append(rect_x)
                    x_max = x_start+x_prod_cntr
                    y_max = y_start+imp_pME_prod_cntr
                    x_start += x_prod_cntr

                col_rect_y = mpl_col.PatchCollection(list_rect_y, facecolor='C0')
                col_rect_x = mpl_col.PatchCollection(list_rect_x, facecolor='gray')
                ax.add_collection(col_rect_x)
                ax.add_collection(col_rect_y)
                ax.autoscale()
                dict_lim[imp_cat][prod] = {}
                dict_lim[imp_cat][prod]['x'] = (0, x_max)
                dict_lim[imp_cat][prod]['y'] = (0, y_max)
                dict_ax[imp_cat][prod] = ax

        dict_imp_cat_prod_cntr_sort_new_trunc = {}
        for imp_cat in self.dict_source_shift:
            dict_imp_cat_prod_cntr_sort_new_trunc[imp_cat] = {}
            for prod in self.dict_source_shift[imp_cat]:
                imp_abs_sum = 0
                dict_cntr = {}
                for cntr in self.dict_source_shift[imp_cat][prod]:
                    imp_pME = self.dict_source_shift[imp_cat][prod][cntr]['imp_pME']
                    y_new = self.dict_source_shift[imp_cat][prod][cntr]['EU_import_new']
                    imp_abs = imp_pME*y_new
                    imp_abs_sum += imp_abs
                    dict_cntr[cntr] = imp_abs
                list_imp_cat_prod_sort = sorted(
                        dict_cntr.items(),
                        key=operator.itemgetter(1), reverse=True)
                list_imp_cat_prod_sort_trunc = []
                if imp_abs_sum > 0:
                    imp_cum = 0
                    bool_add = True
                    for tup_cntr_imp in list_imp_cat_prod_sort:
                        cntr, imp_abs = tup_cntr_imp
                        imp_rel = imp_abs/imp_abs_sum
                        imp_cum += imp_rel
                        if imp_cum <= cfg.imp_cum_lim_source_shift:
                            list_imp_cat_prod_sort_trunc.append(cntr)

                        elif bool_add:
                            list_imp_cat_prod_sort_trunc.append(cntr)
                            bool_add = False
                dict_imp_cat_prod_cntr_sort_new_trunc[imp_cat][prod] = (
                        list_imp_cat_prod_sort_trunc)

        for imp_cat in self.dict_source_shift:
            for prod in self.dict_source_shift[imp_cat]:
                plt.close('all')
                x_start = 0
                y_start = 0
                list_rect_y = []
                ax = dict_ax[imp_cat][prod]
                for cntr in self.dict_source_shift[imp_cat][prod]:
#                    x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
                    imp_pME_prod_cntr = self.dict_source_shift[imp_cat][prod][cntr]['imp_pME']
                    y_prod_cntr_new = self.dict_source_shift[imp_cat][prod][cntr]['EU_import_new']
                    x_prod_cntr = self.dict_source_shift[imp_cat][prod][cntr]['export']
                    rect_y = patches.Rectangle((x_start, y_start),
                                               y_prod_cntr_new,
                                               imp_pME_prod_cntr)
                    if cntr in dict_imp_cat_prod_cntr_sort_new_trunc[imp_cat][prod]:
                        cntr_long = dict_cntr_short_long[cntr]
                        x_text = x_start+x_prod_cntr/2
                        y_text = y_start+imp_pME_prod_cntr
                        ax.text(x_text,
                                y_text,
                                ' '+cntr_long,
                                rotation=90,
                                verticalalignment='bottom',
                                horizontalalignment='center',
                                    color='green')
                    list_rect_y.append(rect_y)
                    x_start += x_prod_cntr

                    rect_y = patches.Rectangle((0, 0), 0, 0)
                    list_rect_y.append(rect_y)

                col_rect_y = mpl_col.PatchCollection(list_rect_y,
                                                     facecolor='green')
                col_rect_y.set_alpha(0.5)
                ax.add_collection(col_rect_y)
                ax.autoscale()
                fig = ax.get_figure()
                unit = imp_cat[-1]
                ax.set_ylabel('{}/M€'.format(unit))
                ax.set_xlabel('M€')
                ax.set_xlim(dict_lim[imp_cat][prod]['x'])
                ax.set_ylim(dict_lim[imp_cat][prod]['y'])
                ax.locator_params(axis='both', nbins=4, tight=True)
                ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

                prod_short = self.dict_prod_long_short[prod]
                prod_short_lower = prod_short.lower()
                prod_short_lower_strip = prod_short_lower.strip()
                prod_short_lower_strip = prod_short_lower_strip.replace(':', '')
                prod_short_lower_strip_us = prod_short_lower_strip.replace(' ',
                                                                           '_')
                fp = self.dict_imp_cat_fp[imp_cat]
                fp_lower = fp.lower()
                fp_prod = fp_lower+'_'+prod_short_lower_strip_us
                fig.tight_layout(pad=0.1)

                shift_dir_path = cfg.result_dir_path+cfg.shift_dir_name

                pdf_file_name = fp_prod+'.pdf'
                pdf_dir_path = shift_dir_path+cfg.pdf_dir_name
                pdf_file_path = pdf_dir_path+pdf_file_name
                fig.savefig(pdf_file_path)

                png_file_name = fp_prod+'.png'
                png_dir_path = shift_dir_path+cfg.png_dir_name
                png_file_path = png_dir_path+png_file_name
                fig.savefig(png_file_path)


    def log(self):
        print('log')