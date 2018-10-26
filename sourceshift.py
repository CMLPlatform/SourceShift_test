# -*- coding: utf-8 -*-
""" Main script for paper on
    reducing import embedded footprints of EU28 by source shifting

@author: B.F. de Boer
@institution: Leiden University CML
"""
from collections import OrderedDict
import math
import matplotlib.collections as mpl_col
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd

import cfg
import data_io as dio
import utils as ut


def set_priority():
    """ Find highest contributing product for each footprint



    """

    # For each EU28 country, calculate import embodied footprints
    dict_imp_cntr = {}
    for cntr_fd in list_reg_fd:
        df_tY_cntr = dict_eb['tY'][cntr_fd].copy()
        df_tY_cntr.loc[cntr_fd] = 0
        df_tY_cntr_fdsum = df_tY_cntr.sum(axis=1)
        dict_df_imp_cntr = dio.get_dict_df_imp(dict_cf, dict_eb,
                                               df_tY_cntr_fdsum)
        dict_imp_cntr[cntr_fd] = dio.get_dict_imp(dict_df_imp_cntr)

    # For each footprint, select highest contributing products
    dict_imp_prod_sort = dio.get_dict_imp_prod_sort(dict_df_imp,
                                                    cfg.imp_cum_lim_priority)

    dict_imp_sel = {}
    for cntr_fd in list_reg_fd:
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
                if imp_cat_eff == imp_cat_sel:
                    for prod in dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff]:
                        if prod not in (
                                dict_imp_sel_reg[imp_cat_sel]):
                            dict_imp_sel_reg[imp_cat_sel][prod] = 0
                        for cntr in dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod]:
                            imp_abs = dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod][cntr]
                            if cntr_fd is not cntr:
                                dict_imp_sel_reg[imp_cat_sel][prod] += imp_abs

##    dict_imp_sel = {}
##    for cntr_fd in list_reg_fd:
##        dict_imp_sel[cntr_fd] = {}
##        for imp_cat_sel in dict_imp_prod_sort:
##            dict_imp_sel[cntr_fd][imp_cat_sel] = {}
##            for imp_cat_eff in dict_imp:
##                dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff] = OrderedDict()
##                for prod in dict_imp_prod_sort[imp_cat_sel]:
##                    dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod] = (
##                            dict_imp_cntr[cntr_fd][imp_cat_eff][prod])
#
#    dict_imp_sel_reg = {}
##    for cntr_fd in dict_imp_sel:
#    for cntr_fd in dict_imp_cntr:
#        for imp_cat in dict_imp_cntr[cntr_fd]:
#            dict_imp_sel_reg[imp_cat] = {}
##            for prod in dict_imp_cntr[cntr_fd][imp_cat]:
##                if prod in dict_imp_prod_sort[imp_cat]:
#            for prod in dict_imp_prod_sort[imp_cat]:
#                if prod not in dict_imp_sel_reg[imp_cat]:
#                    dict_imp_sel_reg[imp_cat][prod] = 0
#                for cntr in dict_imp_cntr[cntr_fd][imp_cat][prod]:
#                    dict_imp_sel_reg[imp_cat][prod] += (
#                            dict_imp_cntr[cntr_fd][imp_cat][prod][cntr])
##            if imp_cat_sel not in dict_imp_sel_reg:
##                dict_imp_sel_reg[imp_cat_sel] = {}
##            for imp_cat_eff in dict_imp_sel[cntr_fd][imp_cat_sel]:
##                if imp_cat_eff == imp_cat_sel:
##                    for prod in dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff]:
##                        if prod not in (
##                                dict_imp_sel_reg[imp_cat_sel]):
##                            dict_imp_sel_reg[imp_cat_sel][prod] = 0
##                        for cntr in dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod]:
##                            imp_abs = dict_imp_sel[cntr_fd][imp_cat_sel][imp_cat_eff][prod][cntr]
##                            if cntr_fd is not cntr:
##                                dict_imp_sel_reg[imp_cat_sel][prod] += imp_abs


    dict_xlim = {}
    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_sel_reg):
        fig_y_size = len(dict_imp_sel_reg[imp_cat_sel])
        plt.close('all')
        fig = plt.figure(figsize=ut.cm2inch((16, 1+fig_y_size*0.4)))
        plot_id = imp_cat_sel_id+1
        plot_loc = 140+plot_id
        ax = fig.add_subplot(plot_loc)
        df = pd.DataFrame(dict_imp_sel_reg[imp_cat_sel],
                          index=['import'])
        df.rename(columns=dict_prod_long_short, inplace=True)
        df.T.plot.barh(stacked=True,
                       ax=ax,
                       legend=False,
                       color='C0')
        xlim = ax.get_xlim()
        xlim_max_magn = 10**np.floor(np.log10(xlim[1]))
        xlim_max_ceil = math.ceil(xlim[1]/xlim_max_magn)*xlim_max_magn
        tup_xlim_max_ceil = (int(xlim[0]), xlim_max_ceil)
        if imp_cat_sel not in dict_xlim:
            dict_xlim[imp_cat_sel] = tup_xlim_max_ceil
        if xlim[1] > dict_xlim[imp_cat_sel][1]:
            dict_xlim[imp_cat_sel] = tup_xlim_max_ceil


    analysis_name = 'priority_setting'
    dict_prod_order = {}
    for imp_cat_sel in dict_imp_sel_reg:
        df = pd.DataFrame(dict_imp_sel_reg[imp_cat_sel],
                          index=['import'])
        df.rename(columns=dict_prod_long_short, inplace=True)
        df_sum_sort = df.sum().sort_values()
        prod_order = list(df_sum_sort.index)
        dict_prod_order[imp_cat_sel] = prod_order

    plt.close('all')
    fig_y_size_max = 0
    dict_imp_cat_unit = dio.get_dict_imp_cat_unit()
    for imp_cat_sel in dict_imp_sel_reg:
        fig_y_size = len(dict_imp_sel_reg[imp_cat_sel])
        if fig_y_size > fig_y_size_max:
            fig_y_size_max = fig_y_size
    fig = plt.figure(figsize=ut.cm2inch((16, 8)))
    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_sel_reg):
        fig_y_size = len(dict_imp_sel_reg[imp_cat_sel])
        prod_order = dict_prod_order[imp_cat_sel]
        plot_id = imp_cat_sel_id+1
        plot_loc = 220+plot_id
        ax = fig.add_subplot(plot_loc)
        fp = dict_imp_cat_fp[imp_cat_sel]
        unit = dict_imp_cat_unit[imp_cat_sel[-1]]
        ax.set_xlabel('{} [{}]'.format(fp, unit))
        df = pd.DataFrame(dict_imp_sel_reg[imp_cat_sel],
                          index=['import'])
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
        ax.set_yticklabels(yticklabels)

        plt.locator_params(axis='x', nbins=1)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.set_xlim(dict_xlim[imp_cat_sel])
        xtick_magnitude = dict_imp_cat_magnitude[imp_cat_sel]

        list_xtick = [i/xtick_magnitude for i in dict_xlim[imp_cat_sel]]
        list_xtick[0] = int(list_xtick[0])
        ax.set_xticks(list(dict_xlim[imp_cat_sel]))
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


def shift_source():
    print('sourceshift')
    dict_cntr_short_long = dio.get_dict_cntr_short_long()

#    x_prod_cntr_min = 0.5

    dict_df_imp_pME = {}
    dict_df_imp_pME['e'] = dict_cf['e'].dot(dict_eb['cRe']).dot(dict_eb['cL'])
    dict_df_imp_pME['m'] = dict_cf['m'].dot(dict_eb['cRm']).dot(dict_eb['cL'])
    dict_df_imp_pME['r'] = dict_cf['r'].dot(dict_eb['cRr']).dot(dict_eb['cL'])
    dict_imp_pME = dio.get_dict_imp(dict_df_imp_pME)

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

    dict_imp_cat_prod_imp_abs_sum = {}
    dict_imp_cat_prod_cntr_sort_trunc = {}

    for imp_cat in dict_imp_prod_cntr_sort:
        dict_imp_cat_prod_imp_abs_sum[imp_cat] = {}
        dict_imp_cat_prod_cntr_sort_trunc[imp_cat] = {}
        for prod in dict_imp_prod_cntr_sort[imp_cat]:
            imp_abs_sum = 0
            for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                imp_abs = dict_imp_prod_cntr_sort[imp_cat][prod][cntr]
                imp_abs_sum += imp_abs
            dict_imp_cat_prod_imp_abs_sum[imp_cat][prod] = imp_abs_sum
            list_imp_cat_prod_sort = sorted(
                    dict_imp_prod_cntr_sort[imp_cat][prod].items(),
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
                    if imp_cum <= cfg.imp_cum_lim_source_shift:
                        list_imp_cat_prod_sort_trunc.append(cntr)
                    elif bool_add:
                        list_imp_cat_prod_sort_trunc.append(cntr)
                        bool_add = False
            dict_imp_cat_prod_cntr_sort_trunc[imp_cat][prod] = (
                    list_imp_cat_prod_sort_trunc)

    dict_lim = {}
    dict_ax = {}
    for imp_cat in dict_imp_prod_cntr_sort:
        dict_lim[imp_cat] = {}
        dict_ax[imp_cat] = {}
        for prod in dict_imp_prod_cntr_sort[imp_cat]:
            plt.close('all')
            x_start = 0
            y_start = 0
            list_rect_y = []
            list_rect_x = []
            fig = plt.figure(figsize=ut.cm2inch((16, 8)))
            ax = plt.gca()
            for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                imp_pME_prod_cntr = dict_imp_pME[imp_cat][prod][cntr]
                y_prod_cntr = dict_tY_eu28_cntr_import[prod][cntr]
                x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
                if x_prod_cntr >= cfg.x_prod_cntr_min:
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

    dict_tY_prod = {}
    for prod in dict_tY_eu28_cntr_import:
        dict_tY_prod[prod] = 0
        for cntr in dict_tY_eu28_cntr_import[prod]:
            dict_tY_prod[prod] += dict_tY_eu28_cntr_import[prod][cntr]

    dict_imp_pME_new = {}
    dict_y_new = {}
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
            dict_imp_pME_new[imp_cat][prod] = {}
            dict_y_new[imp_cat][prod] = {}
            for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                imp_pME_prod_cntr = dict_imp_pME[imp_cat][prod][cntr]
                y_prod_cntr = dict_tY_eu28_cntr_import[prod][cntr]
                x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
                if x_prod_cntr >= cfg.x_prod_cntr_min:
                    if x_prod_cntr < y_prod:
                        rect_y = patches.Rectangle((x_start, y_start),
                                                   x_prod_cntr,
                                                   imp_pME_prod_cntr)
                        y_prod -= x_prod_cntr
                        dict_y_new[imp_cat][prod][cntr] = x_prod_cntr
                        dict_imp_pME_new[imp_cat][prod][cntr] = (
                                imp_pME_prod_cntr)
                    elif y_prod > 0:
                        rect_y = patches.Rectangle((x_start, y_start),
                                                   y_prod,
                                                   imp_pME_prod_cntr)
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

            col_rect_y = mpl_col.PatchCollection(list_rect_y,
                                                 facecolor='green')
            col_rect_x = mpl_col.PatchCollection(list_rect_x, facecolor='gray')
            col_rect_y.set_alpha(cfg.reduc_alpha)

    dict_imp_cat_prod_imp_abs_new = {}
    dict_imp_cat_prod_imp_abs_new_sum = {}
    for imp_cat in dict_imp_pME_new:
        dict_imp_cat_prod_imp_abs_new[imp_cat] = {}
        dict_imp_cat_prod_imp_abs_new_sum[imp_cat] = {}
        for prod in dict_imp_pME_new[imp_cat]:
            dict_imp_cat_prod_imp_abs_new[imp_cat][prod] = {}
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
            list_imp_cat_prod_sort = sorted(
                    dict_imp_cat_prod_imp_abs_new[imp_cat][prod].items(),
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
                    if imp_cum <= cfg.imp_cum_lim_source_shift:
                        list_imp_cat_prod_sort_trunc.append(cntr)

                    elif bool_add:
                        list_imp_cat_prod_sort_trunc.append(cntr)
                        bool_add = False
            dict_imp_cat_prod_cntr_sort_new_trunc[imp_cat][prod] = (
                    list_imp_cat_prod_sort_trunc)

    dict_tY_prod = {}
    for prod in dict_tY_eu28_cntr_import:
        dict_tY_prod[prod] = 0
        for cntr in dict_tY_eu28_cntr_import[prod]:
            dict_tY_prod[prod] += dict_tY_eu28_cntr_import[prod][cntr]

    for imp_cat in dict_imp_prod_cntr_sort:
        for prod in dict_imp_prod_cntr_sort[imp_cat]:
            plt.close('all')
            x_start = 0
            y_start = 0
            list_rect_y = []
            ax = dict_ax[imp_cat][prod]
            y_prod = dict_tY_prod[prod]
            for cntr in dict_imp_prod_cntr_sort[imp_cat][prod]:
                imp_pME_prod_cntr = dict_imp_pME[imp_cat][prod][cntr]
                y_prod_cntr = dict_tY_eu28_cntr_import[prod][cntr]
                x_prod_cntr = dict_tY_world_ex_prod_cntr[prod][cntr]
                if x_prod_cntr >= cfg.x_prod_cntr_min:
                    if x_prod_cntr < y_prod:
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
                        rect_y = patches.Rectangle((x_start, y_start),
                                                   x_prod_cntr,
                                                   imp_pME_prod_cntr)
                        y_prod -= x_prod_cntr
                    elif y_prod > 0:
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
                        rect_y = patches.Rectangle((x_start, y_start),
                                                   y_prod,
                                                   imp_pME_prod_cntr)
                        y_prod -= y_prod

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

            prod_short = dict_prod_long_short[prod]
            prod_short_lower = prod_short.lower()
            prod_short_lower_strip = prod_short_lower.strip()
            prod_short_lower_strip = prod_short_lower_strip.replace(':', '')
            prod_short_lower_strip_us = prod_short_lower_strip.replace(' ',
                                                                       '_')
            fp = dict_imp_cat_fp[imp_cat]
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
    return dict_y_new


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
    dict_imp_new_reg = {}
    for imp_cat_id, imp_cat_sel in enumerate(dict_df_tY_import_new):
        #    Fill import column
        df_tY_eu28 = dict_eb['tY'][list_reg_fd].copy()
        df_tY_eu28_fdsum = df_tY_eu28.sum(axis=1)
        df_tY_eu28_import = df_tY_eu28_fdsum.copy()
        df_tY_eu28_import[:] = 0
        df_tY_eu28_import[list(dict_df_tY_import_new[imp_cat_sel].keys())] = (
                list(dict_df_tY_import_new[imp_cat_sel].values()))
        df_tY_eu28_import.columns = ['import']
        dict_df_imp_new = dio.get_dict_df_imp(dict_cf, dict_eb, df_tY_eu28_import)
        dict_imp_new = dio.get_dict_imp(dict_df_imp_new)
        dict_imp_new_reg[imp_cat_sel] = {}
        for imp_cat_eff in dict_imp_new:
            if imp_cat_eff not in dict_imp_new_reg[imp_cat_sel]:
                dict_imp_new_reg[imp_cat_sel][imp_cat_eff] = {}
            for prod in dict_imp_new[imp_cat_eff]:
                if prod not in dict_imp_new_reg[imp_cat_sel][imp_cat_eff]:
                    dict_imp_new_reg[imp_cat_sel][imp_cat_eff][prod] = 0
                for cntr in dict_imp_new[imp_cat_eff][prod]:
                    dict_imp_new_reg[imp_cat_sel][imp_cat_eff][prod] += (
                            dict_imp_new[imp_cat_eff][prod][cntr])
    return dict_imp_new_reg


def calc_improv():
    dict_imp_new_reg = calc_new_fp()
    dict_imp_cat_unit = dio.get_dict_imp_cat_unit()
    list_prod_order_cons = dio.get_list_prod_order_cons()
    plt.close('all')
    fig = plt.figure(figsize=ut.cm2inch((16, 1+13*0.4)))
    dict_imp_prod_sort_full = dio.get_dict_imp_prod_sort(dict_df_imp, cfg.
                                                         imp_cum_lim_full)

    dict_xlim_improv = {}
    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_reg):
        for imp_cat_eff_id, imp_cat_eff in (
                enumerate(dict_imp_new_reg[imp_cat_sel])):
            plot_id = imp_cat_eff_id+1
            plot_loc = 140+plot_id
            ax = fig.add_subplot(plot_loc)
            df_old = pd.DataFrame(dict_imp_prod_sort_full[imp_cat_eff],
                                  index=['import'])
            df_new = pd.DataFrame(dict_imp_new_reg[imp_cat_sel][imp_cat_eff],
                                  index=['import'])
            df = df_new-df_old
            df.rename(columns=dict_prod_long_short, inplace=True)
            df = df.reindex(list_prod_order_cons, axis=1)
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
                xlim_new = tuple([dict_xlim_improv[imp_cat_eff][0],
                                  xlim_max_ceil])
                dict_xlim_improv[imp_cat_eff] = xlim_new
    plt.close('all')

    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_reg):
        fig = plt.figure(figsize=ut.cm2inch((16, len(list_prod_order_cons)*.4+2)))
        for imp_cat_eff_id, imp_cat_eff in (
                enumerate(dict_imp_new_reg[imp_cat_sel])):
            plot_id = imp_cat_eff_id+1
            plot_loc = 140+plot_id
            ax = fig.add_subplot(plot_loc)
            fp = dict_imp_cat_fp[imp_cat_eff]
            ax.set_title(fp)
            unit = dict_imp_cat_unit[imp_cat_eff[-1]]
            ax.set_xlabel(unit)
            ax.set_xlim(dict_xlim_improv[imp_cat_eff])
            df_old = pd.DataFrame(dict_imp_prod_sort_full[imp_cat_eff],
                                  index=['import'])
            df_new = pd.DataFrame(
                    dict_imp_new_reg[imp_cat_sel][imp_cat_eff],
                    index=['import'])
            df = df_new-df_old
            df.rename(columns=dict_prod_long_short, inplace=True)
            df = df.reindex(list_prod_order_cons, axis=1)
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
                ax.set_yticklabels(yticklabels)

            plt.locator_params(axis='x', nbins=4)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            xtick_magnitude = dict_imp_cat_magnitude[imp_cat_eff]
            list_xtick = (
                    [i/xtick_magnitude for i in dict_xlim_improv[imp_cat_eff]])
            ax.set_xticks(list(dict_xlim_improv[imp_cat_eff]))
            ax.set_xticklabels(list_xtick)

            xtick_objects = ax.xaxis.get_major_ticks()
            xtick_objects[0].label1.set_horizontalalignment('left')
            xtick_objects[-1].label1.set_horizontalalignment('right')

        fig.tight_layout(pad=0)
        plt.subplots_adjust(wspace=0.1)
        fp = dict_imp_cat_fp[imp_cat_sel]
        fp_lower = fp.lower()

        reduc_dir_path = cfg.result_dir_path+cfg.reduc_dir_name
        fig_file_name = fp_lower+'.pdf'
        pdf_dir_path = reduc_dir_path+cfg.pdf_dir_name
        fig_file_path = pdf_dir_path+fig_file_name
        fig.savefig(fig_file_path)

        fig_file_name = fp_lower+'.png'
        png_dir_path = reduc_dir_path+cfg.png_dir_name
        fig_file_path = png_dir_path+fig_file_name
        fig.savefig(fig_file_path)


def calc_improv_agg():
    dict_imp_new_reg = calc_new_fp()

    print('plot_improv_agg')
    dict_imp_prod_sort_full = dio.get_dict_imp_prod_sort(dict_df_imp,
                                                         cfg.imp_cum_lim_full)

    dict_pot_imp_agg = {}
    for imp_cat_sel_id, imp_cat_sel in enumerate(dict_imp_new_reg):
        fp_sel = dict_imp_cat_fp[imp_cat_sel]
        for imp_cat_eff_id, imp_cat_eff in enumerate(
                dict_imp_new_reg[imp_cat_sel]):

            fp_eff = dict_imp_cat_fp[imp_cat_eff]
            if fp_eff not in dict_pot_imp_agg:
                dict_pot_imp_agg[fp_eff] = {}
            if 'Ante' not in dict_pot_imp_agg[fp_eff]:
                df_old = pd.DataFrame(dict_imp_prod_sort_full[imp_cat_eff],
                                      index=['import'])
                df_old_sum = df_old.sum(axis=1)
                dict_pot_imp_agg[fp_eff]['Prior'] = float(
                        df_old_sum['import'])
            df_new = pd.DataFrame(dict_imp_new_reg[imp_cat_sel][imp_cat_eff],
                                  index=['import'])
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
    fig = plt.figure(figsize=ut.cm2inch((16, 16)))

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
        ax.yaxis.set_ticklabels(['50%', '100%'])
        ax.set_rlabel_position(0)
        xtick_count = np.arange(2*math.pi/8,
                                2*math.pi+2*math.pi/8,
                                2*math.pi/4)
        ax.set_xticks(xtick_count)
        ax.set_xticklabels(list_xticklabel)
        ax.set_ylim([0, 1.0])
        y_val = list_imp_rel
        y_val.append(y_val[0])
        x_val = list(xtick_count)
        x_val.append(x_val[0])
        ax.plot(x_val, y_val, color='C2')
        ax_title = 'Optimized {} footprint'.format(fp_sel.lower())
        ax.set_title(ax_title)
    plt.tight_layout(pad=3)
    plot_name = 'spider_plot'
    reduc_agg_dir_path = cfg.result_dir_path+cfg.reduc_agg_dir_name
    pdf_file_name = plot_name+'.pdf'
    pdf_dir_path = reduc_agg_dir_path+cfg.pdf_dir_name
    pdf_file_path = pdf_dir_path+pdf_file_name
    fig.savefig(pdf_file_path)
    png_file_name = plot_name+'.png'
    png_dir_path = reduc_agg_dir_path+cfg.png_dir_name
    png_file_path = png_dir_path+png_file_name
    fig.savefig(png_file_path)

dict_imp_cat_magnitude = dio.get_dict_imp_cat_magnitude()
dict_imp_cat_fp = dio.get_dict_imp_cat_fp()
dict_prod_long_short = dio.get_dict_prod_long_short()

dict_eb = dio.get_dict_eb()

dict_cf = dio.get_dict_cf(dict_eb)
list_reg_fd = dio.get_list_reg_fd('EU28')
df_tY_eu28 = dict_eb['tY'][list_reg_fd].copy()

for cntr in list_reg_fd:
    df_tY_eu28.loc[cntr, cntr] = 0
df_tY_eu28_fdsum = df_tY_eu28.sum(axis=1)
dict_df_imp = dio.get_dict_df_imp(dict_cf, dict_eb, df_tY_eu28_fdsum)
dict_imp = dio.get_dict_imp(dict_df_imp)

dio.make_result_dir()
set_priority()
dict_y_new = shift_source()
calc_improv()
calc_improv_agg()
