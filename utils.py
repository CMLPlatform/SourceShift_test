# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:18:15 2018

@author: bfdeboer
"""

def cm2inch(tup_cm):
    inch = 2.54
    tup_inch = tuple(i/inch for i in tup_cm)
    return tup_inch
