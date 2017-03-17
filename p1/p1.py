#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Shaojiewei"


import pandas

from ggplot import *

def  barplot_compare(path_file):
	bar_plot = pandas.read_csv(path_file)
	gg = ggplot(data = bar_plot, aes(x = 'factor(Congruent)')) + geom_bar()
	return gg



path = 'C:/Users/admin/Desktop/p1/stroopdata.csv'
barplot_compare(path)

