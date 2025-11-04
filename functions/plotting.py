# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:25:00 2025

@author: DevHov
"""


def plot_gw(ax, df, xcol, ycol, ylabel, xlabel='Date', namcol='name'):
    for _, row in df.iterrows():
        ax.plot(row[xcol], row[ycol], label=row[namcol])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax
