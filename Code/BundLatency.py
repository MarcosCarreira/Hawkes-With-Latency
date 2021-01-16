#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:32:18 2021

@author: marcoscscarreira
"""

# %% Imports


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import datetime as dtt

# from functools import partial

import xlwings as xw

# from numba import jit, njit
# from numba import types
# from numba.typed import Dict

from tick.dataset import fetch_hawkes_bund_data

import hawklat as hl

print('numpy ' + np.__version__)
print('pandas ' + pd.__version__)
# print('matplotlib ' + mpl.__version__)
print('xlwings ' + xw.__version__)

# %% Link to Tables workbook


wb = xw.Book('Tables.xlsx')
sht = wb.sheets['Global']
sht2 = wb.sheets['Powell']
sht3 = wb.sheets['LBFGSB']
sht4 = wb.sheets['Bounds']


# %% Import Tick data

timestamps_list = fetch_hawkes_bund_data()

# %% Check data size


# print([len(timestamps_list), len(timestamps_list[0]),
#        len(timestamps_list[0][0])])

# print(np.mean([
#     len(timestamps_list[j][0]) for j in range(20)]))

# print(np.mean([
#     len(timestamps_list[j][1]) for j in range(20)]))

# print(np.mean([
#     len(timestamps_list[j][2]) for j in range(20)]))

# print(np.mean([
#     len(timestamps_list[j][3]) for j in range(20)]))


# %% Count of daily sizes


def sizests(tslist):
    return np.array([[len(tslist[j][i]) for i in range(4)]
                for j in range(20)])

def cutts(x, t1, t2):
    cut1 = np.extract(x >= t1, x)
    return np.extract(cut1 <= t2, cut1) - t1


T1 = 2 * 3600
T2 = 10 * 3600
subts = [[cutts(timestamps_list[j][i], T1, T2) for i in range(4)]
                for j in range(20)]

print(sizests(timestamps_list))

print(sizests(subts))

print(sizests(subts) / sizests(timestamps_list))

ts_list = subts.copy()

# %% Stats on time intervals within series ans series size


print(np.mean([
    len(ts_list[j][0]) for j in range(20)]))

print(np.mean([
    len(ts_list[j][1]) for j in range(20)]))

print(np.mean([
    len(ts_list[j][2]) for j in range(20)]))

print(np.mean([
    len(ts_list[j][3]) for j in range(20)]))

PERCENTILES = [0.10, 0.15, 0.25, 0.50, 0.75, 0.90]

print(pd.Series(np.concatenate([
    np.diff(ts_list[j][0]) for j in range(20)]))
    .describe(percentiles=PERCENTILES))

print(pd.Series(np.concatenate([
    np.diff(ts_list[j][1]) for j in range(20)]))
    .describe(percentiles=PERCENTILES))

print(pd.Series(np.concatenate([
    np.diff(ts_list[j][2]) for j in range(20)]))
    .describe(percentiles=PERCENTILES))

print(pd.Series(np.concatenate([
    np.diff(ts_list[j][3]) for j in range(20)]))
    .describe(percentiles=PERCENTILES))

# %% Define constants

M = 4
N = 20

BOUNDS_BLOCK =\
    [(5e-3, 5e-1)] * 4 +\
    [(1e+2, 2e+4)] * 3 + [(1e+1, 1e+3)] +\
    [(5e+0, 1e+2)] + [(1e+2, 1e+3)] + [(1e-2, 1e+2)] * 2 +\
    [(1e+3, 5e+3)] * 2 + [(5e+2, 2e+3)] + [(1e-1, 1e+3)]

X0_BLOCK = np.array(
    [2.5e-2] * (2) + [1.0e-1] * (2) +
    [5e+2] * (3) + [1e+1] * (1) +
    [2e+1] * (1) + [2e+2] * (1) + [5e+1] + [5e-1] +
    [3e+3] * (2) + [1e+3] * (1) + [1e+1] * (1))

sht4.range('D2').value = np.array([X0_BLOCK]).T

NEW_BOUNDS_BLOCK =\
    [(1e-2, 1e-1)] * 2 + [(5e-2, 5e-1)] * 2 +\
    [(1e+2, 1e+3)] * 2 +[(1e+2, 7e+2)] + [(5e+0, 5e+1)] +\
    [(5e+0, 1e+2)] + [(1e+2, 1e+3)] + [(1e+0, 2e+2)] + [(1e-2, 1e+1)] +\
    [(1e+3, 5e+3)] * 2 + [(8e+2, 2e+3)] + [(1e+0, 5e+2)]

sht4.range('B2').value = NEW_BOUNDS_BLOCK

LAT_BUND = 250e-6

MRNGS = [np.array([0, 1]), np.array([2, 3])]

# %% Functions to process results


def show_results(result):
    λ0s = np.array([result[0][j][0] for j in range(M)])
    αs = np.array([result[0][j][1:M + 1] for j in range(M)])
    βs = np.array([result[0][j][M + 1:] for j in range(M)])
    print([λ0s, αs, βs, result[1]])


def align_results(result):
    λ0s = np.array([result[0][j][0] for j in range(M)])
    αs = np.array([result[0][j][1:M + 1] for j in range(M)]).flatten()
    βs = np.array([result[0][j][M + 1:] for j in range(M)]).flatten()
    return np.concatenate((λ0s, αs, βs, result[1]))


def export_results(result, cell):
    sht.range(cell).value = align_results(result)


def export_results_2(result, cell):
    sht2.range(cell).value = align_results(result)


def export_results_3(result, cell):
    sht3.range(cell).value = align_results(result)

# %% Loop


for j in range(N):
    ts = ts_list[j]
    print('Day ' + str(j + 1))
    print(dtt.datetime.now())
    block_Powell = hl.findθlatblock(ts, 4, mrngs=MRNGS,
                                    bounds=NEW_BOUNDS_BLOCK, x0=X0_BLOCK,
                                    τ=LAT_BUND, method='Powell',
                                    print_m=True, disp=True)
    print('Day ' + str(j + 1) + ' - Powell')
    print(dtt.datetime.now())
    show_results(block_Powell)
    export_results_2(block_Powell, 'B' + str(j + 2))
    block_LBFGSB = hl.findθlatblock(ts, 4, mrngs=MRNGS,
                                    bounds=NEW_BOUNDS_BLOCK, x0=X0_BLOCK,
                                    τ=LAT_BUND, method='L-BFGS-B',
                                    print_m=True, disp=True)
    print('Day ' + str(j + 1) + ' - L-BFGS-B')
    print(dtt.datetime.now())
    show_results(block_LBFGSB)
    export_results_3(block_LBFGSB, 'B' + str(j + 2))
    block_de = hl.findθlatblock_de(ts, 4, mrngs=MRNGS,
                                   bounds=NEW_BOUNDS_BLOCK, τ=LAT_BUND,
                                   maxiter=1000, popsize=25,
                                   atol=0, tol=5e-2,
                                   print_m=True, disp=True)
    print('Day ' + str(j + 1) + ' - Global')
    print(dtt.datetime.now())
    show_results(block_de)
    export_results(block_de, 'B' + str(j + 2))
    print('Exported')

# %% Import results


all_results = pd.read_excel('Tables.xlsx', sheet_name='Powell',
                            header=0, index_col=0)

# %% Plot results - λ0


all_results[['λ0Pu', 'λ0Pd', 'λ0Ta', 'λ0Tb']].\
    plot(figsize=(12, 9), title='λ0');

# %% Plot results - αPP and αTP


all_results[['α11', 'α12', 'α13', 'α14']].\
    plot(figsize=(12, 9), title='αPP and αTP')

# %% Plot results - βPP and βTP


all_results[['β11', 'β13']].\
    plot(figsize=(12, 9), title='β11 (P->P) and β33 (T->P)')

# %% Plot results - αPT and βPT


all_results[['α31', 'α32', 'β31']].\
    plot(figsize=(12, 9), title='αPT and βPT')

# %% Plot results - αTT and βTT


all_results[['α33', 'α34', 'β33']].\
    plot(figsize=(12, 9), title='αTT and βTT')

# %% Mean values


print(all_results[['λ0Pu', 'λ0Pd', 'λ0Ta', 'λ0Tb']].describe())
print(all_results[['α11', 'α12', 'α13', 'α14']].describe())
print(all_results[['α31', 'α32', 'α33', 'α34']].describe())
print(all_results[['β11', 'β13', 'β31', 'β33']].describe())

# %% Ratios

ratios = pd.DataFrame()
ratios['PuPu'] = all_results['α11'] / all_results['β11']
ratios['TaPu'] = all_results['α13'] / all_results['β13']
ratios['PdPu'] = all_results['α12'] / all_results['β11']
ratios['TbPu'] = all_results['α14'] / all_results['β13']
ratios['PuTa'] = all_results['α31'] / all_results['β31']
ratios['TaTa'] = all_results['α33'] / all_results['β33']
ratios['PdTa'] = all_results['α32'] / all_results['β31']
ratios['TbTa'] = all_results['α34'] / all_results['β33']

# %% Mean ratios


print(ratios[['PuPu', 'TaPu', 'PdPu', 'TbPu']].describe())
print(ratios[['PuTa', 'TaTa', 'PdTa', 'TbTa']].describe())

# %% Plot ratios - P


ratios[['PuPu', 'TaPu', 'PdPu', 'TbPu']].\
    plot(figsize=(12, 9), title='α/β for P')

# %% Plot ratios - T


ratios[['PuTa', 'TaTa', 'PdTa', 'TbTa']].\
    plot(figsize=(12, 9), title='α/β for T')