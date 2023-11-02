# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:49:26 2023

@author: WilliamTBradley
"""

'''
This script intends to automate comparison of NODE results via Mann-Whitney U Test.
    Compares N_Bsteps for various data scenarios (i.e., Figure 9 in paper)
'''

# H0: Deriv MSE of NODEs w/larger intervals is same or larger than derivs
    # of shorter intervals

import numpy as np
import pandas as pd
import sys
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

Reg_size = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

# =============================================================================
# # User-defined analysis params
# =============================================================================
# mod = 'fhn' #Either 'fhn','sir', or 'vdp'.  Or redefined below.
comp_type = 'greater' #Type of MWUT analysis to perform. Either 'greater' or 'less'
log_transform = False #Boxplots are easier to visualize if True

# =============================================================================
# # Set up and run Mann-Whitney U Test
# =============================================================================
columns = ['Model','Runs','Noise Lvl','Data pts/interval','p-value']
df_res = pd.DataFrame(columns=columns)

for mod in ['fhn']:#,'sir','fhn']:
    file = 'data/df_hyper+{}+N_Bsteps.csv'.format(mod)
    df_sol = pd.read_csv(file)
    for N_runs in [5,10]:
        for noise in [0.1]:
            df_abr = df_sol[df_sol['Runs'] == N_runs]
            df_abr = df_abr[df_abr['Noise Lvl'] == noise]
            df_abr = df_abr[df_abr['stop criteria'] == 'converge']
            # log-transform data
            if log_transform == True:
                df_abr['MSE_dy pred fitdat'] = np.log10(df_abr['MSE_dy pred fitdat'])
            sample1 = df_abr[df_abr['Data pts/interval'] == 4]['MSE_dy pred fitdat'].to_numpy()
            for N_dps in [7,9,10]:
                sample2 = df_abr[df_abr['Data pts/interval'] == N_dps]['MSE_dy pred fitdat'].to_numpy()      
                # t_stat, p_value = ttest_ind(sample1, sample2,alternative='greater')
                U1, p_value = mannwhitneyu(sample1,sample2,alternative=comp_type)
                df_res.loc[len(df_res.index)] = [mod, N_runs, noise, N_dps, p_value] 
            dum = df_abr.loc[df_abr['Data pts/interval'] == 4,'MSE_dy pred fitdat'].values
            dum2 = df_abr.loc[df_abr['Data pts/interval'] == 7,'MSE_dy pred fitdat'].values
            dum3 = df_abr.loc[df_abr['Data pts/interval'] == 9,'MSE_dy pred fitdat'].values
            dum4 = df_abr.loc[df_abr['Data pts/interval'] == 10,'MSE_dy pred fitdat'].values
            plt.figure()
            plt.boxplot([dum,dum2,dum3,dum4],labels=[4,7,9,10])
            # plt.boxplot([dum2,dum3,dum4],labels=[7,9,10])
            plt.title(mod) # Add # dps, noise lvl, # runs?
            plt.xlabel('# of dps in integration interval')
            plt.ylabel('Mean Squared Error')
            # plt.yscale('log')
# plt.savefig('visuals/Boxplot',dpi=600)
# Print result of model comparisons
print(df_res)
sys.exit()
