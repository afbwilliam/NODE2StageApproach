# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:52:38 2023

@author: WilliamTBradley
"""

# Purpose of this code is to select best SINDy model (i.e., Figs 11 and 12 in paper)
# 1) Tabulate which models are best
# 2) Graph how AICc selects the best model

import pandas as pd
import sys
import matplotlib.pyplot as plt

Reg_size = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)        # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)   # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize

# =============================================================================
# # User-defined analysis params
# =============================================================================
DDmod = 'node' # Either 'node' or 'spline'
# =============================================================================
# # Analyze data to select best model
# =============================================================================
if DDmod == 'node':
    file = 'newcasestudies_csv_results/sindy_node.csv'
    file = 'newcasestudies_csv_results/sindy_nodepoly.csv'
    file = 'newcasestudies_csv_results/sindy_nodepoly-const.csv'
    df_spl = pd.read_csv(file)
if DDmod == 'spline':
    file = 'newcasestudies_csv_results/sindy_spline.csv'
    file = 'newcasestudies_csv_results/sindy_splinepoly.csv'
    file = 'newcasestudies_csv_results/sindy_splinepoly-const.csv'
    df_spl = pd.read_csv(file)

df_spl_res = pd.DataFrame(columns=df_spl.columns)
for noise in [0,5,10]:
    for N_run in [1,3,5,10]:
        for N_dat in [3,5,10]:
            df_temp = df_spl[(df_spl['Runs']==N_run) & 
                                (df_spl['Dps/Run'] == N_dat) &
                                (df_spl['noise']==noise)]
            df_temp = df_temp[df_temp.BIC == df_temp.BIC.min()]
            df_temp = df_temp.iloc[0]
            
            df_spl_res.loc[len(df_spl_res.index)] = df_temp
            
# Print results
print(df_spl_res)            
            # sys.exit()
dum = [str(numeric_string) for numeric_string in df_spl_res['False Pos']]
dum2 = [str(numeric_string) for numeric_string in df_spl_res['False Neg']]
dum3 = [dum[string] + '/' + dum2[string] for string in range(len(dum))]
df_spl_res['FP/FN'] = dum3
# =============================================================================
# # Plot results
# =============================================================================
for noise in [5]:
    for N_run in [5]:
        for N_dat in [10]:
# for noise in [0,5]:
#     for N_run in [3,5,10]:
#         for N_dat in [5,10]:
            # noise, N_dat, N_run
            # noise = 5; N_dat = 5; N_run = 10
            # noise = 5; N_dat = 5; N_run = 10
            df_temp = df_spl[(df_spl['Runs']==N_run) & (df_spl['Dps/Run'] == N_dat) & 
                                (df_spl['noise']==noise)]
            # Remove lambda = 0.05 rows
            df_temp = df_temp[df_temp['lambda'] !=0.05]
            
            #
            thresholds = df_temp['lambda']
            CV_errs = df_temp.CV
            # AICc = df_temp.AICc
            # BIC = df_temp.BIC
            # numFalsepos_avgoversplits = df_temp['False Pos']
            # numFalseneg_avgoversplits = df_temp['False Neg']
            
            # Plot Errors vs lambdas
            # fig = plt.figure()
            
            # plt.figure(1)
            # plt.scatter(thresholds, CV_errs)
            # plt.xlabel('lambda thresholds')
            # plt.ylabel('CV error')
            # plt.show()
            
            
            
            # Box plots
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            # for data in [0,10]:
            
            data = 0
            thresholds = df_temp[df_temp['N_init'] == data]['lambda']
            AICc = df_temp[df_temp['N_init']==0].AICc
            BIC = df_temp[df_temp['N_init']==0].BIC
            
            ax1.plot(thresholds, AICc, color='tab:red', label='AICc score', linestyle='--')
            
            # ax1.plot(thresholds, AICc, color='tab:red', label='AICc', linestyle='--')
            # ax1.plot(thresholds, BIC, color='tab:green', label='BIC')
            
            # width
            numFalsepos_avgoversplits = df_temp[df_temp['N_init'] == 0]['False Pos']
            numFalseneg_avgoversplits = df_temp[df_temp['N_init'] == 0]['False Neg']
            
            ax2 = ax1.twinx()
            ax2.bar(thresholds, numFalsepos_avgoversplits, width=-0.02, align='edge', label='False Positives')
            ax2.bar(thresholds, numFalseneg_avgoversplits, width=0.02, align='edge', label='False Negatives')
            
            lines, labels1 = ax1.get_legend_handles_labels()
            bars, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + bars, labels1 + labels2,loc='upper right')
            
            ax1.set_xlabel('Lambda value')
            ax1.set_ylabel('Information criteria')
            ax2.set_ylabel('Number of terms')
            # Subplot #2
            ax3 = fig.add_subplot(212)
            
            AICc = df_temp[df_temp['N_init']==10].AICc
            BIC = df_temp[df_temp['N_init']==10].BIC
            
            ax3.plot(thresholds, AICc, color='tab:red', label='AICc score', linestyle='--')

            
            numFalsepos_avgoversplits = df_temp[df_temp['N_init'] == 10]['False Pos']
            numFalseneg_avgoversplits = df_temp[df_temp['N_init'] == 10]['False Neg']
            
            ax4 = ax3.twinx()
            ax4.bar(thresholds, numFalsepos_avgoversplits, width=-0.02, align='edge', label='False Positives')
            ax4.bar(thresholds, numFalseneg_avgoversplits, width=0.02, align='edge', label='False Negatives')
            
            lines, labels1 = ax3.get_legend_handles_labels()
            bars, labels2 = ax4.get_legend_handles_labels()
            # ax4.legend(lines + bars, labels1 + labels2,loc='upper right')
            
            ax3.set_xlabel('Lambda value')
            ax3.set_ylabel('Information criteria ')
            ax4.set_ylabel('Number of terms')
            
            plt.gcf().subplots_adjust(left=0.15)
            # if DDmod == 'node'
            # plt.savefig('RangeofLambdas', dpi=600)
            if noise == 5 and N_run == 5 and N_dat == 10:
                if DDmod == 'spline':
                    # For 5% noise, 5 runs, 10 dps/run case, model should be:
                    # (x0)' = 0.682 x1
                    # (x1)' = 0.226  + -0.509 x0 + -0.132 x1x0**2 + -0.146 x0x1**2
                    ax1.text(.0, -1.6, 'Chosen model:', va='top', ha='left', 
                             fontsize = 14, transform = ax1.transAxes)
                    ax1.text(.05, -1.8, '$\dot x_0 = 0.682 x_1$', va='top', ha='left', 
                             fontsize = 14, transform = ax1.transAxes) 
                    ax1.text(.05, -2.0, '$\dot x_1 = 0.226  -0.509 x_0 - 0.132 x_1x_0^2 - 0.146 x_0x_1^2$', va='top', ha='left', 
                             fontsize = 14, transform = ax1.transAxes) 
                if DDmod == 'node':
                    # For 5% noise, 5 runs, 10 dps/run case, model should be:
                    # (x0)' = 1.010 x1
                    # (x1)' = -0.930 x0 + 0.836 x1 + -1.152 x1x0**2
                    ax1.text(.0, -1.6, 'Chosen model:', va='top', ha='left', 
                             fontsize = 14, transform = ax1.transAxes)
                    ax1.text(.05, -1.8, '$\dot x_0 = 1.010 x_1$', va='top', ha='left', 
                             fontsize = 14, transform = ax1.transAxes) 
                    ax1.text(.05, -2.0, '$\dot x_1 = -0.930 x_0 + 0.836 x_1 -1.152 x_1x_0^2$', va='top', ha='left', 
                             fontsize = 14, transform = ax1.transAxes) 
            plt.show()
            # print(noise,N_run,N_dat)



