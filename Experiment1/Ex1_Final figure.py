# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:39:42 2023

This code reads the results saved for each of the 10 datasets
 (results are extracted using Ex1_Case1_Main.py and Ex1_Case2_Main.py for each dataset) 
 and plots the figure of experiment 1

@author: sgazar
"""

import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib


import matplotlib.pyplot as plt
plt.close('all')

#%% Defs
def load_data_from_excel(path, sheets): 
    file_list = os.listdir(path)
    excel_files = [f for f in file_list if f.endswith('.xlsx')]
    data = {}
    for excel_file in excel_files:
        file_path = os.path.join(path, excel_file)
        file_data = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}
        data[excel_file] = file_data
    return data


def process_data(data, sheets):
    processed_data = [pd.DataFrame() for _ in sheets]
    for key, file_data in data.items():
        for i, sheet in enumerate(sheets):
            new_df = pd.DataFrame({f"{sheet}_{key}": file_data[sheet].iloc[:, 1]})
            processed_data[i] = pd.concat([processed_data[i], new_df], axis=1)
    return processed_data

def calculate_statistics(df):
    metrics = ['DSC', 'HD95', 'HD', 'RAVD', 'ASSD']
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1)
    se = std / np.sqrt(df.shape[1])
    degrees_freedom = df.shape[1] - 1
    conf_level = 0.95
    conf_interval = [stats.t.interval(conf_level, degrees_freedom, loc=m, scale=s) for m, s in zip(mean, se)]
    statistics_df = pd.DataFrame({'Metric': metrics, 'Mean': mean, 'Std': std, 'SE': se,
                                  'Low_CI': [ci[0] for ci in conf_interval], 'High_CI': [ci[1] for ci in conf_interval]})
    return statistics_df.set_index('Metric')

#%% load results
sheets = ['Training', 'Validation', 'Test_a', 'Test_b']
path1 = 'D:/First Project/Codes/Experiments/Ex1_New/Data_All/Results/Case1'
path2 = 'D:/First Project/Codes/Experiments/Ex1_New/Data_All/Results/Case2'

#%%
case1_data = load_data_from_excel(path1, sheets)
case2_data = load_data_from_excel(path2, sheets)

Tr1_data, Val1_data, TsA1_data, TsB1_data = process_data(case1_data, sheets)
Tr2_data, Val2_data, TsA2_data, TsB2_data = process_data(case2_data, sheets)

# Calculating statistics
stat_Tr1, stat_Val1, stat_TsA1, stat_TsB1 = map(calculate_statistics, [Tr1_data, Val1_data, TsA1_data, TsB1_data])
stat_Tr2, stat_Val2, stat_TsA2, stat_TsB2 = map(calculate_statistics, [Tr2_data, Val2_data, TsA2_data, TsB2_data])



#%% save results

save_dir = 'D:/First Project/Codes/Experiments/Ex1_New/Data_All/Results/result_table.xlsx'

def format_metrics(df):
    """Format the metrics as 'mean (SE)', rounded to two decimal places."""
    return df.apply(lambda row: f"{row['Mean']:.2f} ({row['SE']:.2f})", axis=1)


formatted_TsA1 = format_metrics(stat_TsA1)
formatted_TsB1 = format_metrics(stat_TsB1)
formatted_TsA2 = format_metrics(stat_TsA2)
formatted_TsB2 = format_metrics(stat_TsB2)


final_df = pd.DataFrame({
    'Metric': stat_TsA1.index,
    'Case 1 - TsA': formatted_TsA1.values,
    'Case 1 - TsB': formatted_TsB1.values,
    'Case 2 - TsA': formatted_TsA2.values,
    'Case 2 - TsB': formatted_TsB2.values
}).set_index('Metric').T  


#%% plots

save_directory = 'D:/First Project/Codes/Experiments/Ex1_New/Data_All/Results'


def extract_metric_data_for_CI(stat_data, metric):
    metric_data = stat_data.loc[metric]
    mean = metric_data['Mean']
    ci_low = metric_data['Low_CI']
    ci_high = metric_data['High_CI']
    return mean, ci_low, ci_high


sheets = ['Training', 'Validation', 'Test', 'Ex-test']
metrics = ['DSC', 'HD95', 'HD', 'RAVD', 'ASSD']



linewidth = 3
capsize = 10
markersize = 10
matplotlib.rcParams['font.family'] = 'Georgia' 

stat_data_cases = {
    'Case1': [stat_Tr1, stat_Val1, stat_TsA1, stat_TsB1],
    'Case2': [stat_Tr2, stat_Val2, stat_TsA2, stat_TsB2]
}

for metric in metrics:
    plt.figure(figsize=(15, 8))
    plt.gca().set_facecolor('#F0F8FF')
    
    for case, stat_data_list in stat_data_cases.items():
        means = []
        conf_intervals = []
        for stat_data in stat_data_list:
            mean, ci_low, ci_high = extract_metric_data_for_CI(stat_data, metric)
            means.append(mean)
            conf_intervals.append([mean - ci_low, ci_high - mean])
        
        x_pos = np.arange(len(sheets))
        offset = -0.15 if case == 'Case1' else 0.15
        
        color = 'blue' if case == 'Case1' else 'red'
        
        plt.errorbar(x_pos + offset, means, yerr=np.transpose(conf_intervals), fmt='o',
                     color=color, ecolor=color, elinewidth=linewidth, capsize=capsize, markersize=markersize, label=case)
    
    if metric == 'DSC':
       plt.ylim(0.6, 1)
    elif metric == 'HD95':
       plt.ylim(0, 10)
    elif metric == 'HD':
       plt.ylim(0, 10)
    elif metric == 'RAVD':
       plt.ylim(-0.6, 0.6)
    elif metric == 'ASSD':
       plt.ylim(0, 3.5)
    
    plt.xticks(x_pos, sheets, fontsize=46)
    plt.yticks(fontsize=34)
    plt.ylabel(f'{metric} score', fontsize=54 ) 

    plt.grid(visible=True, which='major', axis='y', color='white', linestyle='-', linewidth=1.5)
    plt.legend(fontsize=44)
    plt.tight_layout()

    plt.show()
