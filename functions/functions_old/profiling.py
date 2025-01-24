from matplotlib import pyplot as plt
import pandas as pd
import os 
import numpy as np
from decorators import time_function 
import weighted as wghtd

# Alternative way to calculate the average by group 
#def weighted_mean_group(
#    df,
#    data_col,
#    weight_col,
#    by_col
#    ):
#    df['_data_times_weight'] = df[data_col]*df[weight_col]
#    df['_weight_where_notnull'] = df[weight_col]*pd.notnull(df[data_col])
#    g = df.groupby(by_col)
#    result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
#    del df['_data_times_weight'], df['_weight_where_notnull']
#    return result

def weighted_mean_group(
    df,
    data_col,
    weight_col,
    by_col
    ):
    
    gr = df.groupby(by_col)
    return gr.apply(lambda x: np.average(x[data_col].dropna(), weights=x[~x[data_col].isnull()][weight_col]))
    
def weighted_median_group(
    df,
    data_col,
    weight_col,
    by_col
    ):
    
    gr = df.groupby(by_col)
    return gr.apply(lambda x: wghtd.median(x[data_col].dropna(), x[~x[data_col].isnull()][weight_col]))
    
def CountFrequency(df, my_list, weight, normalize=False):
    # Creating an empty dictionary
    freq = {}
    for item, wght in zip(df[my_list], df[weight]):
        if (item in freq):
            freq[item] += wght
        else:
            freq[item] = wght
    freq = pd.Series(freq)
    if normalize == True:
        freq = freq / sum(df[weight])
    return freq    
    
def weighted_frequency_group(
    df,
    data_col,
    weight_col,
    by_col, 
    normalize=False
    ):
    
    gr = df.groupby(by_col)
    return gr.apply(lambda x: CountFrequency(x, data_col, weight_col, normalize=normalize))

@time_function
def numeric_summary_statistics(
    table_name, 
    variable_list, 
    cluster_variable_name, 
    weight_variable_name,
    data_path
    ):
    
    # Ensure that the graph folder exists
    if not os.path.isdir('{0}/output/graphs'.format(data_path)):
        os.makedirs('{0}/output/graphs'.format(data_path))
        
    df_stats = pd.DataFrame()
    for var in variable_list:
    
        # Ensure that the graph folder exists
        if not os.path.isdir('{0}/output/graphs/{1}'.format(data_path, var)):
            os.makedirs('{0}/output/graphs/{1}'.format(data_path, var))
            
        # mean graphs
        col_hist_mean = weighted_mean_group(table_name,var,weight_variable_name,cluster_variable_name)
        Y1_overlay = col_hist_mean.tolist()
        X1_overlay = col_hist_mean.index.tolist()
        Y2_overlay = [np.average(table_name[var].dropna(), weights=table_name[~table_name[var].isnull()][weight_variable_name])]*len(table_name[cluster_variable_name].value_counts())
        
        plt.xlabel("Cluster labels")
        plt.ylabel(var)
        plt.bar(X1_overlay, Y1_overlay, color='maroon', width=0.4)
        plt.plot(X1_overlay, Y2_overlay, label='Overall average')
        plt.legend()
        plt.savefig('{0}/output/graphs/{1}/{1}_mean.png'.format(data_path, var))
        plt.show()
        
        # median graphs
        col_hist_med = weighted_median_group(table_name,var,weight_variable_name,cluster_variable_name)
        Y1_overlay = col_hist_med.tolist()
        X1_overlay = col_hist_med.index.tolist()
        Y2_overlay = [wghtd.median(table_name[var].dropna(), table_name[~table_name[var].isnull()][weight_variable_name])]*len(table_name[cluster_variable_name].value_counts())
        
        plt.xlabel("Cluster labels")
        plt.ylabel(var)
        plt.bar(X1_overlay, Y1_overlay, color='blue', width=0.4)
        plt.plot(X1_overlay, Y2_overlay, label='Overall median')
        plt.legend()
        plt.savefig('{0}/output/graphs/{1}/{1}_median.png'.format(data_path, var))
        plt.show()
        
        # Create summary statistics table
        #df_stats_temp = pd.DataFrame(weighted_mean_group(table_name, var, weight_variable_name, cluster_variable_name)).rename(columns={0: var}).T
        df_stats_temp = pd.DataFrame(col_hist_mean).rename(columns={0: var}).T
        df_stats_temp['Baseline'] = [np.average(table_name[var].dropna(), weights=table_name[~table_name[var].isnull()][weight_variable_name])]
        for i in np.unique(table_name['cluster_labels']):
            df_stats_temp["{0}{1}".format(i, '_baseline_diff')] = (df_stats_temp[i] - df_stats_temp['Baseline']) / df_stats_temp['Baseline']
            df_stats_temp["{0}{1}".format(i, '_baseline_diff')] = ["{:.2%}".format(i) for i in df_stats_temp["{0}{1}".format(i, '_baseline_diff')]]
        df_stats_temp = round(df_stats_temp, 2)
        df_stats = pd.concat([df_stats, df_stats_temp], ignore_index=False)
        
    df_stats = df_stats.reset_index().rename(columns={'index': "Attribute"})
    df_stats.to_csv('{}/output/summary_statistics_numeric.csv'.format(data_path), index=False)
    display(df_stats)
            
    
@time_function
def character_summary_statistics(
    table_name, 
    variable_list, 
    cluster_variable_name, 
    weight_variable_name,
    data_path
    ):
    
    # Ensure that the graph folder exists
    if not os.path.isdir('{0}/output/graphs'.format(data_path)):
        os.makedirs('{0}/output/graphs'.format(data_path))
        
    df_stats = pd.DataFrame()
    for var in variable_list:
        t1 = pd.DataFrame(weighted_frequency_group(table_name, var, weight_variable_name, cluster_variable_name, normalize=True).sort_index()).T.stack().reset_index().rename(columns={'level_1': 'labels'})
        t2 = pd.DataFrame(CountFrequency(table_name, var, weight_variable_name, normalize=True).sort_index()).reset_index().rename(columns={'index': 'labels', 0: 'Baseline'})
        t12 = t1.join(t2, rsuffix='_1').drop('labels_1', axis=1).rename(columns={'level_0': "Attribute"})
        t12['Attribute'] = var
        
        plot = t12.plot(kind='bar', legend=True, xlabel=var, ylabel='Percentage')
        plot.figure.savefig('{0}/output/graphs/{1}_cat_hist.png'.format(data_path, var))
        
        for i in np.unique(table_name['cluster_labels']):
            t12["{0}{1}".format(i, '_baseline_diff')] = (t12[i] - t12['Baseline']) / t12['Baseline']
            t12["{0}{1}".format(i, '_baseline_diff')] = ["{:.2%}".format(j) for j in t12["{0}{1}".format(i, '_baseline_diff')]]
            t12[i] = ["{:.2%}".format(i) for i in t12[i]]
        t12["Baseline"] = ["{:.2%}".format(i) for i in t12['Baseline']]
        t12 = round(t12, 2)
#        df_stats = df_stats.append(t12)
        df_stats = pd.concat([df_stats, t12], ignore_index=False)
        
    df_stats.to_csv('{}/output/summary_statistics_character.csv'.format(data_path), index=False)
    display(df_stats)
