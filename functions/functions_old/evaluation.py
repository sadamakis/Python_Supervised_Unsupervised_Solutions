"""
# This is a set of machine learning tools developed using Python
"""

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import statistics 
from decorators import time_function



def _capture_unit(resp, n_bands=10, top=10):
                """Capture percentage of unit."""
                width = int(round(len(resp)/float(n_bands), 0))
                resp_sum = [sum(resp[i-width:i]) for i in range(width, width*n_bands, width)]
                resp_sum.append(sum(resp[width*(n_bands-1):]))
                resp_cum = np.cumsum(resp_sum)
                count = [width]*(n_bands-1) + [len(resp) - width*(n_bands-1)]
                count_cum = np.cumsum(count)
                resp_adr = np.array(resp_sum) / float(np.sum(resp))
                resp_frac = resp_cum / float(np.sum(resp))
                cut = np.linspace(1.0/n_bands, 1.0, n_bands)
                resp_rate = np.array(resp_sum) / np.array(count)
                resp_rate_cum = resp_cum / count_cum
                cases_rate = 1.0/n_bands
                lift = resp_adr / cases_rate
                cumulative_lift = resp_frac / cut
                return pd.DataFrame(np.c_[cut, count, resp_sum, resp_cum, resp_rate, (1-resp_rate)/(resp_rate + 10e-10), \
                                                                                                                  (1-resp_rate_cum)/(resp_rate_cum+10e-10), resp_adr, resp_frac, lift, cumulative_lift], \
                                                                                                columns = ['Quantile', '# Cases', '# Responses', 'Cumulative # Responses', '% Response', 'FPR', \
                                                                                                                               'Cumulative FPR', 'ADR', 'Cumulative ADR', 'Lift', 'Cumulative Lift']).head(top)

def _capture_value(resp, n_bands=10, top=10):
                """Capture percentage of values."""
                width = int(round(len(resp)/float(n_bands), 0))
                resp_sum = [sum(resp[i-width:i]) for i in range(width, width*n_bands, width)]
                resp_sum.append(sum(resp[width*(n_bands-1):]))
                resp_cum = np.cumsum(resp_sum)
                count = [width]*(n_bands-1) + [len(resp) - width*(n_bands-1)]
                resp_adr = np.array(resp_sum) / float(np.sum(resp))
                resp_frac = resp_cum / float(np.sum(resp))
                return pd.DataFrame(np.c_[resp_sum, resp_cum, resp_adr, resp_frac], \
                                                                                                columns = ['Value', 'Cumulative Value', 'VDR', 'Cumulative VDR']).head(top)
                
# COPIED
def _expand_unit(alist, weight):
                """Expand unit by weight."""
                expanded = []
                for x in alist:
                                if x == 0:
                                                expanded += [0]*weight
                                else:
                                                expanded.append(x)
                return expanded

# COPIED
def _expand_value(alist, weight):
                """Expand value by weight."""
                expanded = []
                for it in alist:
                                if it[0] == 0:
                                                expanded += [0]*weight
                                else:
                                                expanded.append(it[1])
                return expanded
                                                
    

def lift_table(score_unit_value, n_bands=10, rows=10, weight=1):
                """Generate the lifting table.
                
                Parameters
                =============
                score_unit_value: array_like
                                Three columns: score, unit, value
                n_bands: int 
                                Number of bands to separate
                rows: int
                                Number of rows to show
                weight: int
                                Weight added to the negative samples
                                
                Returns
                =============
                out: dataframe
                                Lifting table
                """
                rank_by_score = sorted(score_unit_value, key=lambda t: t[0], reverse=True)
                units = _expand_unit([x[1] for x in rank_by_score], weight)
                values = _expand_value([x[1:3] for x in rank_by_score], weight)
                unit_caprate = _capture_unit(units, n_bands, rows)
                value_caprate = _capture_value(values, n_bands, rows)
                return pd.concat([unit_caprate, value_caprate], axis=1)

def _capture(resp, n_bands=10):
                """Detection rate"""
                width = int(round(len(resp)/float(n_bands), 0))
                resp_sum = [sum(resp[i-width:i]) for i in range(width, width*n_bands, width)]
                resp_sum.append(sum(resp[width*(n_bands-1):]))
                resp_cum = np.cumsum(resp_sum)
                resp_frac = resp_cum / float(np.sum(resp))
                return resp_frac

def compute_gini(score_unit_value, n_bands=10, weight=1):
                """Compute gini for unit and value.

                Parameters
                =============
                score_unit_value: array_like
                                Three columns: score, unit, value
                n_bands: int 
                                Number of bands to separate
                rows: int
                                Number of rows to show
                weight: int
                                Weight added to the negative samples
                                
                Returns
                =============
                out: array-like
                                Unit gini and value gini
                """

                rank_by_score = sorted(score_unit_value, key=lambda t: t[0], reverse=True)
                
                rank_by_unit = sorted(score_unit_value, key=lambda t: t[1], reverse=True)
                units = _expand_unit([x[1] for x in rank_by_score], weight)
                best_units = _expand_unit([x[1] for x in rank_by_unit], weight)
                unit_cap = _capture(units, n_bands)
                best_unit_cap = _capture(best_units, n_bands)
                
                rank_by_value = sorted(score_unit_value, key=lambda t: t[2], reverse=True)
                values = _expand_value([x[1:3] for x in rank_by_score], weight)
                best_values = _expand_value([x[1:3] for x in rank_by_value], weight)
                value_cap = _capture(values, n_bands)
                best_value_cap = _capture(best_values, n_bands)
                
                cut = np.linspace(1.0/n_bands, 1.0, n_bands)
                return sum(unit_cap - cut)/sum(best_unit_cap - cut), sum(value_cap - cut)/sum(best_value_cap - cut)
                
                
def _capt(resp, n_bands=10):
                """Detection rate"""
                width = int(round(len(resp)/float(n_bands), 0))
                resp_sum = [sum(resp[i-width:i]) for i in range(width, width*n_bands, width)]
                resp_sum.append(sum(resp[width*(n_bands-1):]))
                resp_cum = np.cumsum(resp_sum)
                resp_frac = resp_cum / float(np.sum(resp))
                return resp_frac
                
def gini(score_value, n_bands=10):
                """Compute gini for unit and value.

                Parameters
                =============
                score_unit_value: array_like
                                Three columns: score, unit, value
                n_bands: int 
                                Number of bands to separate
                rows: int
                                Number of rows to show
                weight: int
                                Weight added to the negative samples
                                
                Returns
                =============
                out: array-like
                                Unit gini and value gini
                """

                rank_by_score = sorted(score_value, key=lambda t: t[0], reverse=True)
                rank_by_value = sorted(score_value, key=lambda t: t[1], reverse=True)
                values = [t[1] for t in rank_by_score]
                best_values = [t[1] for t in rank_by_value]
                value_cap = _capt(values, n_bands)
                best_value_cap = _capt(best_values, n_bands)
                cut = np.linspace(1.0/n_bands, 1.0, n_bands)
                return sum(value_cap - cut)/sum(best_value_cap - cut)

def _capt_weight(resp, weight, n_bands=10):
                """Detection rate"""        
                arr = np.asarray(weight)
                cum_arr = arr.cumsum() / arr.sum()
                idx = np.searchsorted(cum_arr, np.linspace(0, 1, n_bands, endpoint=False)[1:])
                chunks = np.split(resp, idx)
                w_chunks = np.split(arr, idx)
                resp_sum = [(chunks[x]*w_chunks[x]).sum() for x in range(0,n_bands)]
                resp_cum = np.cumsum(resp_sum)
                resp_frac = resp_cum / resp_cum[n_bands-1]
                
                w_resp_sum = [np.sum(x) for x in w_chunks]
                w_resp_cum = np.cumsum(w_resp_sum)
                w_resp_frac = w_resp_cum / np.sum(arr)

                return resp_frac, w_resp_frac

def gini_weight(score_value, n_bands=10):
                """Compute gini for unit and value.

                Parameters
                =============
                score_value: array_like
                                Three columns: score, unit, weight
                n_bands: int 
                                Number of bands to separate
                Returns
                =============
                out: array-like
                                Unit gini and value gini
                """
                rank_by_score = sorted(score_value, key=lambda t: t[0], reverse=True)
                rank_by_value = sorted(score_value, key=lambda t: t[1], reverse=True)
                values = [t[1] for t in rank_by_score]
                w_values = [t[2] for t in rank_by_score]
                best_values = [t[1] for t in rank_by_value]
                w_best_values = [t[2] for t in rank_by_value]
                value_cap = _capt_weight(resp=values, weight=w_values, n_bands=n_bands)
                best_value_cap = _capt_weight(resp=best_values, weight=w_best_values, n_bands=n_bands)
#                cut = np.linspace(1.0/n_bands, 1.0, n_bands)
                return sum(value_cap[0] - value_cap[1])/sum(best_value_cap[0] - best_value_cap[1])

def compute_gini_weight(score_unit_value, n_bands=10):
                """Compute gini for unit and value.

                Parameters
                =============
                score_unit_value: array_like
                                Four columns: score, unit, value, weight
                n_bands: int 
                                Number of bands to separate
                                
                Returns
                =============
                out: array-like
                                Unit gini and value gini
                """

                rank_by_score = sorted(score_unit_value, key=lambda t: t[0], reverse=True)
                
                rank_by_unit = sorted(score_unit_value, key=lambda t: t[1], reverse=True)
                units = [x[1] for x in rank_by_score]
                w_units = [x[3] for x in rank_by_score]
                best_units = [x[1] for x in rank_by_unit]
                w_best_units = [x[3] for x in rank_by_unit]
                unit_cap = _capt_weight(resp=units, weight=w_units, n_bands=n_bands)
                best_unit_cap = _capt_weight(resp=best_units, weight=w_best_units, n_bands=n_bands)
                
                rank_by_value = sorted(score_unit_value, key=lambda t: t[2], reverse=True)
                values = [x[2] for x in rank_by_score]
                w_values = [x[3] for x in rank_by_score]
                best_values = [x[2] for x in rank_by_value]
                w_best_values = [x[3] for x in rank_by_value]
                value_cap = _capt_weight(resp=values, weight=w_values, n_bands=n_bands)
                best_value_cap = _capt_weight(resp=best_values, weight=w_best_values, n_bands=n_bands)
                
#                cut = np.linspace(1.0/n_bands, 1.0, n_bands)
                return sum(unit_cap[0] - unit_cap[1])/sum(best_unit_cap[0] - best_unit_cap[1]), sum(value_cap[0] - value_cap[1])/sum(best_value_cap[0] - best_value_cap[1])

def fpr_table(score_unit_value, start=3, end=8, weight=1):
                """Generate the FPR table.

                Parameters
                =============
                score_unit_value: array_like
                                Three columns: score, unit, value
                starts int 
                                The smallest FPR to consider.
                end: int
                                The largest FPR to consider.
                weight: int
                                Weight added to the negative samples
                                
                Returns
                =============
                out: dataframe
                                FPR table
                """
                A = sorted(score_unit_value, key=lambda t: t[0], reverse=True)
                tot_counts = 0.
                for it in A:
                                if it[1] == 0:
                                                tot_counts += weight
                                else:
                                                tot_counts += 1
                _, tot_units, tot_values = np.sum(A, 0)
                result = []
                for n in range(start, end+1):
                                counts, units, values = 0, 0, 0
                                for t in A:
                                                if t[1] == 1:
                                                                units += 1
                                                                counts += 1
                                                                values += t[2]
                                                else:
                                                                counts += weight
                                                if counts > 500 and (counts - units) / float(units) >= n:
                                                                result.append([n, counts/float(tot_counts), units, units/float(tot_units), values, values/float(tot_values)])
                                                                break
                return pd.DataFrame(result, columns=['FPR', 'Fraction', 'Unit', 'ADR', 'Value', 'VDR'])
                
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)

def plot_lift_table(lt, xlim=None, ylim=None):
    plt.figure()
    ax = plt.subplot((111))
    ax.plot(lt['Quantile'], lt['ADR'], "bo", label="Account DR", linestyle='solid')
    ax.plot(lt['Quantile'], lt['VDR'], "r^", label="Value DR", linestyle='dashed')
    ax.set_xlabel('Quantile')
    ax.set_ylabel('Detection Rate')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="center right", fontsize="x-small")
    plt.tight_layout()
    plt.show()
    
def plot_fpr_table(fpr, xlim=None, ylim=None):
    plt.figure()
    ax = plt.subplot((111))
    ax.plot(fpr['FPR'], fpr['ADR'], "bo", label="Account DR")
    ax.plot(fpr['FPR'], fpr['VDR'], "r^", label="Value DR")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('Detection Rate')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="center right", fontsize="x-small")
    plt.tight_layout()
    plt.show()




# COPIED
def _capture_unit_weight(resp, weight, n_bands=10, top=10):
    arr = np.asarray(weight)
    cum_arr = arr.cumsum() / arr.sum()
    idx = np.searchsorted(cum_arr, np.linspace(0, 1, n_bands, endpoint=False)[1:])
    chunks = np.split(resp, idx)
    w_chunks = np.split(arr, idx)
    resp_sum = [(chunks[x]*w_chunks[x]).sum() for x in range(0,n_bands)]
    resp_cum = np.cumsum(resp_sum)
    resp_frac = resp_cum / resp_cum[n_bands-1]

    w_resp_sum = [np.sum(x) for x in w_chunks]
    w_resp_cum = np.cumsum(w_resp_sum)
    resp_adr = resp_sum / resp_cum[n_bands-1]
    resp_rate = resp_sum / np.array(w_resp_sum)
    resp_rate_cum = resp_cum / w_resp_cum
    w_resp_frac = w_resp_cum / np.sum(arr)
    cases_rate = np.asarray(w_resp_sum) / float(np.sum(arr))
    lift = resp_adr / cases_rate
    cumulative_lift = resp_frac / w_resp_frac
    return pd.DataFrame(np.c_[w_resp_frac, w_resp_sum, resp_sum, resp_cum, resp_rate, (1-resp_rate)/(resp_rate + 10e-10), \
                    (1-resp_rate_cum)/(resp_rate_cum+10e-10), resp_adr, resp_frac, lift, cumulative_lift], \
                    columns = ['Quantile Unit', '# Cases', '# Responses', 'Cumulative # Responses', '% Response', 'FPR', \
                    'Cumulative FPR', 'ADR', 'Cumulative ADR', 'Lift', 'Cumulative Lift']).head(top)

# COPIED
def _capture_value_weight(resp, weight, n_bands=10, top=10):
    arr = np.asarray(weight)
    cum_arr = arr.cumsum() / arr.sum()
    idx = np.searchsorted(cum_arr, np.linspace(0, 1, n_bands, endpoint=False)[1:])
    chunks = np.split(resp, idx)
    w_chunks = np.split(arr, idx)
    resp_sum = [(chunks[x]*w_chunks[x]).sum() for x in range(0,n_bands)]
    resp_cum = np.cumsum(resp_sum)
    resp_frac = resp_cum / resp_cum[n_bands-1]

    w_resp_sum = [np.sum(x) for x in w_chunks]
    w_resp_cum = np.cumsum(w_resp_sum)
    resp_adr = resp_sum / resp_cum[n_bands-1]
    #resp_rate = np.array(resp_sum) / np.array(w_resp_sum)
    #resp_rate_cum = resp_cum / w_resp_cum
    w_resp_frac = w_resp_cum / np.sum(arr)
    #cases_rate = np.asarray(w_resp_sum) / float(np.sum(arr))
    #lift = resp_adr / cases_rate
    #cumulative_lift = resp_frac / w_resp_frac
    return pd.DataFrame(np.c_[w_resp_frac, resp_sum, resp_cum, resp_adr, resp_frac], \
        columns = ['Quantile Value', 'Value', 'Cumulative Value', 'VDR', 'Cumulative VDR']).head(top)

# COPIED
@time_function
def lift_table_weight(score_unit_value, n_bands=10, rows=10):
                """Generate the lifting table.
                
                Parameters
                =============
                score_unit_value: array_like
                                Four columns in this exact order: score, unit, value, weight
                n_bands: int 
                                Number of bands to separate
                rows: int
                                Number of rows to show
                                
                Returns
                =============
                out: dataframe
                                Lifting table
                """

                rank_by_score = sorted(score_unit_value, key=lambda t: t[0], reverse=True)

                units = [x[1] for x in rank_by_score]
                w_units = [x[3] for x in rank_by_score]
                unit_caprate = _capture_unit_weight(units, w_units, n_bands, rows)
                    
                values = [x[2] for x in rank_by_score]
                w_values = [x[3] for x in rank_by_score]
                value_caprate = _capture_value_weight(values, w_values, n_bands, rows)

                return pd.concat([unit_caprate, value_caprate], axis=1)

# COPIED
@time_function
def plot_ADR_Quantile(lt, xlim=None, ylim=None):
    plt.figure()
    ax = plt.subplot((111))
    ax.plot(lt['Quantile Unit'], lt['ADR'], "bo", label="Account DR", linestyle='solid')
    ax.plot(lt['Quantile Value'], lt['VDR'], "r^", label="Value DR", linestyle='dashed')
    ax.set_xlabel('Population Distribution')
    ax.set_ylabel('Detection Rate')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="center right", fontsize="x-small")
    plt.tight_layout()
    plt.show()
    
# COPIED
@time_function
def plot_cADR_Quantile(lt, xlim=None, ylim=None):
    plt.figure()
    ax = plt.subplot((111))
    ax.plot(lt['Quantile Unit'], lt['Cumulative ADR'], "bo", label="Cum. Account DR", linestyle='solid')
    ax.plot(lt['Quantile Value'], lt['Cumulative VDR'], "r^", label="Cum. Value DR", linestyle='dashed')
    ax.set_xlabel('Population Distribution')
    ax.set_ylabel('Cum. Detection Rate')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="center right", fontsize="x-small")
    plt.tight_layout()
    plt.show()

# COPIED
@time_function
def plot_FPR_Quantile(lt, xlim=None, ylim=None):
    plt.figure()
    ax = plt.subplot((111))
    ax.plot(lt['Quantile Unit'], lt['FPR'], "bo", label="FPR", linestyle='solid')
    ax.set_xlabel('Population Distribution')
    ax.set_ylabel('False Positive Rate')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="center right", fontsize="x-small")
    plt.tight_layout()
    plt.show()

# COPIED
@time_function
def plot_cFPR_Quantile(lt, xlim=None, ylim=None):
    plt.figure()
    ax = plt.subplot((111))
    ax.plot(lt['Quantile Unit'], lt['Cumulative FPR'], "bo", label="Cum. FPR", linestyle='solid')
    ax.set_xlabel('Population Distribution')
    ax.set_ylabel('Cum. False Positive Rate')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="center right", fontsize="x-small")
    plt.tight_layout()
    plt.show()

# COPIED
@time_function
def plot_ROC_curve(
    table_name, # Table name
    target_variable, # Target variable name
    predicted_variable, # Predicted variable name
    weight_variable # Weight variable name
    ): 
    
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(table_name[target_variable].values))]
    # calculate scores
    ns_auc = roc_auc_score(table_name[target_variable].values, ns_probs, sample_weight=table_name[weight_variable].values)
    model_auc = roc_auc_score(table_name[target_variable].values, table_name[predicted_variable].values, sample_weight=table_name[weight_variable].values)
    # summarize scores
    print('Random: ROC AUC=%.3f' % (ns_auc))
    print('Model: ROC AUC=%.3f' % (model_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(table_name[target_variable].values, ns_probs, sample_weight=table_name[weight_variable].values)
    model_fpr, model_tpr, _ = roc_curve(table_name[target_variable].values, table_name[predicted_variable].values, sample_weight=table_name[weight_variable].values)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Random')
    pyplot.plot(model_fpr, model_tpr, marker='.', label='Model')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
# COPIED
@time_function
def plot_precision_recall_curve(
    table_name, # Table name that has the target variable, the predicted variable, and the weights
    target_variable, # Target variable name
    predicted_variable, # Predicted variable name
    weight_variable # Weight variable name
    ):
    
    # predict class values
    model_precision, model_recall, _ = precision_recall_curve(table_name[target_variable].values, table_name[predicted_variable].values, sample_weight=table_name[weight_variable].values)
    # plot the precision-recall curves
    no_skill = table_name[weight_variable].values[table_name[target_variable].values==1].sum() / table_name[weight_variable].values.sum()
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    pyplot.plot(model_recall, model_precision, marker='.', label='Model')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


# COPIED
@time_function
def plot_cutoffs(
    table_name, # Table name that has the target variable, the predicted variable, and the weights
    target_variable, # Target variable name
    predicted_variable, # Predicted variable name
    weight_variable, # Weight variable name
    n_bands, # Number of bands between 0 and 1
    return_table=False # Set to True in order to return the table that produced the graph, otherwise set to False
    ):
 
    threshold_array = np.linspace(0,1,n_bands+1, endpoint=True)
    column_names = ['cutoff', 'f1', 'accuracy', 'sensitivity/recall', 'specificity', 'precision']
    df = pd.DataFrame(columns = column_names)

    for threshold in threshold_array: 
        yhat = (table_name[predicted_variable].values >= threshold)*1
        cm = confusion_matrix(table_name[target_variable].values, yhat, sample_weight=table_name[weight_variable].values)

        true_positive = cm[1,1]
        false_positive = cm[0,1]
        true_negative = cm[0,0]
        false_negative = cm[1,0]
        positive = false_negative + true_positive
        negative = true_negative + false_positive
        #####from confusion matrix calculate metrics
        model_f1 = (2*true_positive)/(2*true_positive+false_positive+false_negative)
        model_accuracy = (true_positive+true_negative)/(positive+negative)
        model_sensitivity = true_positive/positive
        model_specificity = true_negative/negative
        model_precision = true_positive/(true_positive+false_positive)
        # Run different code depending on the Python version
        if sys.version_info[0] < 2:
            df = df.append({'cutoff':threshold, 'f1':model_f1, 'accuracy':model_accuracy, 'sensitivity/recall':model_sensitivity, 'specificity':model_specificity, 'precision':model_precision}, ignore_index=True)
        else: 
            df = pd.concat([df, pd.DataFrame({'cutoff':[threshold], 'f1':[model_f1], 'accuracy':[model_accuracy], 'sensitivity/recall':[model_sensitivity], 'specificity':[model_specificity], 'precision':[model_precision]})], ignore_index=True)
        
    # create overplot
    pyplot.plot(df['cutoff'], df['f1'], marker='.', label='F1 score')
    pyplot.plot(df['cutoff'], df['accuracy'], linestyle='--', label='Accuracy')
    pyplot.plot(df['cutoff'], df['sensitivity/recall'], marker='.', linestyle='--', label='Sensitivity/Recall')
    pyplot.plot(df['cutoff'], df['specificity'], linestyle='dotted', label='Specificity')
    pyplot.plot(df['cutoff'], df['precision'], marker='.', linestyle='dashdot', label='Precision')
    # axis labels
    pyplot.xlabel('Cutoff')
    pyplot.ylabel('Metrics')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    if return_table == True:
        return(df)

@time_function
def calculate_gini(
    table_name, # Table name
    target_variable, # Target variable name
    predicted_variable, # Predicted variable name
    weight_variable,  # Weight variable name
    select_top_percent=100 # Percent between 0 and 100: Gini is calculated for the top select_top_percent% of the observations, sorted by score
    ): 

    ts = table_name.sort_values(by=predicted_variable, ascending=False)
    arr = ts[weight_variable]
    cum_arr = arr.cumsum() / arr.sum()
    idx = np.searchsorted(cum_arr, select_top_percent/100)
    ts = ts[0:idx]

    model_auc = roc_auc_score(ts[target_variable].values, ts[predicted_variable].values, sample_weight=ts[weight_variable].values)

    return 2*model_auc-1


def auc_precision_recall(
    table_name, # Table name that has the target variable, the predicted variable, and the weights
    target_variable, # Target variable name
    predicted_variable, # Predicted variable name
    weight_variable # Weight variable name
    ):
    
    # predict class values
    model_precision, model_recall, _ = precision_recall_curve(table_name[target_variable].values, table_name[predicted_variable].values, sample_weight=table_name[weight_variable].values)
    # calculate precision-recall AUC
    model_pr_auc = auc(model_recall, model_precision)
    return(model_pr_auc)


def plot_cross_validation_score(model # Name of cross-validation model
                               ):

    score_list = []
    for i in [col for col in model.cv_results_.keys() if col.startswith('split')]:
        score_list.append(-model.cv_results_[i][model.best_index_])
    no_lists = [*range(1,len(score_list)+1)]

    plt.bar(no_lists, score_list)
    # horizontal line indicating the threshold
    mean_score = -model.cv_results_['mean_test_score'][model.best_index_]
    std_score = -model.cv_results_['std_test_score'][model.best_index_]
    plt.plot([no_lists[0],no_lists[-1]], [mean_score, mean_score], "k--")
    plt.xlabel('Training instances', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.title('Cross-validation scores for classifier', fontsize=15)
    plt.legend(['Mean score = {}'.format(mean_score.round(decimals=5))], fancybox=True, loc='best', fontsize=10)
    plt.annotate('St.D. score = {}'.format(std_score.round(decimals=5)), xy=(1,max(score_list)), size=10)
    plt.show()
