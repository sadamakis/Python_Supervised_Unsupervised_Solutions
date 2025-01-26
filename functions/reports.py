#import time
import pandas as pd
import numpy as np
from IPython.display import display
from decorators import time_function 
import data_transformation as dtran
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report, log_loss, precision_recall_curve
from matplotlib import pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
import os 
import weighted as wghtd


import useful_functions as ufun


@time_function
def dq_report(
    input_data, 
    data_path, 
    variables, 
    weight_variable = None, 
    dq_report_file = 'data_quality_report.csv'
    ):
            
    data_table = input_data[variables]
    
    # Get missing value rates
    if weight_variable == 'None':
        weight_variable = None
    
    if weight_variable == None:
        missing_list = data_table.apply(lambda x: (sum(x.isnull())/data_table.shape[0]) * 100, axis=0).sort_values(ascending=False)
    else: 
        missing_list = data_table.apply(lambda x: (sum(data_table[x.isnull()][weight_variable])/sum(data_table[weight_variable])) * 100, axis=0).sort_values(ascending=False)

    missing_val_df = pd.DataFrame(missing_list).round(2)
    missing_val_df.columns = ['Missing Value Percentage']
    numeric_vars_100_missing = list(missing_val_df[missing_val_df['Missing Value Percentage']==100].reset_index()['index'])

    # Identify numeric variables and get their min and max values
    numeric_vars = ufun.identify_numeric_variables(data_table)
    
    min_df = data_table[numeric_vars].min(axis=0).to_frame(name = 'Min').round(2)
    max_df = data_table[numeric_vars].max(axis=0).to_frame(name = 'Max').round(2)
    
    # Identify character variables
    character_vars = ufun.identify_character_variables(data_table)
    
    # Identify numeric variables that do not have 100% missing values 
    numeric_vars_not_missing = [val for val in numeric_vars if val not in numeric_vars_100_missing]
    
    # Get weighted average of numeric variables 
    mean_imputer = dtran.impute_missing(numeric_vars_not_missing, imputation_strategy = 'mean')
    mean_imputer.imputation_fit_weight(data_table, weight_variable)
    mean_df = pd.DataFrame(mean_imputer.impute_missing, index=['Mean']).T.round(2)

    # Get median of numeric variables 
    median_imputer = dtran.impute_missing(numeric_vars_not_missing, imputation_strategy = 'median')
    median_imputer.imputation_fit_weight(data_table, weight_variable)
    median_df = pd.DataFrame(median_imputer.impute_missing, index=['Median']).T.round(2)
    
    # Get number of unique values per feature (excluding missing values)
    unique_vals = dict()
    for v in data_table.columns:
        unique_vals[v] = len(data_table[v].value_counts())
    unique_vals_df = pd.DataFrame(unique_vals, index=['Unique Values']).T
    
    # Join all of the stats together to create one data quality report dataframe
    data_quality_df = missing_val_df.join(min_df, how='outer')\
                                        .join(max_df, how='outer')\
                                        .join(mean_df, how='outer')\
                                        .join(median_df, how='outer')\
                                        .join(unique_vals_df, how='outer')

    # Filter variables in data quality report 
    data_quality_df = data_quality_df.loc[variables]
    data_quality_df.rename_axis('Variable Name', inplace=True)
    data_quality_df = data_quality_df.reset_index()
    
    # Save data quality report
    data_quality_df = data_quality_df.sort_values(by=['Missing Value Percentage', 'Variable Name'], ascending=[False, True])
    data_quality_df['Missing Value Percentage'] = data_quality_df['Missing Value Percentage'].apply(lambda x: f"{x:.2f}%")
    data_quality_df.to_csv('{0}/output/{1}'.format(data_path, dq_report_file))
    display(data_quality_df)

#############################################################################################################################################
#############################################################################################################################################

class binary_regression_report():
    
    def __init__(
        self, 
        predictions_dictionary, 
        target_variable, 
        predicted_score_numeric, 
        amount_variable_name, 
        weight_variable_name, 
        sample_values_dict, 
        n_bands, 
        rows, 
        data_path        
    ):

        """Generate the lift table.
        
        Parameters
        =============
        predictions_dictionary: dictionary with all data samples
                        Four columns: unit/target, weight, score/predicted probability, value/amount
        target_variable: string 
                        Name of the target variable        
        predicted_score_numeric: string 
                        Name of the predicted probability column
        amount_variable_name: string 
                        Name of the amount / monetary loss column
        weight_variable_name: string 
                        Name of the sample weight variable
        sample_values_dict: dictionary
                        Sample values
        n_bands: int 
                        Number of bands to separate
        rows: int
                        Number of rows to show
        data_path: string
                        Data path to save the lift tables
        Returns
        =============
        out: dataframe
                        Lifting table
        """

        self.predictions_dictionary = predictions_dictionary
        self.target_variable = target_variable
        self.predicted_score_numeric = predicted_score_numeric
        self.amount_variable_name = amount_variable_name
        self.weight_variable_name = weight_variable_name
        self.sample_values_dict = sample_values_dict
        self.n_bands = n_bands
        self.rows = rows
        self.data_path = data_path

    @time_function
    def get_evaluation(
        self, 
        predicted_score_binary
        ):
        
        if self.weight_variable_name == 'None':
            self.weight_variable_name = None

        dataset = []
        TP = []
        FP = []
        TN = []
        FN = []
        auc = []
        logloss = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        for i, j in self.sample_values_dict.items():
            dataset.append(i)
            
            df = self.predictions_dictionary['data_{}'.format(i)]
            Y = df[self.target_variable]
            if not self.weight_variable_name:
                weight_var = None
            else: 
                weight_var = df[self.weight_variable_name]
            y_hat = df[self.predicted_score_numeric]
            y_0 = df[predicted_score_binary]
            
            cm = confusion_matrix(Y, y_0, sample_weight=weight_var)
            auc.append(roc_auc_score(Y, y_hat, sample_weight=weight_var))
            logloss.append(log_loss(y_true=Y, y_pred=y_hat, sample_weight=weight_var, normalize=False))
            accuracies.append(accuracy_score(Y, y_0, sample_weight=weight_var))
            precisions.append(precision_score(Y, y_0, sample_weight=weight_var))
            recalls.append(recall_score(Y, y_0, sample_weight=weight_var))
            f1_scores.append(f1_score(Y, y_0, sample_weight=weight_var))
            TP.append(cm[0][0])
            FP.append(cm[0][1])
            TN.append(cm[1][1])
            FN.append(cm[1][0])
        eval_df = pd.DataFrame(dataset)
        eval_df['AUC'] = [round(elem, 4) for elem in auc]
        eval_df['Log Loss'] = [round(elem, 4) for elem in logloss]
        eval_df['Accuracy'] = [round(elem, 4) for elem in accuracies]
        eval_df['Precision'] = [round(elem, 4) for elem in precisions]
        eval_df['Recall'] = [round(elem, 4) for elem in recalls]
        eval_df['F1 score'] = [round(elem, 4) for elem in f1_scores]
        eval_df['True Positives'] = [round(elem, 0) for elem in TP]
        eval_df['True Negatives'] = [round(elem, 0) for elem in TN]
        eval_df['False Positives'] = [round(elem, 0) for elem in FP]
        eval_df['False Negatives'] = [round(elem, 0) for elem in FN]
        eval_df.rename(columns = {0: 'dataset'}, inplace = True)
        eval_df.to_csv(self.data_path + '/output/evaluation_metrics.csv', index=False)
        display(eval_df)


    def _capture_unit_weight(
        self, 
        resp, 
        weight):
            
        arr = np.asarray(weight)
        cum_arr = arr.cumsum() / arr.sum()
        idx = np.searchsorted(cum_arr, np.linspace(0, 1, self.n_bands, endpoint=False)[1:])
        chunks = np.split(resp, idx)
        w_chunks = np.split(arr, idx)
        resp_sum = [(chunks[x]*w_chunks[x]).sum() for x in range(0,self.n_bands)]
        resp_cum = np.cumsum(resp_sum)
        resp_frac = resp_cum / resp_cum[self.n_bands-1]

        w_resp_sum = [np.sum(x) for x in w_chunks]
        w_resp_cum = np.cumsum(w_resp_sum)
        resp_adr = resp_sum / resp_cum[self.n_bands-1]
        resp_rate = resp_sum / np.array(w_resp_sum)
        resp_rate_cum = resp_cum / w_resp_cum
        w_resp_frac = w_resp_cum / np.sum(arr)
        cases_rate = np.asarray(w_resp_sum) / float(np.sum(arr))
        lift = resp_adr / cases_rate
        cumulative_lift = resp_frac / w_resp_frac
        return pd.DataFrame(np.c_[w_resp_frac, w_resp_sum, resp_sum, resp_cum, resp_rate, (1-resp_rate)/(resp_rate + 10e-10), \
                        (1-resp_rate_cum)/(resp_rate_cum+10e-10), resp_adr, resp_frac, lift, cumulative_lift], \
                        columns = ['Quantile Unit', '# Cases', '# Responses', 'Cumulative # Responses', '% Response', 'FPR', \
                        'Cumulative FPR', 'ADR', 'Cumulative ADR', 'Lift', 'Cumulative Lift']).head(self.rows)
                        
    def _capture_value_weight(
        self, 
        resp, 
        weight
        ):
            
        arr = np.asarray(weight)
        cum_arr = arr.cumsum() / arr.sum()
        idx = np.searchsorted(cum_arr, np.linspace(0, 1, self.n_bands, endpoint=False)[1:])
        chunks = np.split(resp, idx)
        w_chunks = np.split(arr, idx)
        resp_sum = [(chunks[x]*w_chunks[x]).sum() for x in range(0,self.n_bands)]
        resp_cum = np.cumsum(resp_sum)
        resp_frac = resp_cum / resp_cum[self.n_bands-1]

        w_resp_sum = [np.sum(x) for x in w_chunks]
        w_resp_cum = np.cumsum(w_resp_sum)
        resp_adr = resp_sum / resp_cum[self.n_bands-1]
        #resp_rate = np.array(resp_sum) / np.array(w_resp_sum)
        #resp_rate_cum = resp_cum / w_resp_cum
        w_resp_frac = w_resp_cum / np.sum(arr)
        #cases_rate = np.asarray(w_resp_sum) / float(np.sum(arr))
        #lift = resp_adr / cases_rate
        #cumulative_lift = resp_frac / w_resp_frac
        return pd.DataFrame(np.c_[w_resp_frac, resp_sum, resp_cum, resp_adr, resp_frac], \
            columns = ['Quantile Value', 'Value', 'Cumulative Value', 'VDR', 'Cumulative VDR']).head(self.rows)
            
    @time_function
    def lift_table_weight(
        self, 
        score_unit_value, 
        ):

        rank_by_score = score_unit_value.sort_values(by=self.predicted_score_numeric, ascending=False)

        units = rank_by_score[self.target_variable].tolist()
        w_units = rank_by_score[self.weight_variable_name].tolist() 
        unit_caprate = self._capture_unit_weight(units, w_units)
            
        values = rank_by_score[self.amount_variable_name].tolist() 
        w_values = rank_by_score[self.weight_variable_name].tolist() 
        value_caprate = self._capture_value_weight(values, w_values)

        return pd.concat([unit_caprate, value_caprate], axis=1)

    def create_lift_table(
        self
        ):
        
        self.lift_table_dict = {} 
        
        for i, j in self.sample_values_dict.items():
            print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
            lift_table = self.lift_table_weight(self.predictions_dictionary['data_{}'.format(i)]).round(decimals=3)
            self.lift_table_dict['data_{}'.format(i)] = lift_table
            display(lift_table)
            lift_table.to_csv(self.data_path + '/output/lift_table_' + str(i) + '.csv', index=False)
            
        return self.lift_table_dict

    @time_function
    def plot_ADR_Quantile(
        self, 
        xlim=None, 
        ylim=None
        ):
       
        for i, j in self.sample_values_dict.items():
            print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
            lt = self.lift_table_dict['data_{}'.format(i)]
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
            plt.savefig('{0}/output/graphs/ADR_{1}.png'.format(self.data_path, i))
            plt.show()

    @time_function
    def plot_cADR_Quantile(
        self, 
        xlim=None, 
        ylim=None
        ):
            
        for i, j in self.sample_values_dict.items():
            print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
            lt = self.lift_table_dict['data_{}'.format(i)]
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
            plt.savefig('{0}/output/graphs/cADR_{1}.png'.format(self.data_path, i))
            plt.show()
            
    @time_function
    def plot_FPR_Quantile(
        self, 
        xlim=None, 
        ylim=None
        ):
            
        for i, j in self.sample_values_dict.items():
            print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
            lt = self.lift_table_dict['data_{}'.format(i)]
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
            plt.savefig('{0}/output/graphs/FPR_{1}.png'.format(self.data_path, i))
            plt.show()

    @time_function
    def plot_cFPR_Quantile(
        self, 
        xlim=None, 
        ylim=None
        ):
            
        for i, j in self.sample_values_dict.items():
            print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
            lt = self.lift_table_dict['data_{}'.format(i)]
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
            plt.savefig('{0}/output/graphs/cFPR_{1}.png'.format(self.data_path, i))
            plt.show()

    @time_function
    def plot_ROC_curve(
        self, 
        target_variable, # Target variable name
        predicted_variable, # Predicted variable name
        weight_variable_name # Weight variable name
        ): 
            
        if weight_variable_name == 'None':
            weight_variable_name = None
        
        for i, j in self.sample_values_dict.items():
            print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
            table_name = self.predictions_dictionary['data_{}'.format(i)]
            
            if not weight_variable_name:
                weight_variable = None
            else: 
                weight_variable = table_name[weight_variable_name]
            
            # generate a no skill prediction (majority class)
            ns_probs = [0 for _ in range(len(table_name[target_variable].values))]
            # calculate scores
            ns_auc = roc_auc_score(table_name[target_variable].values, ns_probs, sample_weight=weight_variable)
            model_auc = roc_auc_score(table_name[target_variable].values, table_name[predicted_variable].values, sample_weight=weight_variable)
            # summarize scores
            print('Random: ROC AUC=%.4f' % (ns_auc))
            print('Model: ROC AUC=%.4f' % (model_auc))
            # calculate roc curves
            ns_fpr, ns_tpr, _ = roc_curve(table_name[target_variable].values, ns_probs, sample_weight=weight_variable)
            model_fpr, model_tpr, _ = roc_curve(table_name[target_variable].values, table_name[predicted_variable].values, sample_weight=weight_variable)
            # plot the roc curve for the model
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random')
            plt.plot(model_fpr, model_tpr, marker='.', label='Model')
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the legend
            plt.legend()
            # save the graph
            plt.savefig('{0}/output/graphs/ROC_{1}.png'.format(self.data_path, i))
            # show the plot
            plt.show()
            
    @time_function
    def plot_precision_recall_curve(
        self, 
        target_variable, # Target variable name
        predicted_variable, # Predicted variable name
        weight_variable_name # Weight variable name
        ):
        
        if weight_variable_name == 'None':
            weight_variable_name = None
            
        for i, j in self.sample_values_dict.items():
            print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
            table_name = self.predictions_dictionary['data_{}'.format(i)]
            
            if not weight_variable_name:
                weight_variable = None
            else: 
                weight_variable = table_name[weight_variable_name]
                
            # predict class values
            model_precision, model_recall, _ = precision_recall_curve(table_name[target_variable].values, table_name[predicted_variable].values, sample_weight=weight_variable)
            # plot the precision-recall curves
            if not weight_variable_name:
                no_skill = table_name[target_variable].values.sum() / len(table_name)
            else: 
                no_skill = weight_variable.values[table_name[target_variable].values==1].sum() / weight_variable.values.sum()
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
            plt.plot(model_recall, model_precision, marker='.', label='Model')
            # axis labels
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # show the legend
            plt.legend()
            # save the graph
            plt.savefig('{0}/output/graphs/precision_recall_{1}.png'.format(self.data_path, i))
            # show the plot
            plt.show()

    @time_function
    def plot_cutoffs(
        self, 
        target_variable, # Target variable name
        predicted_variable, # Predicted variable name
        weight_variable_name, # Weight variable name
        n_bands, # Number of bands between 0 and 1
        return_table=False # Set to True in order to return the table that produced the graph, otherwise set to False
        ):
     
        if weight_variable_name == 'None':
            weight_variable_name = None
            
        for i, j in self.sample_values_dict.items():
            print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
            table_name = self.predictions_dictionary['data_{}'.format(i)]
            
            if not weight_variable_name:
                weight_variable = None
            else: 
                weight_variable = table_name[weight_variable_name]

            threshold_array = np.linspace(0,1,n_bands+1, endpoint=False)
#            column_names = ['cutoff', 'f1', 'accuracy', 'sensitivity/recall', 'specificity', 'precision']
#            df = pd.DataFrame(columns = column_names)
            df_empty = pd.DataFrame(columns=['cutoff', 'f1', 'accuracy', 'sensitivity/recall', 'specificity', 'precision'])
            dataframes = []

            for threshold in threshold_array: 
                y_hat = (table_name[predicted_variable].values >= threshold)*1
                cm = confusion_matrix(table_name[target_variable].values, y_hat, sample_weight=weight_variable)

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
#                if sys.version_info[0] < 2:
#                    df = df.append({'cutoff':threshold, 'f1':model_f1, 'accuracy':model_accuracy, 'sensitivity/recall':model_sensitivity, 'specificity':model_specificity, 'precision':model_precision}, ignore_index=True)
#                else: 
#                    df = pd.concat([df, pd.DataFrame({'cutoff':[threshold], 'f1':[model_f1], 'accuracy':[model_accuracy], 'sensitivity/recall':[model_sensitivity], 'specificity':[model_specificity], 'precision':[model_precision]})], ignore_index=True)
                df_new = pd.DataFrame({'cutoff':[threshold], 'f1':[model_f1], 'accuracy':[model_accuracy], 'sensitivity/recall':[model_sensitivity], 'specificity':[model_specificity], 'precision':[model_precision]})
                dataframes.append(df_new)

            df = pd.concat([df_empty] + dataframes, ignore_index=True)
                
            # create overplot
            plt.plot(df['cutoff'], df['f1'], marker='.', label='F1 score')
            plt.plot(df['cutoff'], df['accuracy'], linestyle='--', label='Accuracy')
            plt.plot(df['cutoff'], df['sensitivity/recall'], marker='.', linestyle='--', label='Sensitivity/Recall')
            plt.plot(df['cutoff'], df['specificity'], linestyle='dotted', label='Specificity')
            plt.plot(df['cutoff'], df['precision'], marker='.', linestyle='dashdot', label='Precision')
            # axis labels
            plt.xlabel('Cutoff')
            plt.ylabel('Metrics')
            # show the legend
            plt.legend()
            # save the graph
            plt.savefig('{0}/output/graphs/metrics_{1}.png'.format(self.data_path, i))
            # show the plot
            plt.show()
            
            if return_table == True:
                display(df)
                
#############################################################################################################################################
#############################################################################################################################################

class FeatureImportance:
    def __init__(
        self, 
        X, 
        labels, 
        weights, 
        data_path, 
        filename = 'FeatureImportance'
    ):
    
        self.X = X
        self.labels = labels
        self.weights = weights
        self.data_path = data_path
        self.filename = filename
        self.ordered_feature_names = X.columns.sort_values().tolist()
        self.feature_importance = pd.DataFrame()
        
    def get_feature_imp_unsup2sup(
        self, 
        df
    ):
    
        cluster_feature_weights = {}
        for label in set(self.labels):
            binary_enc = np.vectorize(lambda x: 1 if x==label else 0)(self.labels)
            clf = RandomForestClassifier(random_state=10)
            clf.fit(df, binary_enc, sample_weight = self.weights)
            
            sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1]
            
            ordered_cluster_features = np.take_along_axis(
                np.array(self.ordered_feature_names), 
                sorted_feature_weight_idxes, 
                axis=0)
            ordered_cluster_feature_weights = np.take_along_axis(
                np.array(clf.feature_importances_), 
                sorted_feature_weight_idxes, 
                axis=0)
            cluster_feature_weights[label] = list(zip(ordered_cluster_features, 
                                                        ordered_cluster_feature_weights))
        self.feature_importances = cluster_feature_weights
        return cluster_feature_weights
        
        
    def one_hot_encode(
        self, 
        columns
    ):
        
        if len(columns) == 0: 
            return pd.DataFrame()
        concat_df = pd.concat([pd.get_dummies(self.X[col], drop_first=True, prefix=col) for col in columns], axis=1)
        
        return concat_df
        
    @time_function
    def get_report(
        self
        ):
    
        if len(self.X.select_dtypes(include='object').columns) != 0: 
            cat_cols = self.X.select_dtypes(include='object')
            cat_one_hot_df = self.one_hot_encode(cat_cols.columns)
            df = self.X.join(cat_one_hot_df).drop(cat_cols, axis=1)
        else: 
            df = self.X
            
        cols = df.columns
        self.ordered_feature_names = cols.tolist()
        
        # Feature importance for each cluster 
        self.get_feature_imp_unsup2sup(df)
        # Save feature importances 
        for label in set(self.labels):
            pd.DataFrame(self.feature_importances[label]).round(2).to_csv(f'{self.data_path}/output/{self.filename}_feature_imprtnc{label}.csv', index=False)
            
        # Overall feature importance, scale by cluster weight (% population)
        unique_elements, counts_elements = np.unique(self.labels, return_counts=True)
        
        label_n_weight = pd.DataFrame(self.labels)
        label_n_weight = label_n_weight.rename(columns={0: "label"})
        label_n_weight['weight_variable'] = self.weights

        weighted_list = []
        for value in unique_elements:
            weighted_list.append(label_n_weight[label_n_weight['label']==value]['weight_variable'].sum() / label_n_weight['weight_variable'].sum())
        weight = dict(zip(unique_elements, weighted_list))        
        wght_feat_imp = pd.DataFrame({'Feature': np.sort(self.ordered_feature_names)})
        for label in set(self.labels):
            wght_feat_imp[label] = (pd.DataFrame(self.feature_importances[label]).sort_values(by=0)[1] * weight[label]).values
        wght_feat_imp.set_index('Feature', inplace=True)
        wght_feat_imp['overall_feature_importance'] = wght_feat_imp.sum(axis=1)
        
        res = pd.DataFrame(wght_feat_imp)
        res = res.sort_values(by='overall_feature_importance', ascending=False).round(2)
        res.to_csv(self.data_path + '/output/' + self.filename + '.csv')
        return res

    @time_function
    def feature_importance_keep_vars(
        self, 
        feature_importance_threshold
        ):

        fi_table = pd.read_csv('{0}/output/{1}.csv'.format(self.data_path, self.filename), sep=',')
        return list(fi_table[fi_table['overall_feature_importance'] > feature_importance_threshold]['Feature'])

#############################################################################################################################################
#############################################################################################################################################
class clustering_report:
    
    def __init__(
        self, 
        input_data, 
        cluster_variable_name, 
        weight_variable_name, 
        data_path
    ):
    
        self.input_data = input_data
        self.cluster_variable_name = cluster_variable_name
        self.weight_variable_name = weight_variable_name
        self.data_path = data_path

    def weighted_mean_group(
        self, 
        df,
        data_col,
        weight_col,
        by_col
        ):
        
        gr = df.groupby(by_col)
        return gr.apply(lambda x: np.average(x[data_col].dropna(), weights=x[~x[data_col].isnull()][weight_col]))
        
    def weighted_median_group(
        self, 
        df,
        data_col,
        weight_col,
        by_col
        ):
        
        gr = df.groupby(by_col)
        return gr.apply(lambda x: wghtd.median(x[data_col].dropna(), x[~x[data_col].isnull()][weight_col]))
        
    def CountFrequency(
        self, 
        df, 
        my_list, 
        weight, 
        normalize=False
        ):
            
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
        self, 
        df,
        data_col,
        weight_col,
        by_col, 
        normalize=False
        ):
        
        gr = df.groupby(by_col)
        return gr.apply(lambda x: self.CountFrequency(x, data_col, weight_col, normalize=normalize))

    @time_function
    def numeric_summary_statistics(
        self, 
        variable_list 
        ):
        
        # Ensure that the graph folder exists
        if not os.path.isdir('{0}/output/graphs'.format(self.data_path)):
            os.makedirs('{0}/output/graphs'.format(self.data_path))
            
        df_stats = pd.DataFrame()
        for var in variable_list:
        
            # Ensure that the graph folder exists
            if not os.path.isdir('{0}/output/graphs/{1}'.format(self.data_path, var)):
                os.makedirs('{0}/output/graphs/{1}'.format(self.data_path, var))
                
            # mean graphs
            col_hist_mean = self.weighted_mean_group(self.input_data,var,self.weight_variable_name,self.cluster_variable_name)
            Y1_overlay = col_hist_mean.tolist()
            X1_overlay = col_hist_mean.index.tolist()
            Y2_overlay = [np.average(self.input_data[var].dropna(), weights=self.input_data[~self.input_data[var].isnull()][self.weight_variable_name])]*len(self.input_data[self.cluster_variable_name].value_counts())
            
            plt.xlabel("Cluster labels")
            plt.ylabel(var)
            plt.bar(X1_overlay, Y1_overlay, color='maroon', width=0.4)
            plt.plot(X1_overlay, Y2_overlay, label='Overall average')
            plt.legend()
            plt.savefig('{0}/output/graphs/{1}/{1}_mean.png'.format(self.data_path, var))
            plt.show()
            
            # median graphs
            col_hist_med = self.weighted_median_group(self.input_data,var,self.weight_variable_name,self.cluster_variable_name)
            Y1_overlay = col_hist_med.tolist()
            X1_overlay = col_hist_med.index.tolist()
            Y2_overlay = [wghtd.median(self.input_data[var].dropna(), self.input_data[~self.input_data[var].isnull()][self.weight_variable_name])]*len(self.input_data[self.cluster_variable_name].value_counts())
            
            plt.xlabel("Cluster labels")
            plt.ylabel(var)
            plt.bar(X1_overlay, Y1_overlay, color='blue', width=0.4)
            plt.plot(X1_overlay, Y2_overlay, label='Overall median')
            plt.legend()
            plt.savefig('{0}/output/graphs/{1}/{1}_median.png'.format(self.data_path, var))
            plt.show()
            
            # Create summary statistics table
            #df_stats_temp = pd.DataFrame(self.weighted_mean_group(self.input_data, var, self.weight_variable_name, self.cluster_variable_name)).rename(columns={0: var}).T
            df_stats_temp = pd.DataFrame(col_hist_mean).rename(columns={0: var}).T
            df_stats_temp['Baseline'] = [np.average(self.input_data[var].dropna(), weights=self.input_data[~self.input_data[var].isnull()][self.weight_variable_name])]
            for i in np.unique(self.input_data['cluster_labels']):
                df_stats_temp["{0}{1}".format(i, '_baseline_diff')] = (df_stats_temp[i] - df_stats_temp['Baseline']) / df_stats_temp['Baseline']
            df_stats_temp = round(df_stats_temp, 2)
            
            # Add a column that identifies which cluster is more prominent 
            df_stats_columns = df_stats_temp.columns.tolist()
            df_stats_columns_str = [x for x in df_stats_columns if isinstance(x, str)]
            columns_to_select = [col for col in df_stats_columns_str if col.endswith('_baseline_diff')]
            df_stats_temp["Prominent cluster"] = df_stats_temp[columns_to_select].abs().idxmax(axis=1).str.replace('_baseline_diff', '', regex=False)
            
            # Format the columns
            for i in np.unique(self.input_data['cluster_labels']):
                df_stats_temp["{0}{1}".format(i, '_baseline_diff')] = ["{:.2%}".format(i) for i in df_stats_temp["{0}{1}".format(i, '_baseline_diff')]]
            
            df_stats = pd.concat([df_stats, df_stats_temp], ignore_index=False)
            
        df_stats = df_stats.reset_index().rename(columns={'index': "Attribute"})
        df_stats.to_csv('{}/output/summary_statistics_numeric.csv'.format(self.data_path), index=False)
        display(df_stats)
                
    @time_function
    def character_summary_statistics(
        self, 
        variable_list 
        ):
        
        # Ensure that the graph folder exists
        if not os.path.isdir('{0}/output/graphs'.format(self.data_path)):
            os.makedirs('{0}/output/graphs'.format(self.data_path))
            
        df_stats = pd.DataFrame()
        for var in variable_list:
            t1 = pd.DataFrame(self.weighted_frequency_group(self.input_data, var, self.weight_variable_name, self.cluster_variable_name, normalize=True).sort_index()).T.stack().reset_index().rename(columns={'level_1': 'labels'})
            t2 = pd.DataFrame(self.CountFrequency(self.input_data, var, self.weight_variable_name, normalize=True).sort_index()).reset_index().rename(columns={'index': 'labels', 0: 'Baseline'})
            t12 = t1.join(t2, rsuffix='_1').drop('labels_1', axis=1).rename(columns={'level_0': "Attribute"})
            t12['Attribute'] = var
            
            plot = t12.plot(kind='bar', legend=True, xlabel=var, ylabel='Percentage')
            plot.figure.savefig('{0}/output/graphs/{1}_cat_hist.png'.format(self.data_path, var))
            
            for i in np.unique(self.input_data['cluster_labels']):
                t12["{0}{1}".format(i, '_baseline_diff')] = (t12[i] - t12['Baseline']) / t12['Baseline']
            t12 = round(t12, 2)
            
            # Add a column that identifies which cluster is more prominent 
            df_stats_columns = t12.columns.tolist()
            df_stats_columns_str = [x for x in df_stats_columns if isinstance(x, str)]
            columns_to_select = [col for col in df_stats_columns_str if col.endswith('_baseline_diff')]
            t12["Prominent cluster"] = t12[columns_to_select].abs().idxmax(axis=1).str.replace('_baseline_diff', '', regex=False)
            
            # Format the columns
            for i in np.unique(self.input_data['cluster_labels']):
                t12["{0}{1}".format(i, '_baseline_diff')] = ["{:.2%}".format(j) for j in t12["{0}{1}".format(i, '_baseline_diff')]]
                t12[i] = ["{:.2%}".format(i) for i in t12[i]]
            t12["Baseline"] = ["{:.2%}".format(i) for i in t12['Baseline']]
            
    #        df_stats = df_stats.append(t12)
            df_stats = pd.concat([df_stats, t12], ignore_index=False)
            
        df_stats.to_csv('{}/output/summary_statistics_character.csv'.format(self.data_path), index=False)
        display(df_stats)

#############################################################################################################################################
#############################################################################################################################################





        