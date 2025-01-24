import statsmodels.api as sm
import pandas as pd
import numpy as np
from io import StringIO

from decorators import time_function 
import useful_functions as ufun

class logistic_regression():
    
    def __init__(
    self, 
    input_data, 
    final_feats, 
    target_variable, 
    weight_variable_name, 
    data_path
    ):
        
        self.input_data = input_data
        self.final_feats = final_feats
        self.target_variable = target_variable
        self.weight_variable_name = weight_variable_name
        self.data_path = data_path

        # Dictionary to save each summary table for each sample
        self.glm_bin_summary = {}
        
    @time_function
    def glm_bin(
    self, 
    sample_values_solution
    ):

        i = sample_values_solution[0]
        df = self.input_data['data_{}'.format(i)]
        X = sm.add_constant(df[self.final_feats])
        Y = df[self.target_variable]
        
        # Build the model and fit the data
        self.glm_binom = sm.GLM(Y, X, family=sm.families.Binomial(), freq_weights=df[self.weight_variable_name]).fit()
        
        summary_results = self.glm_binom.summary()
        results_as_csv = summary_results.tables[1].as_csv()
        results_str = StringIO(results_as_csv)
        
        self.glm_bin_summary['log_reg_summary_{}'.format(i)] = pd.read_csv(results_str, sep=',', skipinitialspace=True)
        self.glm_bin_summary['log_reg_summary_{}'.format(i)].columns = ['variable', 'coef', 'std_err', 'z', 'p_value', '[0.025', '0.975]']
        self.glm_bin_summary['log_reg_summary_{}'.format(i)]['variable'] = self.glm_bin_summary['log_reg_summary_{}'.format(i)]['variable'].str.strip()
        self.glm_bin_summary['log_reg_summary_{}'.format(i)]['statistically_significant'] = np.where(self.glm_bin_summary['log_reg_summary_{}'.format(i)]['p_value'] < 0.05, 'Yes', 'No')
        self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] = np.exp(self.glm_bin_summary['log_reg_summary_{}'.format(i)]['coef'])
        
        odds_ratio_condition = [
            (self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] < 1), 
            (self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] > 1), 
            (self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] == 1)]
            
        desc_value_rules = [1 - self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'], 
                            self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] - 1, 
                            1]
        self.glm_bin_summary['log_reg_summary_{}'.format(i)]['desc_value'] = np.select(odds_ratio_condition, desc_value_rules)
        return self.glm_binom, self.glm_bin_summary

    @time_function
    def glm_report(
        self
        ): 
        
        for i, j in self.glm_bin_summary.items():
            temp = self.glm_bin_summary[i]
            temp = temp.drop(['[0.025', '0.975]'], axis=1).round(4)
            display(temp)
            pd.DataFrame(temp).to_csv(self.data_path + '/output/' + str(i) + '.csv', index=False)
        return temp

    @time_function
    def create_predictions(
        self, 
        sample_values_dict, 
        amount_variable_name
    ):
        
        self.pred_dict = self.input_data.copy()

        for i, j in sample_values_dict.items():
            df = self.pred_dict['data_{}'.format(i)]
            X = sm.add_constant(df[self.final_feats])
            y_hat = self.glm_binom.predict(X)
            y_0 = list(map(round, y_hat))
            pred_dict_temp = df[[self.target_variable, self.weight_variable_name, amount_variable_name]].copy()
            pred_dict_temp['predicted_score_numeric'] = y_hat
            pred_dict_temp['predicted_score_binary'] = y_0
            self.pred_dict['data_{}'.format(i)] = pred_dict_temp
        
        return self.pred_dict
    
#############################################################################################################################################
#############################################################################################################################################





            