import numpy as np
import pandas as pd
import time

from decorators import time_function 
import useful_functions as ufun

@time_function
def weight_var_assignment(
    input_data, 
    weight_variable
    ):
    
    if weight_variable == 'None':
        weight_variable = None
    
    if not weight_variable: 
        weight_variable_name_solution = 'weight_variable_solution'
        input_data[weight_variable_name_solution] = 1
    else: 
        weight_variable_name_solution = weight_variable
    
    return input_data, weight_variable_name_solution
    
@time_function
def sample_var_assignment(
    input_data, 
    sample_variable, 
    sample_values
    ):

    if sample_variable == 'None':
        sample_variable = None
    
    if not sample_variable: 
        sample_variable_name_solution = 'sample_variable_solution'
        sample_values_solution = ['training']
        input_data[sample_variable_name_solution] = 'training'
    else: 
        sample_variable_name_solution = sample_variable
        sample_values_solution = sample_values
        
    return input_data, sample_values_solution, sample_variable_name_solution
    
@time_function
def amount_var_assignment(
    input_data, 
    amount_variable
    ):
    
    if amount_variable == 'None':
        amount_variable = None
    
    if not amount_variable: 
        amount_variable_name_solution = 'amount_variable_solution'
        input_data[amount_variable_name_solution] = 0
    else: 
        amount_variable_name_solution = amount_variable
    
    return input_data, amount_variable_name_solution

@time_function
def convert_character_var(
    input_data, 
    character_variables,
    sample_variable
    ):
    
    character_variables_list = list(filter(None, character_variables+[sample_variable]))
    print()
    print(ufun.color.BLUE+'Character variables: '+ufun.color.END+str(character_variables_list))
    input_data[character_variables_list] = input_data[character_variables_list].astype(str).replace(['nan', 'NaN', 'None'], np.nan)
    
    return input_data, character_variables_list
    
@time_function
def convert_numeric_var(
    input_data, 
    numeric_variables,
    weight_variable, 
    amount_variable, 
    target_variable
    ):
    
    numeric_variables_list = list(filter(None, numeric_variables+[target_variable, weight_variable, amount_variable]))
    print()
    print(ufun.color.BLUE+'Numeric variables: '+ufun.color.END+str(numeric_variables_list))
    input_data[numeric_variables_list] = input_data[numeric_variables_list].astype(float)
    
    return input_data, numeric_variables_list
        
class impute_missing(object):

    def __init__(
        self, 
        variables, 
        imputation_strategy = "median" # The imputation strategy: current values are "median", "mean", or a specific value without quotes, e.g. 0
    ):
    
        self.variables = variables
        self.impute_missing = dict()
        self.imputation_strategy = imputation_strategy
        
    def imputation_fit(
        self, 
        input_data
    ):
    
        if self.imputation_strategy == 'median':
            for x in self.variables:
                self.impute_missing[x] = input_data[x].median()
        elif self.imputation_strategy == 'mean':
            for x in self.variables:
                self.impute_missing[x] = input_data[x].mean()
        else:
            for x in self.variables:
                self.impute_missing[x] = self.imputation_strategy
        return self
        
    def imputation_fit_weight(
        self, 
        input_data, 
        weight_variable
    ):
    
        if self.imputation_strategy == 'median':
            # Import the wquantiles library
            import weighted as wghtd
            for x in self.variables:
                self.impute_missing[x] = wghtd.median(input_data[x].dropna(), input_data[~input_data[x].isnull()][weight_variable])
                #self.impute_missing[x] = input_data[x].median()
        elif self.imputation_strategy == 'mean':
            for x in self.variables:
                self.impute_missing[x] = np.average(input_data[x].dropna(), weights=input_data[~input_data[x].isnull()][weight_variable])
        else:
            for x in self.variables:
                self.impute_missing[x] = self.imputation_strategy
        return self
        
    def imputation_transform(
        self, 
        input_data
    ):
    
        for x in self.variables:
            input_data.loc[:, x] = input_data.loc[:, x].fillna(self.impute_missing[x])
    
@time_function
def split_sample_data(
    input_data, 
    sample_values_solution, 
    sample_variable_name_solution
    ):
    
    sample_values_supportive_list = list(range(len(sample_values_solution)))
    dictionary_values = []
    for i in range(len(sample_values_solution)):
        dictionary_values.append('SAMPLE {}'.format(sample_values_supportive_list[i]))
        
    sample_values_dict = dict(zip(sample_values_solution, dictionary_values))
    print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE +'Sample data dictionary: '+ufun.color.END + str(sample_values_dict))
    print() 
    data = {}
    for i, j in sample_values_dict.items():
        start_time = time.time()
        print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
        
        data['data_{}'.format(i)] = input_data[input_data[sample_variable_name_solution]==i]
        print('The shape is: ', data['data_{}'.format(i)].shape)
        
        print('Creating this sample took %.2fs. to run'%(time.time() - start_time))
        
    return data, sample_values_dict  

def outlier_thresholds(
    dataframe, 
    variable, 
    iqr_coef
    ):
        
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + iqr_coef * interquantile_range
    low_limit = quartile1 - iqr_coef * interquantile_range
    return low_limit, up_limit
    
def replace_with_thresholds(
    dataframe, 
    variable, 
    weight_variable, 
    iqr_coef
    ):

    if weight_variable == 'None':
        weight_variable = None
    
    low_limit, up_limit = outlier_thresholds(dataframe, variable, iqr_coef)
    if weight_variable == None:
        below_low_limit = np.sum(dataframe[variable] < low_limit) / dataframe.shape[0]
        above_up_limit = np.sum(dataframe[variable] > up_limit) / dataframe.shape[0]
    else: 
        below_low_limit = np.sum(dataframe[dataframe[variable] < low_limit][weight_variable]) / np.sum(dataframe[weight_variable])
        above_up_limit = np.sum(dataframe[dataframe[variable] > up_limit][weight_variable]) / np.sum(dataframe[weight_variable])
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    return dataframe, below_low_limit, above_up_limit, up_limit, low_limit
    
@time_function
def replace_outliers(
    input_data, 
    variables, 
    weight_variable, 
    data_path, 
    outlier_info_file = 'outlier_info.csv', 
    iqr_coef = 1.5
    ):

    outlier_info = pd.DataFrame({'variable': pd.Series(dtype='str'), 
                            '%_below_lower_limit': pd.Series(dtype='float'), 
                            '%_above_upper_limit': pd.Series(dtype='float'), 
                            'Lower limit': pd.Series(dtype='float'), 
                            'Upper limit': pd.Series(dtype='float')
                            })
    for col in variables:
        input_data, below_low_limit, above_up_limit, up_limit, low_limit = replace_with_thresholds(input_data, col, weight_variable, iqr_coef)
        outlier_info = pd.concat([outlier_info, pd.DataFrame({'variable': col, 
                            '%_below_lower_limit': "{0:.2f}%".format(below_low_limit*100), 
                            '%_above_upper_limit': "{0:.2f}%".format(above_up_limit*100), 
                            'Lower limit': "{0:.2f}".format(low_limit), 
                            'Upper limit': "{0:.2f}".format(up_limit)}, index=[0])], 
                            ignore_index=False)
    
    outlier_info = outlier_info.sort_values(by='variable', ascending=True)
    outlier_info.to_csv('{0}/output/{1}'.format(data_path, outlier_info_file), header=True, index=False)
    display(outlier_info)
    return input_data

@time_function
def character_to_binary(
    input_data, 
    input_variable_list, 
    drop, # Specifies which value to drop from the one hot encoder. None will return binary variables for all categories. 'first' will drop the most populated category. 'last' will drop the less populated category. 
    protected_class_valid_values = None # Specifies accepted values for the protected class column. For non-protected class conversions use 'None'
    ):
    
    for x in input_variable_list:
        if drop == 'last':
            values = input_data[x].value_counts(dropna=False).index
            values = values[:len(values) - 1]
        elif drop == 'first':
            values = input_data[x].value_counts(dropna=False).index
            values = values[1:]
        elif drop == None:
            values = input_data[x].value_counts(dropna=False).index
        else:
            raise Exception("Please choose a value for the drop argument from the pre-defined list: None, 'first', 'last'")

        for v in values:
            if ((protected_class_valid_values is None) or (v in protected_class_valid_values)):
                if v is not np.nan:
                    input_data[x + '_' + str(v).replace(".", "_")] = input_data[x].map(lambda t: 1 if t==v else 0)
                else: 
                    input_data[x + '_' + str(v).replace(".", "_")] = input_data[x].map(lambda t: 1 if t is v else 0)

