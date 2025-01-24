import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from decorators import time_function

#COPIED
def create_missing_info_list(
    input_data, 
    weight_variable, 
    missing_list_file
    ):
    
    missing_list = input_data.apply(lambda x: (sum(input_data[x.isnull()][weight_variable])/sum(input_data[weight_variable])) * 100, axis=0).sort_values(ascending=False)
    missing_list_df = pd.DataFrame(missing_list)
    missing_list_df.columns = ['Missing Value Percentage']
    missing_list_df.to_csv(missing_list_file, header=True, index=True, index_label='variable')
    return missing_list_df
    
#COPIED
def select_missing_variables_to_drop(
    data_path, 
    sample_name, 
    threshold
    ):
    
    missing_list = pd.read_csv('{0}/output/missing_values_results_{1}.csv'.format(data_path, sample_name), sep=',')
    return list(missing_list.loc[missing_list['Missing Value Percentage'] > threshold*100, 'variable'])
    
#COPIED
def identify_character_variables(
    input_data
    ): 
    
    return input_data.columns[input_data.dtypes == object]
    
#COPIED
def identify_numeric_variables(
    input_data
    ): 
    
    return input_data.columns[input_data.dtypes != object]
    
#COPIED
def character_classification(
    input_data, 
    input_variable_list, 
    threshold=50
    ):
    
    cat_vars = {'single':[], 'binary':[], 'small':[], 'large':[]}
    for x in input_variable_list:
        n_values = len(input_data[x].value_counts(dropna=False).index)
        if n_values == 1:
            cat_vars['single'].append(x)
        elif n_values == 2:
            cat_vars['binary'].append(x)
        elif n_values <= threshold:
            cat_vars['small'].append(x)
        else:
            cat_vars['large'].append(x)

    return cat_vars
    
#COPIED
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
                    
#COPIED
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
            
            
def data_split(
    input_data, 
    fraction = 0.7, 
    random_state = 1
    ):
    
    df = input_data.sample(frac=1, replace=False, random_state=random_state)
    cut = int(len(df)*fraction)
    dev = df.head(cut)
    oos = df.tail(len(df) - cut)
    return dev, oos
    
#COPIED
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
    
#COPIED
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
    
#COPIED
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
                            'Lower limit': "{0:.2f}%".format(low_limit), 
                            'Upper limit': "{0:.2f}%".format(up_limit)}, index=[0])], 
                            ignore_index=False)
    
    outlier_info = outlier_info.sort_values(by='variable', ascending=True)
    outlier_info.to_csv('{0}/output/{1}'.format(data_path, outlier_info_file), header=True, index=False)
    display(outlier_info)
    return input_data
    
@time_function
def standardize_data(
    input_data, 
    variables, 
    training_sample, 
    data_path, 
    filename = 'standard_scaler.pkl'
    ):
    
    out = {}
    std = StandardScaler().fit(input_data[training_sample][variables])
    pickle.dump(std, open(data_path + '/output/' + filename, 'wb'))
    
    for k in input_data.keys():
        out[k] = pd.DataFrame(std.transform(input_data[k][variables]))
        out[k].columns = input_data[k][variables].columns
        
    return out
