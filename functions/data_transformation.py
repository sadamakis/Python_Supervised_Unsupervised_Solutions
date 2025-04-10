import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA 
from matplotlib import pyplot as plt
import os 


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

        if weight_variable == 'None':
            weight_variable = None
    
        if self.imputation_strategy == 'median':
            # Import the wquantiles library
            import weighted as wghtd
            for x in self.variables:
                if weight_variable == None:
                    self.impute_missing[x] = input_data[x].median()
                else: 
                    self.impute_missing[x] = wghtd.median(input_data[x].dropna(), input_data[~input_data[x].isnull()][weight_variable])
        elif self.imputation_strategy == 'mean':
            for x in self.variables:
                if weight_variable == None:
                    self.impute_missing[x] = input_data[x].mean()
                else: 
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
    return input_data, outlier_info

@time_function
def character_to_binary(
    input_data, 
    input_variable_list, 
    drop, # Specifies which value to drop from the one hot encoder. None will return binary variables for all categories. 'first' will drop the most populated category. 'last' will drop the least populated category. 
    protected_class_valid_values = None # Specifies accepted values for the protected class column. For non-protected class conversions use 'None'
    ):
        
    input_data = input_data.copy()
    
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
                    
    return input_data

@time_function
def standardize_data(
    input_data, 
    input_variable_list, 
    training_sample, 
    weight_variable, 
    data_path, 
    filename = 'standard_scaler.pkl'
    ):
        
    if weight_variable == 'None':
        weight_variable = None

    standardized_data = {}
    standardized_data_weight = {}

    if not weight_variable:
        std = StandardScaler().fit(input_data[training_sample][input_variable_list])
        pickle.dump(std, open(data_path + '/output/' + filename, 'wb'))
        
        for k in input_data.keys():
            standardized_data[k] = pd.DataFrame(std.transform(input_data[k][input_variable_list]), index=input_data[k].index)
            standardized_data[k].columns = input_data[k][input_variable_list].columns

    else: 
        df0 = input_data[training_sample][input_variable_list].copy()
        weights_sample0 = input_data[training_sample][weight_variable].copy()
        # Compute weighted mean & std from training sample
        weighted_mean = np.average(df0, axis=0, weights=weights_sample0)
        weighted_var = np.average((df0 - weighted_mean) ** 2, axis=0, weights=weights_sample0)
        weighted_std = np.sqrt(weighted_var)

        for k in input_data.keys():
            # Standardize samples using training sample's mean & std
            df0_standardized = (df0 - weighted_mean) / weighted_std
            standardized_data[k] = (input_data[k][input_variable_list] - weighted_mean) / weighted_std
            standardized_data_weight[k] = input_data[k][weight_variable]
            standardized_data_weight[k].index = input_data[k].index
        
    return standardized_data, standardized_data_weight
    

class PCA_reduction: 

    def __init__(
        self, 
        input_data, 
        input_data_weights, 
        data_path, 
        training_sample
        ):
        
        self.input_data = input_data
        self.input_data_weights = input_data_weights
        self.data_path = data_path
        self.training_sample = training_sample
        
    @time_function
    def explore(
        self, 
        solver='full'
        ):

        # Ensure that the graph folder exists
        if not os.path.isdir('{0}/output/graphs'.format(self.data_path)):
            os.makedirs('{0}/output/graphs'.format(self.data_path))

        num_predictors = self.input_data[self.training_sample].shape[1]
        
        if self.input_data_weights == {}:
            pca = PCA(n_components = num_predictors, svd_solver = solver).fit(self.input_data[self.training_sample])
        
        else: 
            # Calculate the weighted covariance matrix
            weighted_cov = np.dot(self.input_data[self.training_sample].T, self.input_data[self.training_sample] * (self.input_data_weights[self.training_sample].values)[:, np.newaxis])
            pca = PCA(n_components = num_predictors, svd_solver = solver).fit(weighted_cov)
        
        PC_values = np.arange(pca.n_components_) + 1
        
        #The second derivative can be used to identify the 'elbow' of a function. 
        #If the function is monotonically increasing, then the elbow can be given by the minimum of the central second derivative approximation. 
        #Central derivative approximation: x[i+1] + x[i-1] - 2*x[i]
        #If we use the forward second derivative approximation, then there is a lag of 1. This means that the 'elbow' is given by the next point that minimizes the second derivative. 
        #Forward derivative approximation: x[i+2] + x[i] - 2*x[i+1]
        #If the function is monotonically decreasing, then the elbow can be given by the maximum of the central second derivative approximation.
        #We use the central derivative approximation to for the explained variance ratio.
        stats_dict = {'components': PC_values, 
                        'explained_variance_ratio': pca.explained_variance_ratio_, 
                        'explained_variance_ratio_cumsum': pca.explained_variance_ratio_.cumsum()
                        }
        # Create a DataFrame from the dictionary
        criteria_df = pd.DataFrame(stats_dict)
        criteria_df['explained_variance_ratio_second_der'] = (
            criteria_df['explained_variance_ratio'].shift(-1) - 2 * criteria_df['explained_variance_ratio'] + criteria_df['explained_variance_ratio'].shift(1)
            )

        # Plot Scree plot and output to graphs folder
        plt.plot(criteria_df['components'], criteria_df['explained_variance_ratio'], 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.savefig(self.data_path + "/output/graphs/PCA_scree_plot.png")
        plt.show()

        # Plot the second derivative for the Scree plot and output to graphs folder. The elbow can be given by the maximum of the central second derivative approximation.
        plt.plot(criteria_df['components'], criteria_df['explained_variance_ratio_second_der'], 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot second derivative')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained second derivative')
        plt.savefig(self.data_path + "/output/graphs/PCA_scree_plot_second_derivative.png")
        plt.show()
        
        # Plot Cumulative variance plot and output to graphs folder
        plt.plot(criteria_df['components'], criteria_df['explained_variance_ratio_cumsum'], 'o-', linewidth=2, color='blue')
        plt.title('Cumulative Variance Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Variance Explained')
        plt.savefig(self.data_path + "/output/graphs/PCA_cumulative_variance_plot.png")
        plt.show()
        
        display(criteria_df)

#        print("Variance explained by each principal component:\n", pca.explained_variance_ratio_)
#        print("Cumulative sum of variance explained by each principal component:\n", pca.explained_variance_ratio_.cumsum())
        
        return pca
        
    def fit_transform(
        self, 
        pca_components, 
        solver='full', 
        filename='pca_model.pkl'
        ):

        out = {}
        
        # Make PCA object to transform data
        if self.input_data_weights == {}:
            pca = PCA(n_components = pca_components, svd_solver = solver).fit(self.input_data[self.training_sample])
            
            # loop through data 
            for k in self.input_data.keys(): 
    #            out[k] = pd.DataFrame(pca.transform(self.input_data[k]))
                pca_data = pca.transform(self.input_data[k])
                out[k] = pd.DataFrame(pca_data, index=self.input_data[k].index) 
            
            pickle.dump(pca, open(self.data_path + '/output/' + filename, 'wb'))
        
        else: 
            # Calculate the weighted covariance matrix
            weighted_cov = np.dot(self.input_data[self.training_sample].T, self.input_data[self.training_sample] * (self.input_data_weights[self.training_sample].values)[:, np.newaxis])
#            pca = PCA(n_components = pca_components, svd_solver = solver).fit(weighted_cov)            

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov)

            # Sort eigenvalues in descending order and get the corresponding eigenvectors
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues_sorted = eigenvalues[sorted_indices]
            eigenvectors_sorted = eigenvectors[:, sorted_indices]

            # Choose the principal components we want to keep
            principal_components = eigenvectors_sorted[:,:pca_components]

            # loop through data 
            for k in self.input_data.keys(): 
                # Project the data onto the principal components
    #            out[k] = pd.DataFrame(pca.transform(self.input_data[k]))
#                pca_data = pca.transform(self.input_data[k])
#                out[k] = pd.DataFrame(pca_data, index=self.input_data[k].index) 
                out[k] = pd.DataFrame(np.dot(self.input_data[k], principal_components), index=self.input_data[k].index)

        return out
