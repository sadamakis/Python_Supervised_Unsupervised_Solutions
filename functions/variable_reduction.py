import pandas as pd
import numpy as np
import time
import os 
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo, FactorAnalyzer
from matplotlib import pyplot as plt
import json

import useful_functions as ufun
from decorators import time_function 

def create_missing_info_list(
    input_data, 
    weight_variable, 
    missing_list_file
    ):
    
    missing_list = input_data.apply(lambda x: (sum(input_data[x.isnull()][weight_variable])/sum(input_data[weight_variable])) * 100, axis=0).sort_values(ascending=False)
    missing_list_df = pd.DataFrame(missing_list)
    missing_list_df.columns = ['Missing Value Percentage']
    missing_list_df['Missing Value Percentage'] = missing_list_df['Missing Value Percentage'].round(2)
    missing_list_df.to_csv(missing_list_file, header=True, index=True, index_label='variable')
    return missing_list_df
    
def select_missing_variables_to_drop(
    data_path, 
    sample_name, 
    threshold
    ):
    
    missing_list = pd.read_csv('{0}/output/missing_values_results_{1}.csv'.format(data_path, sample_name), sep=',')
    return list(missing_list.loc[missing_list['Missing Value Percentage'] > threshold*100, 'variable'])
    
def select_missing_variables_to_drop_dict(
    sample_values_dict, 
    data_path    
    ):
        
    variables_with_missing_dict = {}
    for i, j in sample_values_dict.items():
        start_time = time.time()
        print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
        
        variables_with_missing_dict['variables_with_missing_dict_{}'.format(i)] = select_missing_variables_to_drop(
        data_path = data_path, 
        sample_name = j, 
        threshold = 0
        )
        
        print('This code took %.2fs. to run'%(time.time() - start_time))
    return variables_with_missing_dict

@time_function
def missing_values_vars(
    sample_values_dict, 
    data_path, 
    input_data, 
    weight_variable_name_solution, 
    select_missing_variables_to_drop_threshold
    ):
    
    missing_variables_table = {}
    missing_variables = []

    for i, j in sample_values_dict.items(): 
        start_time = time.time()
        print(ufun.color.BOLD + ufun.color.PURPLE + ufun.color.UNDERLINE + 'SAMPLE ' + i + ufun.color.END)
        
        missing_variables_table['missing_variables_table_{}'.format(i)] = create_missing_info_list(input_data=input_data['data_{}'.format(i)], weight_variable=weight_variable_name_solution, missing_list_file='{0}/output/missing_values_results_{1}.csv'.format(data_path, j))
        
        print(ufun.color.BLACK + 'Creating the missing variables table took %.2fs. to run'%(time.time() - start_time))
        
        missing_variables_temp = select_missing_variables_to_drop(data_path, j, threshold=select_missing_variables_to_drop_threshold)
        missing_variables = missing_variables + missing_variables_temp
        print()
        
    missing_variables = list(set(missing_variables))
    print(ufun.color.PURPLE + 'Variables with too many missing values: ' + ufun.color.BLACK + str(missing_variables))
    print()
        
    return missing_variables_table, missing_variables

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

@time_function
def character_var_levels(
    input_data, 
    data_path, 
    sample_values_solution,
    excluded_variables, 
    character_classification_threshold
    ):
    
    character_vars = ufun.identify_character_variables(input_data=input_data['data_{}'.format(sample_values_solution[0])])
    keep_char_vars = [x for x in character_vars if x not in excluded_variables]
    print(ufun.color.PURPLE + 'Keep character variables' + ufun.color.END + str(keep_char_vars))
    print()
    char_classification = character_classification(input_data=input_data['data_{}'.format(sample_values_solution[0])], input_variable_list=keep_char_vars, threshold=character_classification_threshold)
    print(ufun.color.PURPLE + 'Category variables in 3 classes: ' + ufun.color.END)
    
    for k, v in char_classification.items():
        print(k, v)
        
    char_classification_df = pd.DataFrame.from_dict(char_classification, orient='index').T
    char_classification_df.to_csv('{}/output/character_classification_results.csv'.format(data_path))
    
    keep_char_vars_levels = [x for x in keep_char_vars if x not in (char_classification['single'] + char_classification['large'])]
    print() 
    print(ufun.color.PURPLE + 'Character variables kept: ' + ufun.color.END + str(keep_char_vars_levels))
    print() 
    
    return keep_char_vars_levels

@time_function
def keep_num_variables_one_value(
    keep_num_vars, 
    data_path, 
    dq_report
    ):

    dq_report_table = pd.read_csv('{0}/output/{1}'.format(data_path, dq_report), sep=',')
    num_vars_one_value = list(dq_report_table.loc[dq_report_table['Unique Values'] == 1, 'Variable Name'])
    keep_num_vars_one_v = [x for x in keep_num_vars if x not in num_vars_one_value]
    print(keep_num_vars_one_v)
    print(len(keep_num_vars_one_v))
    return keep_num_vars_one_v


class FactorAnalysis:
    def __init__(
        self, 
        data, 
        training_sample, 
        datapath, 
        filename = 'FactorAnalysis'
    ):

        self.data = data 
        self.X = data[training_sample]
        self.datapath = datapath
        self.filename = filename 
        self.kmo_passed = []
        self.kmo_not_passed = []

    @time_function
    def setup(
        self, 
        kmo_threshold=0.5
    ):
    
        # Ensure that the graph folder exists
        if not os.path.isdir('{0}/output/graphs'.format(self.datapath)):
            os.makedirs('{0}/output/graphs'.format(self.datapath))

        # Bartlett's test of Sphericity
        chi2, p = calculate_bartlett_sphericity(self.X)
        print("Results of Bartlett's test of sphericity:")
        print(f"\tChi squared value : {chi2}")
        print(f"\tp value : {p}")
        print()
        
        # Compute Kaiser-Meyer-Olkin (KMO) test for the original dataset
        kmo_all, kmo_model = calculate_kmo(self.X)
        print("Results of Kaiser-Meyer-Olkin (KMO) test:")
        print("Overall KMO = {:.3f}".format(kmo_model))
        print()

        # Select only adequate variables and recompute KMO 
        self.kmo_passed = list(self.X.columns[kmo_all >= kmo_threshold])
        self.kmo_not_passed = list(self.X.columns[kmo_all < kmo_threshold])
        kmo_all, kmo_model = calculate_kmo(self.X[self.kmo_passed])
        print("Selecting adequate variables and recomputing KMO")
        print("\tOverall KMO = {:.3f}".format(kmo_model))
        print(f"\tVariables with KMO >= {kmo_threshold} = {self.kmo_passed}")
        print(f"\t# of variables with KMO >= {kmo_threshold} = {len(self.kmo_passed)}")
        print(f"\tVariables with KMO < {kmo_threshold} = {self.kmo_not_passed}")
        print(f"\t# of variables with KMO < {kmo_threshold} = {len(self.kmo_not_passed)}")
        print()

        # Determining the number of factors
        fa = FactorAnalyzer(rotation=None, impute='drop', n_factors=self.X[self.kmo_passed].shape[1])
        fa.fit(self.X[self.kmo_passed])
        ev,_ = fa.get_eigenvalues()
        plt.scatter(range(1, self.X[self.kmo_passed].shape[1]+1), ev)
        plt.plot(range(1, self.X[self.kmo_passed].shape[1]+1), ev)
        plt.title('Factor Analysis Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalues')
        plt.grid()
        plt.savefig(f'{self.datapath}/output/graphs/{self.filename}_FA_scree_plot.png')
        plt.show()
    
    @time_function
    def remove_features(
        self, 
        n_factors, 
        loadings_threshold=0.7, 
        **kwargs
    ):

# Set random seed for reproducibility - this does not work very well, results are close, but not exact. 
        np.random.seed(42)
    
        fa = FactorAnalyzer(n_factors=n_factors, **kwargs)
        fa.fit(self.X[self.kmo_passed])
        
        # Get factor loadings 
        loadings = pd.DataFrame(fa.loadings_, index=self.kmo_passed, columns=[f"Factor {i+1}" for i in range(fa.loadings_.shape[1])])
        loadings_c1 = loadings.copy()
        loadings_c1["Highest Loading"] = loadings_c1.abs().idxmax(axis=1)
        loadings_c1 = loadings_c1.reset_index().rename(columns={'index': 'variable_name'})
        loadings_c1 = loadings_c1.sort_values(by=['Highest Loading', 'variable_name'], ascending=[True, True])
        loadings_c2 = loadings_c1.copy()
        columns_to_style = loadings_c2.columns.tolist()
        columns_to_style = [x for x in columns_to_style if x not in ["Highest Loading", "variable_name"]]  
        loadings_c2 = loadings_c2.style.map(lambda x: 'background-color: yellow' if abs(x) >= loadings_threshold else None, subset=columns_to_style)
        print('Factor loadings table')
        display(loadings_c2)
        loadings_c1.to_csv(f'{self.datapath}/output/{self.filename}_loadings.csv')
        
        # See variables that have high loadings for the same factors 
        res = {factor: [] for factor in loadings.columns}
        to_drop = []
        for f in loadings.columns: 
            high_loadings = loadings.index[loadings[f].abs() > loadings_threshold].tolist()
            res[f] += high_loadings
            # The following code selects one variable from each loading in random
#            to_drop += high_loadings[1:]
            # The following code selects the variable with the highest value in a loading 
            keep_variable = loadings_c1[loadings_c1['Highest Loading'] == f].loc[loadings_c1[loadings_c1['Highest Loading'] == f][f].abs().nlargest(1).index]['variable_name'].iloc[0]
            to_drop_list = high_loadings.copy()
            if to_drop_list!=[]:
                to_drop_list.remove(keep_variable)
            to_drop += to_drop_list
            
        # Dedupe the elements in the to_drop list
        to_drop = list(set(to_drop))
            
        remaining_predictors = self.X.columns.drop(to_drop).tolist()
        with open(f'{self.datapath}/output/{self.filename}_summary.json', 'w') as f:
            json.dump(res, f, indent=4)
        print('Features with high loadings')
        display(res)
        print(f'Features dropped: {to_drop}')
        print(f'Remaining features: {remaining_predictors}')
        print(f'Number of remaining features: {len(remaining_predictors)}')
        
        # Drop variables with high loadings in the same factor
        return {k:v[remaining_predictors] for k, v in self.data.items()}



    