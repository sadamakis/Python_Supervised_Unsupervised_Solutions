import time
import pandas as pd
import numpy as np
import data_processing as cpd
import lasso_feature_selection as lfs
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import WLS
from decorators import time_function 

#COPIED
class color:
    PURPLE = '\u001b[0;35m'
    CYAN = '\u001b[36;1m'
    DARKCYAN = '\u001b[46m'
    BLUE = '\u001b[34m'
    GREEN = '\u001b[32m'
    YELLOW = '\u001b[33m'
    RED = '\u001b[31m'
    BOLD = '\u001b[4m'
    UNDERLINE = '\u001b[31;1;4m'
    END = '\u001b[0m'
    BLACK = '\u001b[30m'
    
    
#COPIED
@time_function
def weight_var_assignment(
    data_full, 
    weight_variable_name
    ):
    
    if weight_variable_name == 'None':
        weight_variable_name = None
    
    if not weight_variable_name: 
        weight_variable_name_solution = 'weight_variable_solution'
        data_full[weight_variable_name_solution] = 1
    else: 
        weight_variable_name_solution = weight_variable_name
    
    return data_full, weight_variable_name_solution
    
#COPIED
@time_function
def sample_var_assignment(
    data_full, 
    sample_variable_name, 
    sample_values
    ):

    if sample_variable_name == 'None':
        sample_variable_name = None
    
    if not sample_variable_name: 
        sample_variable_name_solution = 'sample_variable_solution'
        sample_values_solution = ['training']
        data_full[sample_variable_name_solution] = 'training'
    else: 
        sample_variable_name_solution = sample_variable_name
        sample_values_solution = sample_values
        
    return data_full, sample_values_solution, sample_variable_name_solution
    
#COPIED
@time_function
def convert_character_var(
    data_full, 
    original_candidate_variables_character,
    sample_variable_name_solution
    ):
    
    character_variables_list = list(filter(None, original_candidate_variables_character+[sample_variable_name_solution]))
    print()
    print(color.BLUE+'Character variables: '+color.END+str(character_variables_list))
    data_full[character_variables_list] = data_full[character_variables_list].astype(str).replace(['nan', 'NaN', 'None'], np.nan)
    
    return data_full, character_variables_list
    
#COPIED
@time_function
def convert_numeric_var(
    data_full, 
    original_candidate_variables_numeric,
    weight_variable_name_solution, 
    target_variable_name
    ):
    
    numeric_variables_list = list(filter(None, original_candidate_variables_numeric+[target_variable_name, weight_variable_name_solution]))
    print()
    print(color.BLUE+'Numeric variables: '+color.END+str(numeric_variables_list))
    data_full[numeric_variables_list] = data_full[numeric_variables_list].astype(float)
    
    return data_full, numeric_variables_list
    
#COPIED
@time_function
def split_sample_data(
    data_full, 
    sample_values_solution, 
    sample_variable_name_solution
    ):
    
    sample_values_supportive_list = list(range(len(sample_values_solution)))
    dictionary_values = []
    for i in range(len(sample_values_solution)):
        dictionary_values.append('SAMPLE {}'.format(sample_values_supportive_list[i]))
        
    sample_values_dict = dict(zip(sample_values_solution, dictionary_values))
    print(color.BOLD + color.PURPLE + color.UNDERLINE +'Sample data dictionary: '+color.END + str(sample_values_dict))
    print() 
    data = {}
    for i, j in sample_values_dict.items():
        start_time = time.time()
        print(color.BOLD + color.PURPLE + color.UNDERLINE + j + color.END)
        
        data['data_{}'.format(i)] = data_full[data_full[sample_variable_name_solution]==i]
        print('The shape is: ', data['data_{}'.format(i)].shape)
        
        print('Creating this sample took %.2fs. to run'%(time.time() - start_time))
        
    return data, sample_values_dict
    
    
#COPIED
@time_function
def missing_values_vars(
    sample_values_dict, 
    data_path, 
    data, 
    weight_variable_name_solution, 
    select_missing_variables_to_drop_threshold
    ):
    
    missing_variables_table = {}
    missing_variables = []

    for i, j in sample_values_dict.items(): 
        start_time = time.time()
        print(color.BOLD + color.PURPLE + color.UNDERLINE + j + color.END)
        
        missing_variables_table['missing_variables_table_{}'.format(i)] = cpd.create_missing_info_list(input_data=data['data_{}'.format(i)], weight_variable=weight_variable_name_solution, missing_list_file='{0}/output/missing_values_results_{1}.csv'.format(data_path, j))
        
        print(color.BLACK + 'Creating the missing variables table took %.2fs. to run'%(time.time() - start_time))
        
        missing_variables_temp = cpd.select_missing_variables_to_drop(data_path, j, threshold=select_missing_variables_to_drop_threshold)
        missing_variables = missing_variables + missing_variables_temp
        print()
        
    missing_variables = list(set(missing_variables))
    print(color.PURPLE + 'Variables with too many missing values: ' + color.BLACK + str(missing_variables))
    print()
        
    return missing_variables_table, missing_variables
    
#COPIED
@time_function
def character_var_levels(
    data, 
    data_path, 
    sample_values_solution,
    excluded_variables, 
    character_classification_threshold
    ):
    
    character_vars = cpd.identify_character_variables(input_data=data['data_{}'.format(sample_values_solution[0])])
    keep_char_vars = [x for x in character_vars if x not in excluded_variables]
    print(color.PURPLE + 'Keep character variables' + color.END + str(keep_char_vars))
    print()
    char_classification = cpd.character_classification(input_data=data['data_{}'.format(sample_values_solution[0])], input_variable_list=keep_char_vars, threshold=character_classification_threshold)
    print(color.PURPLE + 'Category variables in 3 classes: ' + color.END)
    
    for k, v in char_classification.items():
        print(k, v)
        
    char_classification_df = pd.DataFrame.from_dict(char_classification, orient='index').T
    char_classification_df.to_csv('{}/output/character_classification_results.csv'.format(data_path))
    
    keep_char_vars_levels = [x for x in keep_char_vars if x not in (char_classification['single'] + char_classification['large'])]
    print() 
    print(color.PURPLE + 'Character variables kept: ' + color.END + str(keep_char_vars_levels))
    print() 
    
    return keep_char_vars_levels
    
#COPIED
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


#COPIED
bic_dict_ = {}
def perform_lasso(
    sample_values_dict, 
    sample_values_solution, 
    data, 
    target_variable_name, 
    predictor_variables, 
    data_path, 
    early_stop=True, 
    weight_variable_name=None, 
    standardization=True, 
    c_min=1e-4, 
    c_max=1e4, 
    num=20, 
    vif_threshold=5, 
    random_state=42
    ):
    
    for i in range(len(sample_values_dict)): 
        print("Displaying evaluation metrics on the " + str(list(sample_values_dict.keys())[i]) + "dataset")
        print()
        train_df = data['data_{}'.format(sample_values_solution[0])]
        validation_df = data['data_{}'.format(sample_values_solution[i])]
        lasso = lfs.lasso_selection(
            list(sample_values_dict.keys())[i], 
            train_df, 
            validation_df, 
            data=data,
            target_variable_name=target_variable_name, 
            predictor_variables=predictor_variables, 
            standardization=standardization, 
            weight_variable_name=weight_variable_name, 
            c_min=c_min,
            c_max=c_max, 
            num=num, 
            vif_threshold=vif_threshold, 
            random_state=random_state, 
            data_path=data_path            
            )
            
        bic_df = lasso.fit()
        bic_dict_[list(sample_values_dict.keys())[i]] = bic_df
    return bic_dict_
