import pandas as pd
import time

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




    