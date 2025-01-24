import pandas as pd
import numpy as np
import time
from statsmodels.tools.tools import add_constant 
from sklearn.metrics import roc_auc_score
from statsmodels.regression.linear_model import WLS
import lasso_feature_selection as lfs

import useful_functions as ufun
from decorators import time_function 

@time_function
def gini_values_weight(
    feats, 
    input_data, 
    target_variable, 
    weight_variable, 
    data_path, 
    gini_info_file = 'gini_info.csv', 
    n_bands=10):
        
    feat_gini = dict()
    for x in feats:
#        feat_gini[x] = abs(gini_weight(input_data[[x, target_variable, weight_variable]].values, n_bands))
        feat_gini[x] = abs(2*roc_auc_score(input_data[target_variable].values, input_data[x].values, sample_weight=input_data[weight_variable].values)-1)
    gini_table = sorted(feat_gini.items(), key=lambda t: t[1], reverse=True)
    
    gini_table = pd.DataFrame(gini_table)
    gini_table = gini_table.rename(columns={0: "variable", 1: "Gini coefficient"})
    gini_table['Gini coefficient'] = gini_table['Gini coefficient'].round(2)

    display(gini_table)    
    
    gini_table.to_csv('{0}/output/{1}'.format(data_path, gini_info_file), header=True, index=False)

    return gini_table 

def weighted_mean(
    feat_1, 
    weight
    ):
    return np.sum(feat_1 * weight) / np.sum(weight)

def weighted_cov(
    feat_1, 
    feat_2, 
    weight
    ):
    return np.sum(weight * (feat_1 - weighted_mean(feat_1, weight)) * (feat_2 - weighted_mean(feat_2, weight))) / np.sum(weight)

def weighted_corr(
    feat_1, 
    feat_2, 
    weight
    ):
    return weighted_cov(feat_1, feat_2, weight) / np.sqrt(weighted_cov(feat_1, feat_1, weight) * weighted_cov(feat_2, feat_2, weight))

@time_function    
def calculate_correlations(
    input_data, 
    features, 
    corr_threshold, 
    weight_variable_name
    ):
    
    correlations = []
    weight = input_data[weight_variable_name]
    feat_df = input_data[features]
    
    for i in range(len(feat_df.columns)): 
        for j in range(i):
            feat_1_name = feat_df.columns[i]
            feat_2_name = feat_df.columns[j]
            feat_1 = feat_df[feat_1_name]
            feat_2 = feat_df[feat_2_name]
            
            correlation = weighted_corr(feat_1, feat_2, weight).round(2)
            correlations.append({'feature_1': feat_1_name, 'feature_2': feat_2_name, 'correlation': correlation})
            
    corr_df = pd.DataFrame(correlations, columns=['feature_1', 'feature_2', 'correlation'])
    corr_df['correlation_abs'] = np.abs(corr_df['correlation'])
    corr_df = corr_df.sort_values(by='correlation_abs', ascending=False).reset_index(drop=True)
    
    corr_to_drop_df = corr_df.loc[corr_df['correlation_abs'] > corr_threshold].drop(['correlation_abs'], axis=1)
    print('The following columns have a correlation above your threshold')
    display(corr_to_drop_df)
    
    return corr_df

def correlation_eliminator(
    features, 
    corr_threshold, 
    top_n, 
    data_path, 
    corrs
    ): 
    
    corr_df = corrs.copy()
    corr_elilminated_predictors = corr_df[corr_df['correlation_abs'] > corr_threshold]['feature_1'].unique()
    corr_df.drop(['correlation_abs'], axis=1, inplace=True)
    
    keep_num_vars_corr = list(set(features) - set(corr_elilminated_predictors))
    corr_non_drop_df = corr_df[corr_df['feature_1'].isin(keep_num_vars_corr) & corr_df['feature_2'].isin(keep_num_vars_corr)]
    
    print()
    print(ufun.color.PURPLE + 'Variables eliminated from correlation: ' + ufun.color.END + str(corr_elilminated_predictors))
    print(ufun.color.PURPLE + 'Number of variables eliminated from correlation: ' + ufun.color.END + str(len(corr_elilminated_predictors)))
    print(ufun.color.PURPLE + 'Keeping the following variables (Correlation < ' + str(corr_threshold) + '): ' + ufun.color.END + str(list(keep_num_vars_corr)))
    print(ufun.color.PURPLE + 'Number of variables kept from correlation: ' + ufun.color.END + str(len(keep_num_vars_corr)))
    print('Here are the top ' + str(top_n) + ' features with the highest correlations after removing highly correlated features')
    
    display(corr_non_drop_df.head(top_n))
    corr_non_drop_df.to_csv(data_path + '/output/correlation_results.csv')
    return corr_elilminated_predictors, keep_num_vars_corr

@time_function
def correlation_elimination(
    method, 
    features, 
    input_data, 
    data_path, 
    corr_threshold, 
    top_n, 
    weight_variable_name, 
    correlations = None, 
    vif_threshold = None, 
    init_vifs = None
    ):
    
    if method == 'correlation':
        corr_elilminated_predictors, keep_num_vars_corr = correlation_eliminator(features=features, 
                                                                                 corr_threshold=corr_threshold, 
                                                                                 top_n=top_n, 
                                                                                 data_path=data_path, 
                                                                                 corrs=correlations)
        return corr_elilminated_predictors, keep_num_vars_corr
        
    elif method == 'VIF': 
        final_vif_table, VIF_eliminated_predictors, VIF_remaining_predictors = vif_eliminator(init_vif_table=init_vifs, 
                                                                                              vif_threshold=vif_threshold, 
                                                                                              input_data=input_data, 
                                                                                              data_path=data_path, 
                                                                                              weight_variable_name=weight_variable_name)
        
        print()
        print(ufun.color.PURPLE + 'Variables eliminated from VIF: ' + ufun.color.END + str(VIF_eliminated_predictors))
        print(ufun.color.PURPLE + 'Number of variables eliminated from VIF: ' + ufun.color.END + str(len(VIF_eliminated_predictors)))
        print(ufun.color.PURPLE + 'Variables kept after VIF: ' + ufun.color.END + str(VIF_remaining_predictors))
        print(ufun.color.PURPLE + 'Number of variables kept after VIF: ' + ufun.color.END + str(len(VIF_remaining_predictors)))
        return VIF_eliminated_predictors, VIF_remaining_predictors
    else: 
        print("Choose either 'correlation' or 'VIF' as your feature elimination method")

def weighted_variance_inflation_factor(
    x, 
    feature_index, 
    weight_vector
    ):
    
    target_vector = x[:, feature_index]
    feat_df = np.delete(x, feature_index, axis=1)
    r_squared = WLS(target_vector, feat_df, weight_vector).fit().rsquared
    vif = 1 / (1-r_squared)
    return vif

def calculate_vifs(
    input_data, 
    features, 
    weight_variable_name, 
    silent=False
    ):
    
    vifs_dict = dict()
    
    if len(features) <= 1:
        print("<=1 Remaining features, VIF cannot be calculated")
        return pd.DataFrame(data=[0], columns=['Variance Inflation Factor'], index=features)
        
    weight_vector = input_data[weight_variable_name].values 
    X = add_constant(input_data[features].values)
    assert(X[:, 0].std() == 0)
    
    for i in range(len(features)):
        vifs_dict[features[i]] = weighted_variance_inflation_factor(X, i+1, weight_vector)
    init_vifs = pd.DataFrame(vifs_dict, index=['Variance Inflation Factor']).T.sort_values('Variance Inflation Factor', ascending=False)
    
    if not silent:
        display(init_vifs)
        
    return init_vifs

@time_function
def vif_eliminator(
    init_vif_table, 
    vif_threshold, 
    input_data, 
    data_path, 
    weight_variable_name
    ):
        
    remaining_predictors = []
    eliminated_predictors = []
    vifs = init_vif_table 
    
    while vifs['Variance Inflation Factor'].max() > vif_threshold:
        start_time = time.time()
        remove_index = vifs.idxmax().values[0]
        eliminated_predictors.append(remove_index)
        print('Dropped {}'.format(remove_index))
        
        temp_remaining_predictors = [x for x in vifs.index if x not in eliminated_predictors]
        vifs = calculate_vifs(input_data, temp_remaining_predictors, weight_variable_name, silent=True)
        print('This step of VIF feature elimination took %.2fs. to run'%(time.time()-start_time))
        
    print()
    print('Final VIF table')
    display(vifs)
    remaining_predictors = list(vifs.index)
    final_vif_table = pd.merge(init_vif_table, vifs, how='left', left_index=True, right_index=True, suffixes=(' Initial', ' Final'))
    final_vif_table.to_csv(data_path + '/output/VIF_results_pre_post.csv')
    
    return vifs, eliminated_predictors, remaining_predictors

@time_function
def run_VIF(
    VIF_reduction, 
    features, 
    input_data, 
    data_path, 
    vif_threshold, 
    corr_threshold, 
    weight_variable_name
    ):
    
    if VIF_reduction == True: 
        print('Initial VIF table')
        init_vifs = calculate_vifs(input_data=input_data, 
                                    features=features, 
                                    weight_variable_name=weight_variable_name
                                    )
        eliminated, remaining_predictors = correlation_elimination('VIF', 
                                                                    features=features, 
                                                                    input_data=input_data, 
                                                                    data_path=data_path, 
                                                                    vif_threshold=vif_threshold, 
                                                                    corr_threshold=corr_threshold, 
                                                                    top_n=0,
                                                                    weight_variable_name=weight_variable_name, 
                                                                    init_vifs=init_vifs
                                                                    )
        return eliminated, remaining_predictors
    else: 
        print("VIF was not run. If you wish to execute VIF, change the VIF_reduction boolean flag to 'True'")
        return [], features

bic_dict_ = {}
def perform_lasso(
    sample_values_dict, 
    sample_values_solution, 
    data, 
    target_variable_name, 
    predictor_variables, 
    data_path, 
    LogisticRegression_solver,
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
        
        print()
        print(ufun.color.PURPLE + "Displaying evaluation metrics on the " + str(list(sample_values_dict.keys())[i]) + " dataset" + ufun.color.END )
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
            solver= LogisticRegression_solver,
            data_path=data_path            
            )
            
        bic_df = lasso.fit()
        bic_dict_[list(sample_values_dict.keys())[i]] = bic_df
    return bic_dict_













