import time
import pandas as pd
import numpy as np
from statsmodels.tools.tools import add_constant 
from statsmodels.regression.linear_model import WLS
from decorators import time_function 

class color:
    PURPLE = '\u001b[0;35m'
    CYAN = '\u001b[36;1m'
    DARKCYAN = '\\u001b[46m'
    BLUE = '\u001b[34m'
    GREEN = '\u001b[32m'
    YELLOW = '\u001b[33m'
    RED = '\u001b[31m'
    BOLD = '\u001b[4m'
    UNDERLINE = '\u001b[31;1;4m'
    END = '\u001b[0m'
    BLACK = '\u001b[30m'
    
#COPIED
def weighted_mean(
    feat_1, 
    weight
    ):
    return np.sum(feat_1 * weight) / np.sum(weight)
    
#COPIED
def weighted_cov(
    feat_1, 
    feat_2, 
    weight
    ):
    return np.sum(weight * (feat_1 - weighted_mean(feat_1, weight)) * (feat_2 - weighted_mean(feat_2, weight))) / np.sum(weight)
    
#COPIED
def weighted_corr(
    feat_1, 
    feat_2, 
    weight
    ):
    return weighted_cov(feat_1, feat_2, weight) / np.sqrt(weighted_cov(feat_1, feat_1, weight) * weighted_cov(feat_2, feat_2, weight))
    
#COPIED
@time_function    
def calculate_correlations(
    train_df, 
    features, 
    corr_threshold, 
    weight_variable_name
    ):
    
    correlations = []
    weight = train_df[weight_variable_name]
    feat_df = train_df[features]
    
    for i in range(len(feat_df.columns)): 
        for j in range(i):
            feat_1_name = feat_df.columns[i]
            feat_2_name = feat_df.columns[j]
            feat_1 = feat_df[feat_1_name]
            feat_2 = feat_df[feat_2_name]
            
            correlation = weighted_corr(feat_1, feat_2, weight)
            correlations.append({'feature_1': feat_1_name, 'feature_2': feat_2_name, 'correlation': correlation})
            
    corr_df = pd.DataFrame(correlations, columns=['feature_1', 'feature_2', 'correlation'])
    corr_df['correlation_abs'] = np.abs(corr_df['correlation'])
    corr_df = corr_df.sort_values(by='correlation_abs', ascending=False).reset_index(drop=True)
    
    corr_to_drop_df = corr_df.loc[corr_df['correlation_abs'] > corr_threshold].drop(['correlation_abs'], axis=1)
    print('The following columns have a correlation above your threshold')
    display(corr_to_drop_df)
    
    return corr_df
    
    
#COPIED
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
    print(color.PURPLE + 'Variables eliminated from correlation: ' + color.END + str(corr_elilminated_predictors))
    print(color.PURPLE + 'Number of variables eliminated from correlation: ' + color.END + str(len(corr_elilminated_predictors)))
    print(color.PURPLE + 'Keeping the following variables (Correlation < ' + str(corr_threshold) + '): ' + color.END + str(list(keep_num_vars_corr)))
    print(color.PURPLE + 'Number of variables kept from correlation: ' + color.END + str(len(keep_num_vars_corr)))
    print('Here are the top ' + str(top_n) + ' features with the highest correlations after removing highly correlated features')
    
    display(corr_non_drop_df.head(top_n))
    corr_non_drop_df.to_csv(data_path + '/output/correlation_results.csv')
    return corr_elilminated_predictors, keep_num_vars_corr
    
#COPIED
@time_function
def correlation_elimination(
    method, 
    features, 
    train_df, 
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
                                                                                              train_df=train_df, 
                                                                                              data_path=data_path, 
                                                                                              weight_variable_name=weight_variable_name)
        
        print()
        print(color.PURPLE + 'Variables eliminated from VIF: ' + color.END + str(VIF_eliminated_predictors))
        print(color.PURPLE + 'Number of variables eliminated from VIF: ' + color.END + str(len(VIF_eliminated_predictors)))
        print(color.PURPLE + 'Variables kept after VIF: ' + color.END + str(VIF_remaining_predictors))
        print(color.PURPLE + 'Number of variables kept after VIF: ' + color.END + str(len(VIF_remaining_predictors)))
        return VIF_eliminated_predictors, VIF_remaining_predictors
    else: 
        print("Choose either 'correlation' or 'VIF' as your feature elimination method")

#COPIED
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
    
#COPIED
def calculate_vifs(
    train_df, 
    features, 
    weight_variable_name, 
    silent=False
    ):
    
    vifs_dict = dict()
    
    if len(features) <= 1:
        print("<=1 Remaining features, VIF cannot be calculated")
        return pd.DataFrame(data=[0], columns=['Variance Inflation Factor'], index=features)
        
    weight_vector = train_df[weight_variable_name].values 
    X = add_constant(train_df[features].values)
    assert(X[:, 0].std() == 0)
    
    for i in range(len(features)):
        vifs_dict[features[i]] = weighted_variance_inflation_factor(X, i+1, weight_vector)
    init_vifs = pd.DataFrame(vifs_dict, index=['Variance Inflation Factor']).T.sort_values('Variance Inflation Factor', ascending=False)
    
    if not silent:
        display(init_vifs)
        
    return init_vifs
    
#COPIED
@time_function
def vif_eliminator(
    init_vif_table, 
    vif_threshold, 
    train_df, 
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
        vifs = calculate_vifs(train_df, temp_remaining_predictors, weight_variable_name, silent=True)
        print('This step of VIF feature elimination took %.2fs. to run'%(time.time()-start_time))
        
    print()
    print('Final VIF table')
    display(vifs)
    remaining_predictors = list(vifs.index)
    final_vif_table = pd.merge(init_vif_table, vifs, how='left', left_index=True, right_index=True, suffixes=(' Initial', ' Final'))
    final_vif_table.to_csv(data_path + '/output/VIF_results_pre_post.csv')
    
    return vifs, eliminated_predictors, remaining_predictors

#COPIED
@time_function
def run_VIF(
    VIF_reduction, 
    features, 
    train_df, 
    data_path, 
    vif_threshold, 
    corr_threshold, 
    weight_variable_name
    ):
    
    if VIF_reduction == True: 
        print('Initial VIF table')
        init_vifs = calculate_vifs(train_df=train_df, 
                                    features=features, 
                                    weight_variable_name=weight_variable_name
                                    )
        eliminated, remaining_predictors = correlation_elimination('VIF', 
                                                                    features=features, 
                                                                    train_df=train_df, 
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








