import pandas as pd
import numpy as np
import time
from statsmodels.tools.tools import add_constant 
from sklearn.metrics import roc_auc_score
from statsmodels.regression.linear_model import WLS
import lasso_feature_selection as lfs

#from evaluation import gini
#from evaluation import gini_weight
import os 
import sys
if os.path.basename(os.path.dirname(sys.executable)) == 'Supervised_Modeling_ML': 
    from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
#from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import roc_auc_score

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
    random_state=42, 
    lasso_criterion='BIC'
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
            early_stop=early_stop,
            weight_variable_name=weight_variable_name, 
            c_min=c_min,
            c_max=c_max, 
            num=num, 
            vif_threshold=vif_threshold, 
            random_state=random_state, 
            solver= LogisticRegression_solver,
            data_path=data_path, 
            lasso_criterion=lasso_criterion
            )
            
        bic_df = lasso.fit()
        bic_dict_[list(sample_values_dict.keys())[i]] = bic_df
    return bic_dict_

if os.path.basename(os.path.dirname(sys.executable)) == 'Supervised_Modeling_ML': 
    class SelectBest_weight(object):
        def __init__(self, df, target, weight):
                        
                        '''
                        expect input as following:
                                        df: array_like 
                                                        dataframe used for gini calculation
                                        target: string
                                                        target variable
                                        weight: string
                                                        weight variable
                        '''
                        
                        self.df = df
                        self.target = target
                        self.weight = weight
                        
        @time_function
        def best_univar_gini(self, feats, n=1):
                        '''
                        input:
                                        class initial input df, feats, target, top 
                        output: list
                                        top features with descending absolute gini value.  
                        '''    
                        feat_gini = dict()
                        for x in feats:
    #                        feat_gini[x] = gini_weight(self.df[[x, self.target, self.weight]].values)
                            feat_gini[x] = abs(2*roc_auc_score(self.df[self.target].values, self.df[x].values, sample_weight=self.df[self.weight].values)-1)
                        rank = sorted(feat_gini.items(), key=lambda t: abs(t[1]), reverse=True)
                        return rank, [x[0] for x in rank][:n]
        
        @time_function
        def top_rf_feat(self, feats, model=RandomForestClassifier(n_estimators=200, max_depth=5,random_state=1234), n=1):
                        '''
                        input:
                                        1. class initial input df, feats, target, top 
                                        2. model: 
                                                                        random forest model with pre defined hyperparameters.     
                                                                        
                        output: list
                                        top features with descending variable importance in random forest model.  
                        '''   
                        model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                        feat_importance = dict(zip(feats, model.feature_importances_)) 
                        rank = sorted(feat_importance.items(), key=lambda t: abs(t[1]), reverse=True)
                        return rank, [x[0] for x in rank][:n]
                        
        @time_function
        def top_gbm_feat(self, feats, model=GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=1234), n=1):
                        '''
                        input:
                                        1. class initial input df, feats, target, top 
                                        2. model: 
                                                                        GBM model with pre defined hyperparameters.     
                                                                        
                        output: list
                                        top features with descending variable importance in GBM model.  
                        '''   
                        model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                        feat_importance = dict(zip(feats, model.feature_importances_))  
                        rank = sorted(feat_importance.items(), key=lambda t: abs(t[1]), reverse=True)
                        return rank, [x[0] for x in rank][:n]
        
        @time_function
        def top_lgbm_feat(self, feats, model=LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=1234, n_jobs=6), n=1):
                        '''
                        input:
                                        1. class initial input df, feats, target, top 
                                        2. model: 
                                                                        lightGBM model with pre defined hyperparameters.     
                                                                        
                        output: list
                                        top features with descending variable importance in lightGBM model.  
                        '''   
                        model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                        feat_importance = dict(zip(feats, model.feature_importances_))  
                        rank = sorted(feat_importance.items(), key=lambda t: abs(t[1]), reverse=True)
                        return rank, [x[0] for x in rank][:n]
        
        @time_function
        def top_svc_feat(self, feats, model=LinearSVC(C=0.01, penalty="l1", dual=False,random_state=42), n=1):
                        '''
                        input:
                                        1. class initial input dev, feats, target, top 
                           
                                        2. model: 
                                                        support vector classification with pre defined hyperparameters. 
                                                        
                        output: list
                                        top features with descending absolute coefficient in SVC model.  
                        '''    
                        model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                        feat_coef = dict()
                        fc=model.coef_[0]
        
                        for i,x in enumerate(feats):
                                        feat_coef[x] =fc[i]
                                        
                        rank = sorted(feat_coef.items(), key=lambda t: abs(t[1]), reverse=True)
                        return rank, [x[0] for x in rank][:n]
        
        @time_function
        def top_lr_feat(self, feats, model=LogisticRegression(C=0.01, penalty="l1",random_state=42), n=1):
                        '''
                        input:
                                        1. class initial input dev, feats, target, top 
                                        2. model: 
                                                        logistic regression model with pre defined hyperparameters.      
                                                        
                        output: list
                                        top features with descending absolute coefficient in logistic regression model.  
                        '''
                        
                        model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                        feat_coef = dict()
                        fc = model.coef_[0]
                        
                        for i,x in enumerate(feats):
                                        feat_coef[x] =fc[i]
                                        
                        rank = sorted(feat_coef.items(), key=lambda t: abs(t[1]), reverse=True)
                        return rank, [x[0] for x in rank][:n]
        
        @time_function
        def get_best(self, remaining_feats,oos, model, classification):
                        best_feat, best_gini = " ", 0
                        for v in remaining_feats:
                                        left = remaining_feats[:]
                                        left.remove(v)
                                        model.fit(self.df[left], self.df[self.target], sample_weight=self.df[self.weight])
                                        if classification==True:
                                                        oos.loc[:, 'score']=model.predict_proba(oos[left])[:, 1]
                                        else:
                                                        oos.loc[:, 'score']=model.predict(oos[left])
    #                                    gini_v = gini_weight(oos[['score', self.target, self.weight]].values)
                                        gini_v = abs(2*roc_auc_score(oos[self.target].values, oos['score'].values, sample_weight=oos[self.weight].values)-1)
                                        if gini_v > best_gini:
                                                        best_gini = gini_v
                                                        best_feat = v
                        return best_feat, best_gini

        @time_function
        def backward_recur(self, feats, oos, model, min_feats=5, classification=True):
                        '''
                        input:
                                        1. class initial input dev, feats, target, top 
                                        
                                        2. oos: array_like
                                                                        cross validation dataset
                                        3. model: 
                                                                        model used for backward selection. eg. logistic regression or random forest
                                        4. min_feats: int
                                                                        minimum number of features to keep
                                        5. classification: Boolean (True or False)
                                                                        if a model is a classification model or not.
                                                                        
                        output: list
                                        remaining features after backward selection. 
                        '''
                        keep = feats[:]
                        best_gini =  0
                        
                        for i in range(len(feats)-min_feats):
                                        remove_feat, gini_i = self.get_best(keep, oos, model,classification)
                                        
                                        if (gini_i <= best_gini) or (len(keep)<=2):
                                                        return keep
                                        else:
                                                        print('step i =', i+1, 'feature removed:', remove_feat, 'gini:',gini_i)
                                                        keep.remove(remove_feat)
                        return keep











