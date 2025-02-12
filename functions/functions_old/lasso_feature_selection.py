import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, make_scorer, log_loss, roc_auc_score
from statsmodels.tools.tools import add_constant 
from statsmodels.stats.outliers_influence import variance_inflation_factor
import variable_reduction as vr
from decorators import time_function 

# COPIED
class lasso_selection():
    
    def __init__(
        self, curr_dataset, train, valid, data, target_variable_name, predictor_variables, data_path, standardization=True, early_stop=True, weight_variable_name=None, c_min=1e-4, c_max=1e4, num=20, vif_threshold=5, random_state=42
        ):
        
        self.predictor_variables = predictor_variables
        self.train_df = train
        self.validation_df = valid 
        self.early_stop = early_stop
        self.data_path = data_path
        self.weights = weight_variable_name 
        self.vif_threshold = vif_threshold
        self.vifs_dict = dict()
        self.standardization = standardization
        self.vifs = None
        
        if self.standardization == True:
            self.X_scaler = StandardScaler().fit(self.train_df[self.predictor_variables].values)
            self.X_train = self.X_scaler.transform(self.train_df[self.predictor_variables].values)
            self.X_validation = self.X_scaler.transform(self.validation_df[self.predictor_variables].values)
        else: 
            self.X_train = self.train_df[self.predictor_variables].values
            self.X_validation = self.validation_df[self.predictor_variables].values
        
        self.weights_train = self.train_df[self.weights].values
        self.y_train = self.train_df[target_variable_name].values.astype(int)
        self.y_validation = self.validation_df[target_variable_name].values.astype(int)
        self.weights_validation = self.validation_df[self.weights].values
        self.cs = np.linspace(c_min, c_max, num=num)
        self.random_state = random_state
        self.bic_dict = {"C": [], 
                        "BIC": [],
                        "AIC": [], 
                        "AUC": [],
                        "Log Loss": [],
                        "Remaining_Features": []
                        }
        self.lrs = []
        self.bic_df = None 
        self.curr_dataset = curr_dataset
    
    def bic(
        self, 
        lr, 
        LL
        ):
    
        n = self.X_validation.shape[0]
        k = (lr.coef_ != 0).sum()
        if lr.intercept_[0] != 0:
            k += 1
        return k*np.log(n) - 2*LL
    
    def aic(
        self, 
        lr, 
        LL
        ):
        
        k = (lr.coef_ != 0).sum()
        if lr.intercept_[0] != 0:
            k += 1
        return 2*k - 2*LL
    
    def auc(
        self, 
        lr
        ):
    
        yhat = lr.predict_proba(self.X_validation)[:, 1]

        if self.weights == 'None':
            self.weights = None
        
        if self.weights == None:
            return roc_auc_score(self.y_validation, yhat)
        else: 
            return roc_auc_score(self.y_validation, yhat, sample_weight=self.weights_validation)
        
    def log_loss(
        self, 
        lr
        ):
    
        yhat = lr.predict_proba(self.X_validation)[:, 1]
        
        if self.weights == 'None':
            self.weights = None

        if self.weights == None:
            LL = -log_loss(self.y_validation, yhat, normalize=False)
        else: 
            LL = -log_loss(self.y_validation, yhat, normalize=False, sample_weight=self.weights_validation)
        return LL
    
    @time_function
    def fit(
        self
        ):
    
        count = 0
        for i in range(len(self.cs)):
            C = self.cs[i]
            print("{0}/{1} models trained".format(count, len(self.cs)))
            lr = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=self.random_state)
            
            if self.weights == 'None':
                self.weights = None
                
            if self.weights == None: 
                lr = lr.fit(X=self.X_train, y=self.y_train)
            else: 
                assert(self.weights != None)
                lr = lr.fit(X=self.X_train, y=self.y_train, sample_weight=self.weights_train)
                
            self.bic_dict["C"].append(C)
            Lg_loss = self.log_loss(lr)
            self.bic_dict["BIC"].append(self.bic(lr, Lg_loss))
            self.bic_dict["AIC"].append(self.aic(lr, Lg_loss))
            self.bic_dict["Log Loss"].append(Lg_loss)
            self.bic_dict["Remaining_Features"].append((lr.coef_ != 0).sum())
            self.bic_dict["AUC"].append(self.auc(lr))
            self.lrs.append((C, lr))
            count += 1
            
            if self.early_stop and i >= 3:
                if self.bic_dict["BIC"][i-3] < min(self.bic_dict["BIC"][i-2:]): 
                    break 
                    
        print("{0}/{1} models trained".format(count, len(self.cs)))
        
        self.bic_df = pd.DataFrame(self.bic_dict)
        self.bic_df.to_csv(self.data_path + '/output/BIC_AIC_scores_' + self.curr_dataset + '.csv', index=False)
        display(self.bic_df)
        
        fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
        fig.suptitle('BIC/AIC And Number of Variables for Different L1 Penalties', fontsize=20)
        axs[0].plot(self.bic_dict['C'], self.bic_dict['BIC'], "b-", label="BIC")
        axs[0].plot(self.bic_dict['C'], self.bic_dict['AIC'], "r-", label="AIC")
        axs[0].legend(fontsize=16)
        axs[1].plot(self.bic_dict['C'], self.bic_dict['Remaining_Features'], label='Remaining Features')
        axs[1].legend(fontsize=16)
        axs[0].set_ylabel("BIC/AIC Score", fontsize=16)
        axs[1].set_ylabel("Remaining Features", fontsize=16)
        axs[1].set_xlabel("C (L1 penalty term)", fontsize=16)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        return self
    
    def bic_scores(
        self
        ):
    
        return self.bic_df
    
    def get_model(
        self, 
        C=None
        ):
    
        if C == None:
            return self.lrs 
        else: 
            for tup in self.lrs:
                if tup[0] == C: 
                    return tup[1]
            print("Value of C not found, returning all Lasso models")
            return self.lrs
        
    def get_min_C(
        self, 
        criterion
        ):
    
        min_C = self.bic_df.loc[self.bic_df[criterion] == self.bic_df[criterion].min(), 'C'].values[0]
        min_C_index = np.where(self.cs == min_C)[0][0]
        return min_C, min_C_index
    
    def best_vars(
        self, 
        criterion
        ):
        
        self.min_C, self.min_C_index = self.get_min_C(criterion)
        self.criterion = criterion
        for tup in self.lrs:
            if tup[0] == self.min_C:
                coefs = tup[1].coef_
                
        print("Best C value found via gridsearch was {0}".format(self.min_C))
        print("Remaining features are: {}".format(np.array(self.predictor_variables)[(coefs != 0).flatten()]))
        print("Eliminated features are: {}".format(np.array(self.predictor_variables)[(coefs == 0).flatten()]))
        self.lasso_features = np.array(self.predictor_variables)[(coefs != 0).flatten()]
#        self.lasso_X = StandardScaler().fit_transform(self.train_df[self.lasso_features].values)
        return self.lasso_features
    
    def calculate_vifs(
        self, 
        features, 
        weight_variable_name, 
        silent=False
        ):
    
        self.vifs_dict = dict()
        if len(features) <= 1:
            print("<=1 Remaining Features, VIF cannot be calculated")
            self.vifs = pd.DataFrame(data=[0], columns=['Variance Inflation Factor'], index=features)
            self.vifs.to_csv('{0}/output/variance_inflation_factor_{1}.csv'.format(self.data_path, self.criterion, index_label='Variable'))
            return self.vifs
            
        weight_vector = self.train_df[weight_variable_name].values 
        if self.standardization == True:
            X = add_constant(StandardScaler().fit_transform(self.train_df[features].values))
        else: 
            X = add_constant(self.train_df[features].values)

        assert(X[:, 0].std() == 0)
        for i in range(len(features)):
            self.vifs_dict[features[i]] = vr.weighted_variance_inflation_factor(X, i+1, weight_vector)
        self.vifs = pd.DataFrame(self.vifs_dict, index=['Variance Inflation Factor']).T.sort_values('Variance Inflation Factor', ascending=False)
        if not silent:
            display(self.vifs)
            self.vifs.to_csv('{0}/output/variance_inflation_factor_{1}.csv'.format(self.data_path, self.criterion, index_label='Variable'))
            
        return self.vifs

    def remaining_predictors(
        self
        ):

        if len(self.lasso_features) <= 1:
            print("Skipping VIF elimination, too few remaining features to calculate VIF")
            self.final_predictors = np.array(self.lasso_features)
        else: 
            remaining_predictors = []
            eliminated_predictors = []
            while self.vifs['Variance Inflation Factor'].max() > self.vif_threshold:
                max_vif = self.vifs['Variance Inflation Factor'].max()
                remove_index = self.vifs.index[self.vifs['Variance Inflation Factor'] == self.vifs['Variance Inflation Factor'].max()][0]
                eliminated_predictors.append(remove_index)
                temp_list = list(self.vifs.index)
                temp_list.remove(remove_index)
                temp_remaining_predictors = temp_list
                self.calculate_vifs(features=temp_remaining_predictors, weight_variable_name=self.weights)
            remaining_predictors = list(self.vifs.index)
            
            print("Eliminated features: {}".format(eliminated_predictors))
            print("Remaining features: {}".format(remaining_predictors))
            print("Number of remaining features: {}".format(len(remaining_predictors)))
        
            self.final_predictors = np.array(remaining_predictors)
            
        pd.DataFrame(self.final_predictors, columns=['final_features']).to_csv('{0}/output/final_features_{1}.csv'.format(self.data_path, self.criterion), index=False)
        return self.final_predictors
            


