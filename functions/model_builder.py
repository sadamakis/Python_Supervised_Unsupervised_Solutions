"""
# This is a set of machine learning tools developed using Python
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from sklearn.externals import joblib
import joblib
from sklearn.metrics import log_loss

import numpy as np
import pandas as pd

from sklearn.metrics import  make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras import optimizers
from keras.constraints import maxnorm
from keras.optimizers import Adam, Nadam, RMSprop, SGD, Adadelta, Adamax
from keras.losses import logcosh, binary_crossentropy
from keras.activations import relu, elu

import keras_functions as ks_fn
import optuna
from decorators import time_function

                                
def fit_model(df, feats, target, model, model_name):
    model.fit(df[feats].values, df[target].values)
    joblib.dump(model, model_name)
    return model

@time_function
def fit_model_weight(df, feats, target, weight, model, model_name):
    model.fit(df[feats].values, df[target].values, sample_weight=df[weight].values)
    joblib.dump(model, model_name)
    return model

                
@time_function
def feature_imp(model, feats, data_path, feature_imp_file):
    feature_importance_df = pd.DataFrame({
        'Feature': feats,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv('{0}/output/{1}'.format(data_path, feature_imp_file), header=True, index=False)
    return feature_importance_df                
                

def feature_imp_rank(model, feats):
                imp = dict(zip(feats, model.feature_importances_))
                return sorted(imp.items(), key=lambda t: t[1], reverse=True)

def logloss(y_true, y_pred, eps=1e-15):
    tot = 0
    for i in range(len(y_true)): 
        tot += y_true[i]*np.log(y_pred[i] + eps) + (1-y_true[i])*np.log(1-y_pred[i] + eps)
    return -tot/len(y_true)

def logloss_weight(y_true, y_pred, weight_variable, eps=1e-15):
    tot = 0
    for i in range(len(y_true)): 
        tot += (y_true[i]*np.log(y_pred[i] + eps) + (1-y_true[i])*np.log(1-y_pred[i] + eps))*weight_variable[i]
    return -tot/weight_variable.sum()

def step_search(estimator, params, grid, target, dev, val, keep):
    loss = 10e10
    print("Search Progress:")
    for key, values in grid.items():
        for v in values:
            curr_params = {k:v for k, v in params.items()}
            curr_params[key] = v
            model = estimator(**curr_params)
            model.fit(dev[keep].values, dev[target].values)
            val['score'] = model.predict_proba(val[keep].values)[:, 1]
            curr_loss = logloss(val[target].values, val['score'].values)
            if curr_loss < loss:
                loss = curr_loss
                params[key] = v
            print(params, key, v, curr_loss, loss)
        print('')
    return params, loss

@time_function
def step_search_weight(estimator, params, grid, target, weight, dev, val, keep):
    print("Search Progress:")
    
    if type(grid) is dict:
        loss = 10e10

        for key, values in grid.items():
            for v in values:

                curr_params = {k:v for k, v in params.items()}
                curr_params[key] = v

                if estimator==KerasClassifier:
                    keras_function = ks_fn.neural_network_function_wrapper(
                        num_neurons_=curr_params['num_neurons']
                        , num_hidden_layers_=curr_params['num_hidden_layers']
                        , input_dim_=curr_params['input_dim']
                        , kernel_initializer_=curr_params['kernel_initializer']
                        , activation_=curr_params['activation']
                        , kernel_constraint_=curr_params['kernel_constraint']
                        , dropout_rate_=curr_params['dropout_rate']
                        , output_kernel_initializer_=curr_params['output_kernel_initializer']
                        , output_activation_=curr_params['output_activation']
                        , loss_=curr_params['loss']
                        , optimizer_=curr_params['optimizer']
                        , learning_rate_=curr_params['learning_rate']
                        , momentum_=curr_params['momentum']
                        , rho_=curr_params['rho']
                        , beta_1_=curr_params['beta_1']
                        , beta_2_=curr_params['beta_2']
                        , return_metrics_=curr_params['return_metrics']
                        )
                    model = estimator(build_fn=keras_function, verbose=0)
                    model.fit(dev[keep].values, dev[target].values, sample_weight=dev[weight].values, use_multiprocessing=True, workers=8, verbose=0,
                          epochs=curr_params['epochs'], batch_size=curr_params['batch_size'], validation_data=(val[keep].values, val[target]), shuffle=False)

                else:
                    model = estimator(**curr_params)
                    model.fit(dev[keep].values, dev[target].values, sample_weight=dev[weight].values)

                val.loc[:, 'score'] = model.predict_proba(val[keep].values)[:, 1]

                if val['score'].mean()!=np.nan:
                    curr_loss = logloss_weight(y_true=val[target].values, y_pred=val['score'].values, weight_variable=val[weight].values, eps=1e-15)
                else:
                    curr_loss = 1e+10

                if curr_loss < loss:
                    loss = curr_loss
                    params[key] = v
                print(params, key, v, curr_loss, loss)
            print('')
            
        return params, loss

            
            
    elif type(grid) is list:
        
        params_list = []
        loss_list = []
        
        for i in range(len(grid)):
            loss = 10e10
            grid_ = grid[i]
            params_ = params[i]

            for key, values in grid_.items():
                for v in values:

                    curr_params = {k:v for k, v in params_.items()}
                    curr_params[key] = v

                    if estimator==KerasClassifier:
                        keras_function = ks_fn.neural_network_function_wrapper(
                            num_neurons_=curr_params['num_neurons']
                            , num_hidden_layers_=curr_params['num_hidden_layers']
                            , input_dim_=curr_params['input_dim']
                            , kernel_initializer_=curr_params['kernel_initializer']
                            , activation_=curr_params['activation']
                            , kernel_constraint_=curr_params['kernel_constraint']
                            , dropout_rate_=curr_params['dropout_rate']
                            , output_kernel_initializer_=curr_params['output_kernel_initializer']
                            , output_activation_=curr_params['output_activation']
                            , loss_=curr_params['loss']
                            , optimizer_=curr_params['optimizer']
                            , learning_rate_=curr_params['learning_rate']
                            , momentum_=curr_params['momentum']
                            , rho_=curr_params['rho']
                            , beta_1_=curr_params['beta_1']
                            , beta_2_=curr_params['beta_2']
                            , return_metrics_=curr_params['return_metrics']
                            )
                        model = estimator(build_fn=keras_function, verbose=0)
                        model.fit(dev[keep].values, dev[target].values, sample_weight=dev[weight].values, use_multiprocessing=True, workers=8, verbose=0,
                              epochs=curr_params['epochs'], batch_size=curr_params['batch_size'], validation_data=(val[keep].values, val[target]), shuffle=False)

                    else:
                        model = estimator(**curr_params)
                        model.fit(dev[keep].values, dev[target].values, sample_weight=dev[weight].values)

                    val['score'] = model.predict_proba(val[keep].values)[:, 1]

                    if val['score'].mean()!=np.nan:
                        curr_loss = logloss_weight(y_true=val[target].values, y_pred=val['score'].values, weight_variable=val[weight].values, eps=1e-15)
                    else:
                        curr_loss = 1e+10

                    if curr_loss < loss:
                        loss = curr_loss
                        params_[key] = v
                    print(params_, key, v, curr_loss, loss)
                print('')
                
            params_list.append(params_)
            loss_list.append(loss)
            
        return params_list, loss_list

#def score_f(y_true, y_pred, sample_weight):
#      return log_loss(y_true.values, y_pred,
#                      sample_weight=sample_weight.loc[y_true.index.values].values.reshape(-1),
#                      normalize=True)

def score_f(y_true, y_pred, sample_weight):
      return logloss_weight(y_true.values, y_pred,
                      weight_variable=sample_weight.loc[y_true.index.values].values.reshape(-1))

@time_function
def grid_search_cv(n_splits, classifier, keras_function,
                   grid_params, dev_df, feats, target, weight_variable, randomized_search, n_random_grids, random_state, n_jobs):
#    n_splits: Number of cross-validation splits
#    classifier: Classifier name, e.g. RandomForestClassifier
#    keras_function: Define Keras function. If Keras is not used, then leave this parameter blank
#    grid_params: Grid space
#    dev_df: Development sample that this will analysis will be performed
#    feats: List of predictor names
#    target: Target variable name
#    weight_variable: Weight variable name
#    randomized_search: Set to True if randomized grid search will be performed, or to False if exhaustive grid search will be performed
#    n_random_grids: Number of grid searches when randomized_search=True. If randomized_search=False, then this parameter is not applicable
#    random_state: If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
#    n_jobs: Number of jobs to run in parallel
    
    inner_cv = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state)

    if classifier==GradientBoostingClassifier:
        rfc = classifier(random_state=random_state)
    elif classifier==KerasClassifier:
#        callback = EarlyStopping(monitor="val_auc", patience=50, verbose=0, mode='max')
        rfc = classifier(build_fn=keras_function, verbose=0)
    else:
        rfc = classifier(n_jobs=n_jobs, random_state=random_state)

    search_params = grid_params

    score_params = {"sample_weight": dev_df[weight_variable]}

    my_scorer = make_scorer(score_f,
                              greater_is_better=False, 
                              needs_proba=True, 
                              needs_threshold=False,
                              **score_params)

    if randomized_search==False:
        if classifier==GradientBoostingClassifier: 
            grid_clf = GridSearchCV(estimator=rfc,
                                      scoring=my_scorer,
                                      cv=inner_cv,
                                      param_grid=search_params,
                                      refit=True,
                                      return_train_score=False)
        else:
            grid_clf = GridSearchCV(estimator=rfc,
                                      scoring=my_scorer,
                                      cv=inner_cv,
                                      param_grid=search_params,
                                      refit=True,
                                      return_train_score=False, 
                                       n_jobs=n_jobs)
    elif randomized_search==True:
        if classifier==GradientBoostingClassifier: 
            grid_clf = RandomizedSearchCV(estimator=rfc,
                                      scoring=my_scorer,
                                      cv=inner_cv,
                                      param_distributions=search_params,
                                      refit=True,
                                      return_train_score=False, 
                                        n_iter=n_random_grids)
        else:
            grid_clf = RandomizedSearchCV(estimator=rfc,
                                      scoring=my_scorer,
                                      cv=inner_cv,
                                      param_distributions=search_params,
                                      refit=True,
                                      return_train_score=False, 
                                       n_jobs=n_jobs, 
                                        n_iter=n_random_grids)

            
    if classifier==KerasClassifier: 
#        return grid_clf.fit(dev_df[feats].values, dev_df[target], sample_weight=dev_df[weight_variable].values, validation_data=(oos[feats].values, oos[target], oos[weight_variable]), use_multiprocessing=True, workers=8)
        return grid_clf.fit(dev_df[feats].values, dev_df[target], sample_weight=dev_df[weight_variable].values, use_multiprocessing=True, workers=8)
    else: 
        return grid_clf.fit(dev_df[feats].values, dev_df[target], sample_weight=dev_df[weight_variable].values)

class grid_search_optuna():
    def __init__(
        self, 
        classifier, 
        grid_params, 
        dev_df, 
        feats, 
        target, 
        weight_variable, 
        random_state, 
        n_splits_kfold, 
        n_repeats_kfold, 
        n_random_grids, 
        timeout, 
        n_jobs
    ):

    #    classifier: Classifier name, e.g. RandomForestClassifier
    #    grid_params: Grid space
    #    dev_df: Development sample that this will analysis will be performed
    #    feats: List of predictor names
    #    target: Target variable name
    #    weight_variable: Weight variable name
    #    random_state: If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    #    n_splits_kfold: Number of cross-validation splits
    #    n_repeats_kfold: Number that KFold cross-validation will be repeated
    #    n_random_grids: Number of grid searches from Optuna. The more searches, the longer that the algorithm will take to complete. 
    #    timeout: Time in seconds that Optuna will stop
    #    n_jobs: Number of jobs to run in parallel

        self.classifier = classifier
        self.grid_params = grid_params
        self.dev_df = dev_df
        self.feats = feats
        self.target = target
        self.weight_variable = weight_variable
        self.random_state = random_state
        self.n_splits_kfold = n_splits_kfold
        self.n_repeats_kfold = n_repeats_kfold
        self.n_random_grids = n_random_grids
        self.timeout = timeout
        self.n_jobs = n_jobs

        self.X_train = self.dev_df[self.feats]
        self.y_train = self.dev_df[self.target]
        self.weight_train = self.dev_df[self.weight_variable]

        score_params = {"sample_weight": self.weight_train}
        
        self.my_scorer = make_scorer(score_f,
                                      greater_is_better=False, 
                                      needs_proba=True, 
                                      needs_threshold=False,
                                      **score_params)

    def objective(self, trial):
        """Objective function for Optuna."""
        params = {
        par: (trial.suggest_int(par, arg[0][0], arg[0][1]) if arg[1]=='int'
             else trial.suggest_float(par, arg[0][0], arg[0][1]) if arg[1]=='float'
             else trial.suggest_categorical(par, arg[0]) if arg[1]=='cat'
             else None)
        for par, arg in self.grid_params.items()
        }        
        
        model = self.classifier(**params, random_state=self.random_state)
        # Set up Repeated K-Fold
        rkf = RepeatedKFold(n_splits=self.n_splits_kfold, n_repeats=self.n_repeats_kfold, random_state=self.random_state)
    
        score = cross_val_score(model, self.X_train, self.y_train, cv=rkf, scoring=self.my_scorer).mean()
        return score  # Maximize log-loss

    @time_function
    def optimize(self):
        """Run Optuna optimization."""
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=self.n_random_grids, timeout=self.timeout,  n_jobs=self.n_jobs)

    def best_params(self):
        """Return the best hyperparameters found by Optuna."""
        return self.study.best_params if self.study else None
    
    def best_score(self):
        """Return the best score found by Optuna."""
        return -self.study.best_value if self.study else None
        
    def train_best_model(self):
        """Train the best model using the entire dataset."""
        best_params = self.study.best_params
        self.best_model = self.classifier(**best_params, random_state=self.random_state)
        self.best_model.fit(self.X_train, self.y_train, sample_weight=self.weight_train)
    
    def predict_probabilities(self, X_test):
        """Predict probabilities using the best model."""
        if self.best_model is None:
            raise ValueError("Model is not trained yet. Call train_best_model() first.")
        return self.best_model.predict_proba(X_test)

        

