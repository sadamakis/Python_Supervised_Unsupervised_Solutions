"""
# This is a set of machine learning tools developed using Python
# Facilities to preprocess the data:
1. Process categorical variables
2. Impute the missing values
3. Exclude variables 
"""

import numpy as np
import pandas as pd
from evaluation import gini
from evaluation import gini_weight
from sklearn.metrics import roc_auc_score
from decorators import time_function 

# COPIED
def get_cat_vars(df):
                """Get categorical variable list.
                
                Parameters
                =============
                df: array_like
                                The input dataframe.
                                
                Returns
                =============
                out: list
                                The list of categorical variables.
                """
                return df.columns[df.dtypes == object]
                
# COPIED
def get_num_vars(df):
                """Get numerical variable list.
                
                Parameters
                =============
                df: array_like
                                The input dataframe.
                                
                Returns
                =============
                out: list
                                The list of numerical variables.
                """
                return df.columns[df.dtypes != object]
                
# COPIED
def classify_vars(df):
                """Classify variables into categorical and numerical.
                Parameters
                =============
                df: array_like
                                The input dataframe.
                                
                Returns
                =============
                out: tuple
                                The list of numerical and categorical variables.
                """
                num_vars = df.columns[df.dtypes != object]
                cat_vars = df.columns[df.dtypes == object]
                return list(num_vars), list(cat_vars)

def var_missing(df, weight_variable, threshold=0.99):
                """Find the variables with large missing rate.
                Parameters
                =============
                df: array_like
                                The input dataframe.
                weight_variable: object
                                The name of the weight variable.
                threshold: float
                                The threshold to considering dropping the variables
                                
                Returns
                =============
                out: list
                                The list of variables with missing percentage equal or above the threshold.
                """
                return list(df.columns[df.apply(lambda x: sum(df[x.isnull()][weight_variable])/sum(df[weight_variable]), axis=0)>threshold])
                

def classify_cats(df, cat_var_list, threshold=50):
                """Classify categorical variables by the values.
                Parameters
                =============
                df: array_like
                                The input dataframe.
                cat_var_list: list
                                The list of categorical variables.
                                
                Returns
                =============
                out: dictionary
                                The dictionary contains classified categorical variables.
                                                'single': only one single value_counts
                                                'binary': there are two values
                                                'small': small amount of values 3 - threshold
                                                'large': large amount of values > threshold
                """
                cats = {'single':[], 'binary':[], 'small':[], 'large':[]}
                for x in cat_var_list:
                                n_values = len(df[x].value_counts(dropna=False).index)
                                if n_values == 1:
                                                cats['single'].append(x)
                                elif n_values == 2:
                                                cats['binary'].append(x)
                                elif n_values <= threshold:
                                                cats['small'].append(x)
                                else:
                                                cats['large'].append(x)
                return cats

def cats_binary_num(df, cats_binary):
                """Convert binary variables to binary 

                Parameters
                =============
                df: array_like
                                The input dataframe.
                cats_binary:
                                The list of categorical variables that contain 2 values.
                                
                Returns
                =============
                                Append numerical binary variables in-place.
                """
                for x in cats_binary:
                    val = df[x].value_counts().index[0]
                    df[x + '_ind'] = df[x].map(lambda t: 1 if t == val else 0)
                                
def cats_small_num(df, cats_small, drop):
                """Convert categorical variables into binary.

                Parameters
                =============
                df: array_like
                                The input dataframe.
                cats_small: list
                                The list of categorical variables that contain more than 2 values.
                drop: None, 'first', 'last'
                                
                Returns
                =============
                Append numerical binary variables in-place.
                """
                for x in cats_small:
                    if drop == 'last':
                        values = df[x].value_counts(dropna=False).index
                        values = values[:len(values)-1]
                    elif drop == 'first':
                        values = df[x].value_counts(dropna=False).index
                        values = values[1:]
                    elif drop == None:
                        values = df[x].value_counts(dropna=False).index
                    else:
                        raise Exception("Please choose a value for the drop argument from the pre-defined list: None, 'first', 'last'")
                    for v in values:
                        if v is not np.nan:
                            df[x + '_' + str(v)] = df[x].map(lambda t: 1 if t == v else 0)
                        else: 
                            df[x + '_' + str(v)] = df[x].map(lambda t: 1 if t is v else 0)
                                                                                
# COPIED 
def sample_data(df, target, pos, neg):
                """Sample the records by positive and negative"""
                df_pos = df[df[target] == 1].sample(frac=pos, replace=False)
                df_neg = df[df[target] == 0].sample(frac=neg, replace=False)
                return pd.concat([df_pos, df_neg], axis=0)

def binarize(df, cat):
                values = df[cat].value_counts().index
                for x in values:
                                df[cat + '_' + x] = df[cat].map(lambda t: 1 if t == x else 0)
                df[cat + '_' + 'NA'] = df[cat].map(lambda t: 1 if t is np.nan else 0)
                return values

def generate_cats(df, cats):
                cat_values = dict()        
                for cat in cats:
                                cat_values[cat] = binarize(df, cat)
                return cat_values

class Imputer(object):
                """Imputation class"""
                def __init__(self, vars, strategy="median"):
                                self.vars = vars
                                self.imputer = dict()
                                self.strategy = strategy
                                
                def fit(self, df):
                                """Compute the imputation values.
                                The list of categorical variables that contain 2 values.
                                """
                                if self.strategy == 'median':
                                                for x in self.vars:
                                                                self.imputer[x] = df[x].median()
                                elif self.strategy == 'mean':
                                                for x in self.vars:
                                                                self.imputer[x] = df[x].mean()
                                elif self.strategy == 0:
                                                for x in self.vars:
                                                                self.imputer[x] = 0
                                elif self.strategy == 'rankplot':
                                                pass       
                                return self
                                
                def fit_weight(self, df, weight_variable):
                                """Compute the imputation values.
                                The list of categorical variables that contain 2 values.
                                """
                                if self.strategy == 'median':
                                                import weighted as wghtd
                                                for x in self.vars:
                                                                self.imputer[x] = wghtd.median(df[x].dropna(), df[~df[x].isnull()][weight_variable])
                                elif self.strategy == 'mean':
                                                for x in self.vars:
                                                                self.imputer[x] = np.average(df[x].dropna(), weights=df[~df[x].isnull()][weight_variable])
                                elif self.strategy == 0:
                                                for x in self.vars:
                                                                self.imputer[x] = 0
                                elif self.strategy == 'rankplot':
                                                pass       
                                return self

                def transform(self, df):
                                """Impute the data"""
                                for x in self.vars:
                                                df[x] = df[x].fillna(self.imputer[x])
                

def data_split(df_in, frac=0.7):
                """Split the data into development(DEV) and out of sample(OOS).

                Parameters
                =============
                df_in: array_like
                                The input dataframe.
                frac: float
                                The fraction of DEV data.
                                
                Returns
                =============
                dev: dataframe
                                The DEV dataset.
                oos: dataframe
                                the OOS dataset.
                """
                df = df_in.sample(frac=1, replace=False)
                cut = int(len(df)*frac)
                dev = df.head(cut)
                oos = df.tail(len(df)-cut)
                return dev, oos
                
def outerlier_treat():
                pass

def special_value_treat():
                pass

def gini_values(feats, input_data, target_variable):
    feat_gini = dict()
    for x in feats:
        feat_gini[x] = abs(gini(input_data[[x, target_variable]].values))
    rank = sorted(feat_gini.items(), key=lambda t: t[1], reverse=True)
    return rank 

#COPIED
@time_function
def gini_values_weight(feats, input_data, target_variable, weight_variable, data_path, gini_info_file = 'gini_info.csv', n_bands=10):
    feat_gini = dict()
    for x in feats:
#        feat_gini[x] = abs(gini_weight(input_data[[x, target_variable, weight_variable]].values, n_bands))
        feat_gini[x] = abs(2*roc_auc_score(input_data[target_variable].values, input_data[x].values, sample_weight=input_data[weight_variable].values)-1)
    gini_table = sorted(feat_gini.items(), key=lambda t: t[1], reverse=True)
    
    gini_table = pd.DataFrame(gini_table)
    gini_table = gini_table.rename(columns={0: "variable", 1: "Gini coefficient"})
    display(gini_table)    
    
    gini_table.to_csv('{0}/output/{1}'.format(data_path, gini_info_file), header=True, index=False)

    return gini_table 

def gini_selection(feats, input_data, target_variable, gini_threshold):
    feat_gini = dict()
    for x in feats:
        feat_gini[x] = gini(input_data[[x, target_variable]].values)
    rank = sorted(feat_gini.items(), key=lambda t: abs(t[1]), reverse=True)
    return [x[0] for x in rank if abs(x[1]) > gini_threshold] 

def gini_selection_weight(feats, input_data, target_variable, weight_variable, n_bands=10, gini_threshold=0.0001):
    feat_gini = dict()
    for x in feats:
#        feat_gini[x] = gini_weight(input_data[[x, target_variable, weight_variable]].values, n_bands)
        feat_gini[x] = abs(2*roc_auc_score(input_data[target_variable].values, input_data[x].values, sample_weight=input_data[weight_variable].values)-1)
    rank = sorted(feat_gini.items(), key=lambda t: abs(t[1]), reverse=True)
    return [x[0] for x in rank if abs(x[1]) > gini_threshold] 

# COPIED 
def target_stratified_sampling(
    df, # Input dataframe that has the target value
    target_variable, # Target variable name
    weight_variable, # Weight variable name: this field will be the updated weight variable
    good_bad_ratio # Ratio of goods/bads to keep after the sampling: set to >1 to select more goods than bads, 
                    # set to <1 to select more bads than goods, set to =1 to select the same number of goods/bads
):
    import random
    random.seed(10)
    good_proportion = good_bad_ratio*df[target_variable].value_counts()[1]/df[target_variable].value_counts()[0]
    df['random_variable'] = [1 if x==1 else random.uniform(0, 1) for x in df[target_variable]] 
    good_counts = df[target_variable].value_counts()[0]
    df = df[df['random_variable']>(1-good_proportion)].drop('random_variable', axis=1)
    weight_value = good_counts/df[target_variable].value_counts()[0]
    #print(weight_value)
    df['weight_temp'] = [1 if x==1 
                         else weight_value for x in df[target_variable]] 
    df[weight_variable] = df[weight_variable]*df['weight_temp']
    df = df.drop('weight_temp', axis=1)
    return df
