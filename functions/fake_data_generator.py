from sklearn.datasets import make_regression, make_classification
import pandas as pd
import numpy as np
import random
from random import seed
from random import sample 
from decorators import time_function
import time

def createList (r1, r2):
    return [item for item in range (r1, r2+1)]

@time_function
def fake_data_generator(
    #The output of this function is a pandas dataframe that has: target variable, model predictors. 
    target_type, #String. Define whether the target variable should be numeric (provide 'n') or binary (provide any other string). 
    target_1_probability, #Float, takes values 0-1 Relevant if the target variable is binary: provide the percentage of the 1s in the. target variable. If the target is numeric, then you can leave this argument blank.
    sample_size, #Integer. Number of records in the output dataset
    predictors, #Integer. Number of predictors in the output dataset.
    pct_informative_predictors, #Float Percentage of informative predictors - takes values between 0 and 1
    bias_var, #Float. The bias term in the underlying linear model, 0 means no bias. bias var=0 and noise_var=0 means perfect correlation, e.g. R 2-/Gini =
    noise_var, #Float. The standard deviation of the gaussian noise applied to the output, 0 means no noise applied to the output. 
    weight_var, # String. weight_variable information: '1' returns a vector with 19, random returns a vector with random weights. 
    seed #Integer. set the seed so that the fake dataset will be reproducible.
    ):

    start_time_initial = time.time()
    if target_type == 'n':
        test_data = make_regression (
            n_samples = sample_size
            , n_features = predictors
            , n_informative = round (pct_informative_predictors*predictors)
            , n_targets = 1
            , bias = bias_var
            , noise = noise_var
            , random_state = seed
         )
    else:
        test_data = make_classification (
            n_samples = sample_size
            , n_features = predictors
            , n_informative = round (pct_informative_predictors*predictors) 
            , n_classes = 2
            , weights = [1-target_1_probability, target_1_probability] 
            , random_state = seed
        )
    print('This sklearn sample took %.2fs. to run'%(time.time()-start_time_initial)) 
    start_time = time.time()
    #configuring random predictors
    x_data = pd.DataFrame (test_data[0])
    x_data = x_data.add_prefix ('random_var_')
    fcols = x_data.select_dtypes ('float').columns
    x_data[fcols]= x_data.apply (pd.to_numeric, downcast='float') #reducing dataframe size

    #Setting Seed
    np.random. seed (seed)
    #Categorical variable creation cat_3, cat_5, cat_20, cat_200
    cat_3 = createList (1,3)


    x_data['cat_3'] = np.array(np.random.choice (cat_3, size =sample_size), dtype=np.int32)
    cat_5 = createList (1,5)
    x_data['cat_5'] = np.array(np.random.choice (cat_5, size =sample_size), dtype=np.int32)
    cat_20 = createList (1,20)
    x_data['cat_20'] = np.array(np.random.choice (cat_20, size =sample_size), dtype=np.int32)
    cat_200 =createList (1,200)
    x_data['cat_200'] = np.array (np.random.choice (cat_200, size = sample_size), dtype=np.int32)
    np.random.seed (seed)
    # Numerical variable creation num_2%, num_20%, num_90%, num_99% (percentages indicate & rows to be replaced with null values)
    x_data['num_2%'] = np.random.rand (1,len (x_data.index))[0].astype (np.float32)+10
    x_data['num_20%'] = np.random.rand (1,len (x_data.index)) [0].astype (np.float32) *100
    x_data['num_90%'] = np.random.rand (1,len (x_data.index))[0].astype (np.float32) *1000
    x_data['num_99%'] = np.random.rand (1,len (x_data.index))[0].astype (np.float32) *10000
    
    #Imputing null values
    np.random.seed(seed)
    mask_array = np.column_stack((
        np.random.choice ([True, False], size=len (x_data.index), p=[0.99, 0.01]),
        np.random.choice ([True, False], size=len (x_data.index), p=[0.10, 0.90]), 
        np.random.choice ([True, False], size=len (x_data.index), p=[0.02, 0.98]), 
        np.random.choice ([True, False], size=len (x_data.index), p=[0.20, 0.80]), 
        np.random.choice ([True, False], size=len (x_data.index), p=[0.90, 0.10]), 
        np.random.choice ([True, False], size=len (x_data.index), p=[0.99, 0.01])
    ))
    x_data[['cat_5', 'cat_20', 'num_2%', 'num_20%', 'num_90%', 'num_99%']] =  x_data[['cat_5', 'cat_20', 'num_2%', 'num_20%', 'num_90%', 'num_99%']].mask (mask_array)

    #Convert variables to character
    for var in ['cat_3', 'cat_5', 'cat_20', 'cat_200']:
        x_data [var] = x_data[var].astype (str)
    print('Random numeric and categorical variables took %.2fs, to run'%(time.time()-start_time))
    
    #weight_variable
    if weight_var == 'random':
        weights = x_data.apply(lambda x: np.random.random(), axis=1). astype (np.float32)
    else:
        weights = int (weight_var)
    x_data['weight_variable'] = weights
    print('GMI attributes took %.2fs. to run'%(time.time() -start_time))
    
    start_time_initial = time.time()

    # sample_variable
    x_data['sample variable'] = np.array (list (map (lambda x: str(x).replace('True', 'training').replace('False', 'validation'), np.random.choice([True, False], size=sample_size, p=[0.70, 0.30]))))
    print('Train/Test split took %.2fs. to run'%(time.time() -start_time))
    
    # Configuring the target variable
    y_data = pd.DataFrame(test_data[1])
    y_data = y_data.rename(columns={0: "target"}).astype(np.int32)
    
    # final df 
    df = x_data.join(y_data, how='left')
    
    # amount_variable
    df['amount'] = [0 if x ==0 else np.random.normal(loc=10000, scale=2000, size=None) for x in df['target']]

    print('This code took %.2fs. to run'%(time.time() -start_time))
    return df
        