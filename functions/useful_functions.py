import os 

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

def identify_character_variables(
#    self,
    input_data
    ): 
    
    return input_data.columns[input_data.dtypes == object]
    
def identify_numeric_variables(
#    self,
    input_data
    ): 
    
    return input_data.columns[input_data.dtypes != object]
    
def create_folder(
    data_path, 
    folder_name
    ):
        
    if not os.path.isdir('{0}/{1}'.format(data_path, folder_name)):
        os.makedirs('{0}/{1}'.format(data_path, folder_name))

###################################################################################################################################
###################################################################################################################################
def data_split(
    input_data, 
    fraction = 0.7, 
    random_state = 1
    ):
    
    df = input_data.sample(frac=1, replace=False, random_state=random_state)
    cut = int(len(df)*fraction)
    dev = df.head(cut)
    oos = df.tail(len(df) - cut)
    return dev, oos

def _expand_unit(alist, weight):
    """Expand unit by weight."""
    expanded = []
    for x in alist:
        if x == 0:
            expanded += [0]*weight
        else:
            expanded.append(x)
    return expanded

def _expand_value(alist, weight):
    """Expand value by weight."""
    expanded = []
    for it in alist:
        if it[0] == 0:
            expanded += [0]*weight
        else:
            expanded.append(it[1])
    return expanded

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

def sample_data(df, target, pos, neg):
                """Sample the records by positive and negative"""
                df_pos = df[df[target] == 1].sample(frac=pos, replace=False)
                df_neg = df[df[target] == 0].sample(frac=neg, replace=False)
                return pd.concat([df_pos, df_neg], axis=0)
                
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


                