import pandas as pd
import os
import random
import time
from decorators import time_function 

# COPIED
@time_function
def load_data(
    method, 
    data_path, 
    table_name, 
    sample=1
    ):

    assert(0 <= sample <= 1)
    sample = float(sample)
    
    if method == 'csv':
        if sample != 1:
            random.seed(42)
            n = sum(1 for line in open(os.path.join(data_path, table_name)))
            s = int(n*sample)
            skip = sorted(random.sample(range(1, n+1), n-s))
            data_full = pd.read_csv(os.path.join(data_path, table_name), sep=',', skiprows=skip)
        else: 
            data_full = pd.read_csv(os.path.join(data_path, table_name), sep=',')
        return data_full
    elif method == 'parq':
        data_full = pd.read_parquet(os.path.join(data_path, table_name), sep=',')
        return data_full.sample(frac=sample, random_state=42)
    else:
        raise Exception("Method not found. Please set method equal to an acceptable input.")
            

