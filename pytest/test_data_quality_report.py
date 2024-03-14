import os
import sys
import pandas as pd
import numpy as np
from IPython.display import display

functions_path = os.path.join(os.path.dirname(os.getcwd()), 'functions')
sys.path.insert(0, functions_path)

from data_quality_report import dq_report

def test_dq_report_1():
    data_full = pd.read_csv("data/input/test_data_1.csv")
    variables = ["Z", "Y", "X", "weights"]

    dq = dq_report(data_full, "data", variables, weight_variable = "weights")

    rep = dq.data_quality_df 

    # Test that all the variables are in the data quality report 
    assert rep["Variable Name"].tolist() == variables 
    # Test the weighted and unweighted missing value percentage 
    assert rep.loc[0, "Missing Value Percentage"] == 20
    assert rep.loc[1, "Missing Value Percentage"] == 10
    # Test the weighted means 
    assert np.isclose(rep.loc[2, "Mean"], 2.3)
    assert np.isclose(rep.loc[1, "Mean"], 12.2222)
    # Test that column 2 through 5 (mean, median, min, max) are all null for a categorical variable 
    assert rep.iloc[0, 2:6].isnull().all()
    # Test min and max are correct 
    assert rep.loc[2, "Min"] == 1 
    assert rep.loc[2, "Max"] == 3
    # Test unique values 
    assert rep.loc[0, "Unique Values"] == 2 
    # Test correct file is created 
    assert os.path.isfile("data/output/data_quality_report.csv")    
