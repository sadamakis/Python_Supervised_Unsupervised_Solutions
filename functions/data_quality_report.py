import pandas
import numpy as np
from data_processing import * 
import time
from IPython.display import display

class dq_report:
    def __init__(self, 
                df, 
                data_path, 
                variables, 
                weight_variable = None, 
                dq_report_file = 'data_quality_report.csv'):
                
        start_time = time.time()
    
        self.df = df[variables]
        self.variables = variables 
        self.data_path = data_path
        self.weight_variable = weight_variable
        self.dq_report_file = dq_report_file
        
        # Get missing value rates
        if self.weight_variable == None:
            missing_list = self.df.apply(lambda x: (sum(x.isnull())/self.df.shape[0]) * 100, axis=0).sort_values(ascending=False)
            self.missing_val_df = pd.DataFrame(missing_list)
            self.missing_val_df.columns = ['Missing Value Percentage']
            numeric_vars_100_missing = list(self.missing_val_df[self.missing_val_df['Missing Value Percentage']==100].reset_index()['index'])
        else: 
            missing_list = self.df.apply(lambda x: (sum(self.df[x.isnull()][self.weight_variable])/sum(self.df[self.weight_variable])) * 100, axis=0).sort_values(ascending=False)
            self.missing_val_df = pd.DataFrame(missing_list)
            self.missing_val_df.columns = ['Missing Value Percentage']
            numeric_vars_100_missing = list(self.missing_val_df[self.missing_val_df['Missing Value Percentage']==100].reset_index()['index'])
        
        # Identify numeric variables and get their min and max values
        self.numeric_vars = identify_numeric_variables(self.df)
        
        min_df = self.df[self.numeric_vars].min(axis=0).to_frame(name = 'Min')
        max_df = self.df[self.numeric_vars].max(axis=0).to_frame(name = 'Max')
        
        # Identify character variables
        self.character_vars = identify_character_variables(self.df)
        
        # Identify numeric variables that do not have 100% missing values 
        self.numeric_vars_not_missing = [val for val in self.numeric_vars if val not in numeric_vars_100_missing]
        
        # Get weighted average of numeric variables 
        mean_imputer = impute_missing(self.numeric_vars_not_missing, imputation_strategy = 'mean')
        mean_imputer.imputation_fit_weight(self.df, self.weight_variable)
        mean_df = pd.DataFrame(mean_imputer.impute_missing, index=['Mean']).T

        # Get median of numeric variables 
        median_imputer = impute_missing(self.numeric_vars_not_missing, imputation_strategy = 'median')
        median_imputer.imputation_fit_weight(self.df, self.weight_variable)
        median_df = pd.DataFrame(median_imputer.impute_missing, index=['Median']).T
#        median_df = self.df[self.numeric_vars_not_missing].median().to_frame(name='Median')
        
        # Get number of unique values per feature (excluding missing values)
        unique_vals = dict()
        for v in self.df.columns:
            unique_vals[v] = len(self.df[v].value_counts())
        unique_vals_df = pd.DataFrame(unique_vals, index=['Unique Values']).T
        
        # Join all of the stats together to create one data quality report dataframe
        data_quality_df = self.missing_val_df.join(min_df, how='outer')\
                                            .join(max_df, how='outer')\
                                            .join(mean_df, how='outer')\
                                            .join(median_df, how='outer')\
                                            .join(unique_vals_df, how='outer')

        # Filter variables in data quality report 
        data_quality_df = data_quality_df.loc[self.variables]
        data_quality_df.rename_axis('Variable Name', inplace=True)
        data_quality_df = data_quality_df.reset_index()
        
        # Save data quality report
        self.data_quality_df = data_quality_df.sort_values(by='Missing Value Percentage', ascending=False)
        self.data_quality_df.to_csv('{0}/output/{1}'.format(self.data_path, self.dq_report_file))
        display(self.data_quality_df)
        print('Data quality report took %.2fs. to run'%(time.time() - start_time))