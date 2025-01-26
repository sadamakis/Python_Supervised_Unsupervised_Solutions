from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo, FactorAnalyzer
from matplotlib import pyplot as plt
import pandas as pd
import json
import os 
from decorators import time_function 

#COPIED 
class FactorAnalysis:
    def __init__(
        self, 
        data, 
        training_sample, 
        datapath, 
        filename = 'FactorAnalysis'
    ):

        self.data = data 
        self.X = data[training_sample]
        self.datapath = datapath
        self.filename = filename 
        self.kmo_passed = []
        self.kmo_not_passed = []

    @time_function
    def setup(
        self, 
        kmo_threshold=0.5
    ):
    
        # Ensure that the graph folder exists
        if not os.path.isdir('{0}/output/graphs'.format(self.datapath)):
            os.makedirs('{0}/output/graphs'.format(self.datapath))

        # Bartlett's test of Sphericity
        chi2, p = calculate_bartlett_sphericity(self.X)
        print("Results of Bartlett's test of sphericity:")
        print(f"\tChi squared value : {chi2}")
        print(f"\tp value : {p}")
        print()
        
        # Compute Kaiser-Meyer-Olkin (KMO) test for the original dataset
        kmo_all, kmo_model = calculate_kmo(self.X)
        print("Results of Kaiser-Meyer-Olkin (KMO) test:")
        print("Overall KMO = {:.3f}".format(kmo_model))
        print()

        # Select only adequate variables and recompute KMO 
        self.kmo_passed = list(self.X.columns[kmo_all >= kmo_threshold])
        self.kmo_not_passed = list(self.X.columns[kmo_all < kmo_threshold])
        kmo_all, kmo_model = calculate_kmo(self.X[self.kmo_passed])
        print("Selecting adequate variables and recomputing KMO")
        print("\tOverall KMO = {:.3f}".format(kmo_model))
        print(f"\tVariables with KMO >= {kmo_threshold} = {self.kmo_passed}")
        print(f"\t# of variables with KMO >= {kmo_threshold} = {len(self.kmo_passed)}")
        print(f"\tVariables with KMO < {kmo_threshold} = {self.kmo_not_passed}")
        print(f"\t# of variables with KMO < {kmo_threshold} = {len(self.kmo_not_passed)}")
        print()

        # Determining the number of factors
        fa = FactorAnalyzer(rotation=None, impute='drop', n_factors=self.X[self.kmo_passed].shape[1])
        fa.fit(self.X[self.kmo_passed])
        ev,_ = fa.get_eigenvalues()
        plt.scatter(range(1, self.X[self.kmo_passed].shape[1]+1), ev)
        plt.plot(range(1, self.X[self.kmo_passed].shape[1]+1), ev)
        plt.title('Factor Analysis Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalues')
        plt.grid()
        plt.savefig(f'{self.datapath}/output/graphs/{self.filename}_FA_scree_plot.png')
        plt.show()
    
    @time_function
    def remove_features(
        self, 
        n_factors, 
        loadings_threshold=0.7, 
        **kwargs
    ):
    
        fa = FactorAnalyzer(n_factors=n_factors, **kwargs)
        fa.fit(self.X[self.kmo_passed])
        
        # Get factor loadings 
        loadings = pd.DataFrame(fa.loadings_, index=self.kmo_passed)
        print('Factor loadings table')
        display(loadings)
        loadings.to_csv(f'{self.datapath}/output/{self.filename}_loadings.csv')
        
        # See variables that have high loadings for the same factors 
        res = {factor: [] for factor in loadings.columns}
        to_drop = []
        for f in loadings.columns: 
            high_loadings = loadings.index[loadings[f].abs() > loadings_threshold].tolist()
            res[f] += high_loadings
            to_drop += high_loadings[1:]
        remaining_predictors = self.X.columns.drop(to_drop).tolist()
        with open(f'{self.datapath}/output/{self.filename}_summary.json', 'w') as f:
            json.dump(res, f, indent=4)
        print('Features with high loadings')
        display(res)
        print(f'Features dropped: {to_drop}')
        print(f'Remaining features: {remaining_predictors}')
        print(f'Number of remaining features: {len(remaining_predictors)}')
        
        # Drop variables with high loadings in the same factor
        return {k:v[remaining_predictors] for k, v in self.data.items()}
