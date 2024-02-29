import pandas as pd
import numpy as np
from sklearn.decomposition import PCA 
from matplotlib import pyplot as plt
import pickle
import os 
from decorators import time_function 

class dimension_reduction: 

    def __init__(
        self, 
        dic_of_dfs, 
        data_path, 
        training_sample
        ):
        
        self.data = dic_of_dfs
        self.data_path = data_path
        self.training_sample = training_sample
        
    @time_function
    def explore(
        self, 
        solver='full'
        ):

        # Ensure that the graph folder exists
        if not os.path.isdir('{0}/output/graphs'.format(self.data_path)):
            os.makedirs('{0}/output/graphs'.format(self.data_path))

        num_predictors = self.data[self.training_sample].shape[1]
        pca = PCA(n_components = num_predictors, svd_solver = solver).fit(self.data[self.training_sample])
        
        PC_values = np.arange(pca.n_components_) + 1
        
        # Plot Scree plot and output to graphs folder
        plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.savefig(self.data_path + "/output/graphs/PCA_scree_plot.png")
        plt.show()
        
        # Plot Cumulative variance plot and output to graphs folder
        plt.plot(PC_values, pca.explained_variance_ratio_.cumsum(), 'o-', linewidth=2, color='blue')
        plt.title('Cumulative Variance Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Variance Explained')
        plt.savefig(self.data_path + "/output/graphs/PCA_cumulative_variance_plot.png")
        plt.show()

        print("Variance explained by each principal component:\n", pca.explained_variance_ratio_)
        print("Cumulative sum of variance explained by each principal component:\n", pca.explained_variance_ratio_.cumsum())
        
        return pca
        
    def fit_transform(
        self, 
        pca_components, 
        solver='full', 
        filename='pca_model.pkl'
        ):

        out = {}
        
        # Make PCA object to transform data
        pca = PCA(n_components = pca_components, svd_solver = solver).fit(self.data[self.training_sample])
        
        pickle.dump(pca, open(self.data_path + '/output/' + filename, 'wb'))
        
        # loop through data 
        for k in self.data.keys(): 
            out[k] = pd.DataFrame(pca.transform(self.data[k]))
            
        return out

