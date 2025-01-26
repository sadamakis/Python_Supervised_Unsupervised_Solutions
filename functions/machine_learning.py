import statsmodels.api as sm
import pandas as pd
import numpy as np
from io import StringIO
from matplotlib import pyplot as plt
import re
import pickle
import time
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score

from decorators import time_function 
import useful_functions as ufun

class logistic_regression():
    
    def __init__(
    self, 
    input_data, 
    final_feats, 
    target_variable, 
    weight_variable_name, 
    data_path
    ):
        
        self.input_data = input_data
        self.final_feats = final_feats
        self.target_variable = target_variable
        self.weight_variable_name = weight_variable_name
        self.data_path = data_path

        # Dictionary to save each summary table for each sample
        self.glm_bin_summary = {}
        
    @time_function
    def glm_bin(
    self, 
    sample_values_solution
    ):

        i = sample_values_solution[0]
        df = self.input_data['data_{}'.format(i)]
        X = sm.add_constant(df[self.final_feats])
        Y = df[self.target_variable]
        
        # Build the model and fit the data
        self.glm_binom = sm.GLM(Y, X, family=sm.families.Binomial(), freq_weights=df[self.weight_variable_name]).fit()
        
        summary_results = self.glm_binom.summary()
        results_as_csv = summary_results.tables[1].as_csv()
        results_str = StringIO(results_as_csv)
        
        self.glm_bin_summary['log_reg_summary_{}'.format(i)] = pd.read_csv(results_str, sep=',', skipinitialspace=True)
        self.glm_bin_summary['log_reg_summary_{}'.format(i)].columns = ['variable', 'coef', 'std_err', 'z', 'p_value', '[0.025', '0.975]']
        self.glm_bin_summary['log_reg_summary_{}'.format(i)]['variable'] = self.glm_bin_summary['log_reg_summary_{}'.format(i)]['variable'].str.strip()
        self.glm_bin_summary['log_reg_summary_{}'.format(i)]['statistically_significant'] = np.where(self.glm_bin_summary['log_reg_summary_{}'.format(i)]['p_value'] < 0.05, 'Yes', 'No')
        self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] = np.exp(self.glm_bin_summary['log_reg_summary_{}'.format(i)]['coef'])
        
        odds_ratio_condition = [
            (self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] < 1), 
            (self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] > 1), 
            (self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] == 1)]
            
        desc_value_rules = [1 - self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'], 
                            self.glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] - 1, 
                            1]
        self.glm_bin_summary['log_reg_summary_{}'.format(i)]['desc_value'] = np.select(odds_ratio_condition, desc_value_rules)
        return self.glm_binom, self.glm_bin_summary

    @time_function
    def glm_report(
        self
        ): 
        
        for i, j in self.glm_bin_summary.items():
            temp = self.glm_bin_summary[i]
            temp = temp.drop(['[0.025', '0.975]'], axis=1).round(4)
            display(temp)
            pd.DataFrame(temp).to_csv(self.data_path + '/output/' + str(i) + '.csv', index=False)
        return temp

    @time_function
    def create_predictions(
        self, 
        sample_values_dict, 
        amount_variable_name
    ):
        
        self.pred_dict = self.input_data.copy()

        for i, j in sample_values_dict.items():
            df = self.pred_dict['data_{}'.format(i)]
            X = sm.add_constant(df[self.final_feats])
            y_hat = self.glm_binom.predict(X)
            y_0 = list(map(round, y_hat))
            pred_dict_temp = df[[self.target_variable, self.weight_variable_name, amount_variable_name]].copy()
            pred_dict_temp['predicted_score_numeric'] = y_hat
            pred_dict_temp['predicted_score_binary'] = y_0
            self.pred_dict['data_{}'.format(i)] = pred_dict_temp
        
        return self.pred_dict
    
#############################################################################################################################################
#############################################################################################################################################

class clustering: 

    def __init__(
        self, 
        df, 
        sample_values_solution, 
        weights, 
        data_path, 
        filename = 'ClusterProfile'
        ):
        
        self.train_data = pd.concat((df['data_{}'.format(sample_values_solution[0])], weights.reset_index(drop=True)), axis=1)
        self.df = df
        self.sample_values = sample_values_solution
        self.weights = weights
        self.data_path = data_path
        self.filename = filename
        
    def set_test_model(
        self, 
        model_inputs
        ):
    
        self.model = model_inputs["Model"]
        self.model_nm = re.sub('\'|>', '', str(model_inputs["Model"])).split('.')[-1]
        self.model_args = model_inputs["default_args"]
        self.indep_param_nm = model_inputs["test_arg"]
        self.indep_param_values = model_inputs["test_values"]
        
        self.profile_df = pd.DataFrame({model_inputs["test_arg"]: model_inputs["test_values"],
                                        'Silhouette score': [np.nan]*len(model_inputs["test_values"]),
                                        'Calinski Harabasz score': [np.nan]*len(model_inputs["test_values"]),
                                        'Davies Bouldin score': [np.nan]*len(model_inputs["test_values"]),
                                        'Adjusted Rand Index score': [np.nan]*len(model_inputs["test_values"])
                                        })
                                        
        if self.model_nm.lower() == 'kmeans': 
            self.profile_df['WCSS'] = [np.nan]*len(model_inputs["test_values"])
            self.profile_df['Stability score'] = [np.nan]*len(model_inputs["test_values"])
        if self.model_nm.lower() == 'dbscan':
            self.profile_df['n_clusters'] = [np.nan]*len(model_inputs["test_values"])
            self.profile_df['outlier_pct'] = [np.nan]*len(model_inputs["test_values"])
            
            
    def adjusted_rand_index(
        self, 
        t, 
        boostraps, 
        sample_size
        ):
    
        rng = np.random.RandomState(6)
        data = self.train_data.sample(frac=sample_size)
        X = data.iloc[:, :-1]
        weights = data.iloc[:, -1]
        args = self.model_args
        args[self.indep_param_nm] = t
        
        labels = []
        indices = []
        for i in range(boostraps):
            sample_indices = rng.randint(0, X.shape[0], X.shape[0])
            indices.append(sample_indices)
            est = self.model(**args)
            if hasattr(est, "random_state"): 
                est.random_state = rng.randint(1e5)
            X_bootstrap = X.iloc[sample_indices]
            weights_bootstrap = weights.iloc[sample_indices]
            est.fit(X_bootstrap, sample_weight = weights_bootstrap)
            relabel = -np.ones(X.shape[0], dtype=int)
            relabel[sample_indices] = est.labels_
            labels.append(relabel)
        scores = []
        for l, i in zip(labels, indices):
            for k, j in zip(labels, indices):
                in_both = np.intersect1d(i, j)
                scores.append(adjusted_rand_score(l[in_both], k[in_both]))
        return np.mean(scores)
        
    def cluster_stability_kmeans(
        self, 
        t, 
        bootstraps, 
        sample_size
        ):
        
        args = self.model_args
        args[self.indep_param_nm] = t
        
        data = self.train_data.sample(frac=sample_size)
        orig = data.iloc[:, :-1]
        weights = data.iloc[:, -1]
        orig_model = self.model(**args)
        orig_model.fit(orig, sample_weight=weights)
        
        ari = []
        
        for b in range(bootstraps):
            new_model = self.model(**args)
            new_model.fit(orig, sample_weight=weights)
            ari.append(adjusted_rand_score(orig_model.labels_, new_model.labels_))
            
        return np.mean(ari)
        
    def get_metrics(
        self, 
        labels, 
        t, 
        bootstraps, 
        sample_size
        ):
            
# Set random seed for reproducibility
        np.random.seed(42)
        
        if self.model_nm.lower() == 'dbscan':
            df = self.df['data_{}'.format(self.sample_values[0])][labels != -1]
            labels = labels[labels != -1]
        else: 
            df = self.df['data_{}'.format(self.sample_values[0])]
        # Silhouette score 
        ss = round(silhouette_score(df, labels), 2)
        # Calinski Harabasz score 
        cs = round (calinski_harabasz_score(df, labels), 2)
        # Davies Bouldin score 
        db = round(davies_bouldin_score(df, labels), 2)
        # Adjusted Rand Index 
        ari = round(self.adjusted_rand_index(t, bootstraps, sample_size), 2)
        return (ss, cs, db, ari)
        
    @time_function
    def get_profile(
        self, 
        bootstraps=5, 
        sample_size=0.1
        ):
        
        for t in self.indep_param_values: 
            t0 = time.time()
            # Add independent parameter into model arguments 
            args = self.model_args
            args[self.indep_param_nm] = t 
            
            # Fit model 
            m = self.model(**args)
            m.fit(self.train_data.iloc[:, :-1], sample_weight=self.train_data.iloc[:, -1])
            
            # Add metrics to profile_df
            if len(set(m.labels_)) - (1 if -1 in m.labels_ else 0) > 1:
                metrics = self.get_metrics(m.labels_, t, bootstraps, sample_size)
                if self.model_nm.lower() == 'kmeans': 
                    metrics += (round(m.inertia_, 2), )
                    metrics +=(round(self.cluster_stability_kmeans(t, bootstraps, sample_size), 2), )
                if self.model_nm.lower() == 'dbscan': 
                    metrics += (len(set(m.labels_)) - (1 if -1 in m.labels_ else 0), )
                    metrics +=(np.round(np.sum(m.labels_ == -1) / len(m.labels_), 3), )
                self.profile_df.loc[self.profile_df[self.indep_param_nm]==t, self.profile_df.columns[1:]] = metrics
                
            print('Cluster profiling for {0} took {1}s. to run'.format(t, round(time.time()-t0, 2)))
        
        # Write output
        self.profile_df.to_csv(self.data_path + '/output/' + self.filename + self.model_nm + '.csv', index=False)
        
        return self.profile_df 
    
    def plot_profile(
        self
        ):
        
        for metric in self.profile_df.columns[1:]:
            fig, ax = plt.subplots()
            plt.plot(self.profile_df[self.indep_param_nm], self.profile_df[metric])
            plt.xlabel('Number of clusters')
            plt.ylabel(metric, fontsize = 15)
            plt.savefig('{0}/output/graphs/{1}_elbow.png'.format(self.data_path, self.model_nm+metric.replace(' score', '')))
            plt.show()
            
    @time_function
    def create_model(
        self, 
        model_inputs, 
        filename, 
        **kwargs
        ):
    
        self.model_nm = re.sub('\'|>', '', str(model_inputs["Model"])).split('.')[-1]
        self.kwargs = model_inputs["kwargs"]
        self.fit_model = model_inputs["Model"](**model_inputs["kwargs"]).fit(self.train_data.iloc[:, :-1], sample_weight = self.train_data.iloc[:, -1])
        
        with open(self.data_path + '/output/' + filename, 'wb') as file:
            pickle.dump(self.fit_model, file)
            
        return self.fit_model
        
    @time_function
    def validate_data(
        self
        ):
        
        self.valid_df = pd.DataFrame({'Split': self.sample_values,
                                        'Sample Size': [np.nan]*len(self.sample_values),
                                        'Silhouette score': [np.nan]*len(self.sample_values),
                                        'Scaled Calinski Harabasz score': [np.nan]*len(self.sample_values),
                                        'Davies Bouldin score': [np.nan]*len(self.sample_values)
                                    })
        if self.model_nm.lower() =='kmeans':
            self.valid_df['Scaled WCSS'] = [np.nan]*len(self.sample_values)
            
        for i, split in enumerate(self.sample_values):
            df = self.df['data_{}'.format(split)]
            
            labels = self.fit_model.predict(df)
            
            # Sample size 
            self.valid_df.loc[self.valid_df.Split == split, "Sample Size"] = len(labels)
            # Silhouette score
            if len(df)>100000:
                self.valid_df.loc[self.valid_df.Split == split, "Silhouette score"] = round(silhouette_score(df, labels, sample_size=100000), 3)
            else:
                self.valid_df.loc[self.valid_df.Split == split, "Silhouette score"] = round(silhouette_score(df, labels, sample_size=None), 3)
            # Scaled Calinski Harabasz score
            self.valid_df.loc[self.valid_df.Split == split, "Scaled Calinski Harabasz score"] = round(calinski_harabasz_score(df, labels) / len(labels), 3)
            # Davies Bouldin score
            self.valid_df.loc[self.valid_df.Split == split, "Davies Bouldin score"] = round(davies_bouldin_score(df, labels), 3)
            if self.model_nm.lower() == 'kmeans': 
                # Scaled WCSS
                self.valid_df.loc[self.valid_df.Split == split, "Scaled WCSS"] = round(self.fit_model.score(df) * -1 / len(labels), 3)
                
            for j in np.unique(labels):
                self.valid_df.loc[self.valid_df.Split == split, "Cluster {} Size".format(j+1)] = round((labels == j).mean(), 3)
                
        return self.valid_df




            