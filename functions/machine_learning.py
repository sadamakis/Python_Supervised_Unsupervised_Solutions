import statsmodels.api as sm
import pandas as pd
import numpy as np
from io import StringIO
from matplotlib import pyplot as plt
import re
import pickle
import time
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import resample

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
        
#        self.train_data = pd.concat((df['data_{}'.format(sample_values_solution[0])], weights.reset_index(drop=True)), axis=1)
        self.train_data = pd.concat((df['data_{}'.format(sample_values_solution[0])], weights), axis=1)
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
            self.profile_df['Stability score (Bootstrapped ARI)'] = [np.nan]*len(model_inputs["test_values"])
            self.profile_df['Bootstrapped Std. Dev. Scaled WCSS'] = [np.nan]*len(model_inputs["test_values"])
        if self.model_nm.lower() == 'dbscan':
            self.profile_df['n_clusters'] = [np.nan]*len(model_inputs["test_values"])
            self.profile_df['outlier_pct'] = [np.nan]*len(model_inputs["test_values"])
            
    def weighted_contingency_matrix(
        self, 
        labels_true, 
        labels_pred, 
        sample_weight=None
        ):
        """
        Compute a weighted contingency matrix.

        Parameters:
            labels_true: Ground truth labels.
            labels_pred: Cluster labels.
            sample_weight: Optional weight for each sample.

        Returns:
            Weighted contingency matrix.
        """
        labels_true = np.array(labels_true)
        labels_pred = np.array(labels_pred)

        # If no sample weights provided, assume equal weight (1 per sample)
        if sample_weight is None:
            sample_weight = np.ones_like(labels_true, dtype=float)

        # Find unique clusters
        unique_true = np.unique(labels_true)
        unique_pred = np.unique(labels_pred)

        # Create an empty contingency matrix
        cont_matrix = np.zeros((len(unique_true), len(unique_pred)))

        # Compute weighted contingency counts
        for i, true_label in enumerate(unique_true):
            for j, pred_label in enumerate(unique_pred):
                mask = (labels_true == true_label) & (labels_pred == pred_label)
                cont_matrix[i, j] = np.sum(sample_weight[mask])  # Weighted count

        return cont_matrix

    def weighted_adjusted_rand_index(
        self, 
        labels_true, 
        labels_pred, 
        sample_weights=None
        ):
        # Convert labels to numpy arrays
        labels_true = np.asarray(labels_true)
        labels_pred = np.asarray(labels_pred)

        n = len(labels_true)
        
        if sample_weights is None:
            sample_weights = np.ones(n)  # Default to equal weights

        # Compute weighted contingency table
        contingency = self.weighted_contingency_matrix(labels_true, labels_pred, sample_weight=sample_weights)

        # Compute weighted sums for the formula
        sum_comb_c = np.sum(contingency * (contingency - 1)) / 2  # Sum of combinations for clusters
        sum_comb_a = np.sum(np.sum(contingency, axis=1) * (np.sum(contingency, axis=1) - 1)) / 2
        sum_comb_b = np.sum(np.sum(contingency, axis=0) * (np.sum(contingency, axis=0) - 1)) / 2
        w_sum = np.float64(np.sum(sample_weights))
        sum_comb_n = w_sum * (w_sum - 1) / 2  # Total weighted pairs

        expected_index = (sum_comb_a * sum_comb_b) / sum_comb_n  # Expected index
        max_index = (sum_comb_a + sum_comb_b) / 2  # Maximum index

        if max_index == expected_index:
            return 1.0  # Avoid division by zero, perfect match

        weighted_ARI = (sum_comb_c - expected_index) / (max_index - expected_index)
        return weighted_ARI
            
    def adjusted_rand_index(
        self, 
        t,
        boostraps, 
        sample_size,
        weight
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
        
        # Enumerate to avoid testing the similarity to the same bootstrap samples
        num=0
        for l, i in zip(labels, indices):
            num = num+1
            for k, j in zip(labels[num:], indices[num:]):
                in_both = np.intersect1d(i, j)
#                scores.append(adjusted_rand_score(l[in_both], k[in_both]))
                scores.append(self.weighted_adjusted_rand_index(labels_true=l[in_both], labels_pred=k[in_both], sample_weights=weight[in_both]))
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
#            ari.append(adjusted_rand_score(orig_model.labels_, new_model.labels_))
            ari.append(self.weighted_adjusted_rand_index(labels_true=orig_model.labels_, labels_pred=new_model.labels_, sample_weights=weights))
            
        return np.mean(ari)
        
    def bootstrap_std_scaled_wcss(
        self, 
        t, 
        bootstraps, 
        sample_size
        ):

        args = self.model_args
        args[self.indep_param_nm] = t
        
        data = self.train_data.sample(frac=sample_size)
        
        scores = []
        for _ in range(bootstraps):
            data_sample = resample(data, n_samples=len(data), replace=True)
            X_sample = data_sample.iloc[:, :-1]
            weights_sample = data_sample.iloc[:, -1]
            m = self.model(**args)
            m.fit(X_sample, sample_weight=weights_sample)
    #        scores.append(m.inertia_ / len(X_sample))
            scores.append(sum(weights_sample.values[i] * np.linalg.norm(X_sample.values[i] - m.cluster_centers_[m.labels_[i]])**2 for i in range(len(X_sample))) / len(X_sample))
       
        return np.std(scores)
        
        
    def weighted_calinski_harabasz_score(
        self, 
        X, 
        labels, 
        sample_weights=None
        ):
            
        unique_labels = np.unique(labels)
        k = len(unique_labels)  # Number of clusters
        n = X.shape[0]  # Number of samples

        if sample_weights is None:
            sample_weights = np.ones(n)  # Default to equal weights
        
        overall_mean = np.average(X, axis=0, weights=sample_weights)  # Weighted overall mean

        # Compute cluster-wise statistics
        B_k = 0  # Between-cluster scatter
        W_k = 0  # Within-cluster scatter

        for label in unique_labels:
            cluster_points = X[labels == label]
            cluster_weights = sample_weights[labels == label]
            cluster_size = np.sum(cluster_weights)

            if cluster_size == 0:
                continue

            # Compute weighted centroid
            cluster_mean = np.average(cluster_points, axis=0, weights=cluster_weights)

            # Between-cluster scatter
            B_k += cluster_size * np.sum((cluster_mean - overall_mean) ** 2)

            # Within-cluster scatter
            W_k += np.sum(cluster_weights[:, None] * (cluster_points - cluster_mean) ** 2)

        # Compute the weighted CH score
        CH_score = (B_k / (k - 1)) / (W_k / (n - k))
        return CH_score
        
    def weighted_davies_bouldin_score_exact(
        self, 
        X, 
        labels, 
        sample_weights):
        """
        Calculates the weighted Davies-Bouldin index for a given dataset and its cluster assignments.

        Args:
            X: The dataset, where each row is a data point.
            labels: The cluster assignments for each data point.
            sample_weights: An array of weights for each data point.

        Returns:
            The weighted Davies-Bouldin index.
        """

        n_samples = X.shape[0]
        n_clusters = len(np.unique(labels))

        # Calculate weighted cluster centers
        cluster_centers = np.zeros((n_clusters, X.shape[1]))
        cluster_sizes = np.zeros(n_clusters)
        for i in range(n_samples):
            cluster_centers[labels[i]] += X[i] * sample_weights[i]
            cluster_sizes[labels[i]] += sample_weights[i]
        cluster_centers /= cluster_sizes[:, np.newaxis]

        # Calculate weighted within-cluster distances
        within_cluster_distances = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            cluster_weights = sample_weights[labels == i]
            for j in range(len(cluster_points)):
                within_cluster_distances[i] += cluster_weights[j] * np.linalg.norm(cluster_points[j] - cluster_centers[i])
            within_cluster_distances[i] /= np.sum(cluster_weights)

        # Calculate Davies-Bouldin scores for each cluster
        db_scores = []
        for i in range(n_clusters):
            max_ratio = 0
            for j in range(n_clusters):
                if i != j:
                    distance_between_centroids = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    ratio = (within_cluster_distances[i] + within_cluster_distances[j]) / distance_between_centroids
                    max_ratio = max(max_ratio, ratio)
            db_scores.append(max_ratio)

        # Calculate weighted average Davies-Bouldin score
        weighted_db_score = np.average(db_scores, weights=cluster_sizes)

        return weighted_db_score
    
    def weighted_davies_bouldin_score_fast(
        self, 
        X, 
        labels, 
        sample_weights=None):
            
        unique_labels = np.unique(labels)
        k = len(unique_labels)  # Number of clusters
        n = X.shape[0]  # Number of samples

        if sample_weights is None:
            sample_weights = np.ones(n)  # Default to equal weights

        # Compute weighted centroids
        centroids = np.array([
            np.average(X[labels == label], axis=0, weights=sample_weights[labels == label])
            for label in unique_labels
        ])

        # Compute weighted intra-cluster distances (Si)
        S = np.zeros(k)
        for i, label in enumerate(unique_labels):
            cluster_points = X[labels == label]
            cluster_weights = sample_weights[labels == label]
            cluster_size = np.sum(cluster_weights)

            if cluster_size == 0:
                continue

            # Compute weighted mean distance to centroid
            S[i] = np.average(np.linalg.norm(cluster_points - centroids[i], axis=1), weights=cluster_weights)

        # Compute centroid distances (Mij)
        M = euclidean_distances(centroids)

        # Compute Davies-Bouldin Index
        DB_scores = []
        for i in range(k):
            scores = [
                (S[i] + S[j]) / M[i, j] if i != j else 0
                for j in range(k)
            ]
            DB_scores.append(np.max(scores))

        return np.mean(DB_scores)
        
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
#        ss = round(silhouette_score(df, labels), 2)
        # The calculation of the weighted Silhouette Score is approximate
        silhouette_vals = silhouette_samples(df, labels)
        ss = round(np.average(silhouette_vals, weights=self.weights), 2)
        # Calinski Harabasz score 
#        cs = round (calinski_harabasz_score(df, labels), 2)
        # The calculation of the weighted Calinski Harabasz Score is approximate
        cs = round(self.weighted_calinski_harabasz_score(df.values, labels, np.array(self.weights)), 2)
        # Davies Bouldin score 
#        db = round(davies_bouldin_score(df, labels), 2)
        db = round(self.weighted_davies_bouldin_score_fast(df.values, labels, np.array(self.weights)), 2)
        # Adjusted Rand Index 
        ari = round(self.adjusted_rand_index(t, bootstraps, sample_size, weight = np.array(self.weights)), 2)
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
            
            df = self.df['data_{}'.format(self.sample_values[0])].values
            labels = m.labels_
            weights_array = self.weights.values
            
            # Add metrics to profile_df
            if len(set(labels)) - (1 if -1 in labels else 0) > 1:
                metrics = self.get_metrics(labels, t, bootstraps, sample_size)
                if self.model_nm.lower() == 'kmeans': 
#                    metrics += (round(m.inertia_, 2), )
                    metrics += (round(sum(weights_array[i] * np.linalg.norm(df[i] - m.cluster_centers_[labels[i]])**2 for i in range(len(df))), 2), )
                    metrics += (round(self.cluster_stability_kmeans(t, bootstraps, sample_size), 2), )
                    metrics += (round(self.bootstrap_std_scaled_wcss(t, bootstraps, sample_size), 2), )
                if self.model_nm.lower() == 'dbscan': 
                    metrics += (len(set(labels)) - (1 if -1 in labels else 0), )
                    metrics +=(np.round(np.sum(labels == -1) / len(labels), 3), )
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
#                self.valid_df.loc[self.valid_df.Split == split, "Scaled WCSS"] = round(self.fit_model.score(df) * -1 / len(labels), 3)
                self.valid_df.loc[self.valid_df.Split == split, "Scaled WCSS"] = round(sum(self.weights.values[i] * np.linalg.norm(df.values[i] - self.fit_model.cluster_centers_[labels[i]])**2 for i in range(len(df.values))) / np.sum(self.weights.values), 3)
                
            for j in np.unique(labels):
                self.valid_df.loc[self.valid_df.Split == split, "Cluster {} Size".format(j+1)] = round((labels == j).mean(), 3)
                
        return self.valid_df



        # Silhouette score 
#        ss = round(silhouette_score(df, labels), 2)
        # The calculation of the weighted Silhouette Score is approximate
#        silhouette_vals = silhouette_samples(df, labels)
#        ss = round(np.average(silhouette_vals, weights=self.weights), 2)
        # Calinski Harabasz score 
#        cs = round (calinski_harabasz_score(df, labels), 2)
        # The calculation of the weighted Calinski Harabasz Score is approximate
#        cs = round(self.weighted_calinski_harabasz_score(df.values, labels, np.array(self.weights)), 2)
        # Davies Bouldin score 
#        db = round(davies_bouldin_score(df, labels), 2)
#        db = round(self.weighted_davies_bouldin_score_fast(df.values, labels, np.array(self.weights)), 2)
        # Adjusted Rand Index 
#        ari = round(self.adjusted_rand_index(t, bootstraps, sample_size, weight = np.array(self.weights)), 2)





#############################################################################################################################################
#############################################################################################################################################

def silhouette_score_with_faiss(
    X, 
    labels
    ):
        
    """
    Calculates the silhouette score for a given dataset using FAISS. This function is not used in the solution because it takes a long time to compute. 

    Args:
        X: Numpy array. The dataset, where each row is a data point.
        labels: Numpy array. The cluster assignments for each data point.

    Returns:
        The average silhouette score for all data points using FAISS.
    """

    import time
    start_time = time.time()
    import faiss
    from sklearn.metrics import pairwise_distances
    
    # Convert data to FAISS index for fast nearest neighbor search
    X_faiss = np.ascontiguousarray(X.astype('float32'))  # Required format for Faiss
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X_faiss)
    
    # Find nearest neighbor distances
    k = len(X)  # Number of nearest neighbors to retrieve
    distances, indices = index.search(X_faiss, k)  # Find 2 nearest neighbors (first is itself)
    
    ###############################
    import pandas as pd 
    df = pd.DataFrame(X)
    df['labels'] = labels
    ###############################
    
    silhouette_score_list = []
    for row in range(len(df)):
        # Compute alpha
        count_table = pd.DataFrame(df['labels'].value_counts()).reset_index()
        count_row = count_table[count_table['labels'] == df['labels'][row]]['count'].values[0]
        alpha_i = np.mean(distances[row][1:count_row])
    
        # Compute beta
        distances_to_other_clusters = []
        for cluster in count_table['labels']:
            if cluster != df['labels'][row]:
                #print(np.where(df['labels'] == cluster)[0].tolist())
                distances_index = [i for i, x in enumerate(indices[row]) if x in np.where(df['labels'] == cluster)[0].tolist()]
                np.mean(distances[row][distances_index])
                distances_to_other_clusters.append(np.mean(distances[row][distances_index]))
        beta_i = np.min(distances_to_other_clusters)
    
        # Compute Silhouette score
        silhouette_score_i = (beta_i - alpha_i) / max(alpha_i, beta_i)
        silhouette_score_list.append(silhouette_score_i)
    silhouette_score_faiss = np.mean(silhouette_score_list)
    print("Approximate Silhouette Score with Faiss:", silhouette_score_faiss)
    print('Faiss took %.2fs. to run'%(time.time() - start_time))
    
    return silhouette_score_faiss

#############################################################################################################################################
#############################################################################################################################################

def weighted_silhouette_score(X, labels, sample_weights):
    """
    Calculates the weighted silhouette score for a given dataset and its cluster assignments. Although this calculation is accurate, it is not used in the solution because it takes a very long time to compute.  

    Args:
        X: The dataset, where each row is a data point.
        labels: The cluster assignments for each data point.
        sample_weights: An array of weights for each data point.

    Returns:
        The weighted average silhouette score for all data points.
    """

    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    silhouette_scores = []

    for i in range(n_samples):
        label_i = labels[i]
        weight_i = sample_weights[i]

        # Calculate a: weighted average distance to other points within the same cluster
        a = 0
        cluster_i_points = X[labels == label_i]
        cluster_i_weights = sample_weights[labels == label_i]
        for j in range(len(cluster_i_points)):
            if j != i:
                a += weight_i * np.linalg.norm(X[i] - cluster_i_points[j])
        a /= np.sum(cluster_i_weights) - weight_i

        # Calculate b: weighted average distance to the nearest cluster
        b = float('inf')
        for k in unique_labels:
            if k != label_i:
                cluster_k_points = X[labels == k]
                cluster_k_weights = sample_weights[labels == k]
                distances_to_k = np.linalg.norm(cluster_k_points - X[i], axis=1)
                b_k = np.average(distances_to_k, weights=cluster_k_weights)
                b = min(b, b_k)

        # Calculate silhouette score for this data point
        silhouette_scores.append(weight_i * ((b - a) / max(a, b)))

    return np.sum(silhouette_scores) / np.sum(sample_weights), silhouette_scores            
    

            
#############################################################################################################################################
#############################################################################################################################################
