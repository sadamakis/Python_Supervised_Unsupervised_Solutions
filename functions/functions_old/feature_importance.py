from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from decorators import time_function

#COPIED 
class FeatureImportance:
    def __init__(
        self, 
        X, 
        labels, 
        weights, 
        data_path, 
        filename = 'FeatureImportance'
    ):
    
        self.X = X
        self.labels = labels
        self.weights = weights
        self.data_path = data_path
        self.filename = filename
        self.ordered_feature_names = X.columns.sort_values().tolist()
        self.feature_importance = pd.DataFrame()
        
    def get_feature_imp_unsup2sup(
        self, 
        df
    ):
    
        cluster_feature_weights = {}
        for label in set(self.labels):
            binary_enc = np.vectorize(lambda x: 1 if x==label else 0)(self.labels)
            clf = RandomForestClassifier(random_state=10)
            clf.fit(df, binary_enc, sample_weight = self.weights)
            
            sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1]
            
            ordered_cluster_features = np.take_along_axis(
                np.array(self.ordered_feature_names), 
                sorted_feature_weight_idxes, 
                axis=0)
            ordered_cluster_feature_weights = np.take_along_axis(
                np.array(clf.feature_importances_), 
                sorted_feature_weight_idxes, 
                axis=0)
            cluster_feature_weights[label] = list(zip(ordered_cluster_features, 
                                                        ordered_cluster_feature_weights))
        self.feature_importances = cluster_feature_weights
        return cluster_feature_weights
        
        
    def one_hot_encode(
        self, 
        columns
    ):
        
        if len(columns) == 0: 
            return pd.DataFrame()
        concat_df = pd.concat([pd.get_dummies(self.X[col], drop_first=True, prefix=col) for col in columns], axis=1)
        
        return concat_df
        
    @time_function
    def get_report(
        self
        ):
    
        if len(self.X.select_dtypes(include='object').columns) != 0: 
            cat_cols = self.X.select_dtypes(include='object')
            cat_one_hot_df = self.one_hot_encode(cat_cols.columns)
            df = self.X.join(cat_one_hot_df).drop(cat_cols, axis=1)
        else: 
            df = self.X
            
        cols = df.columns
        self.ordered_feature_names = cols.tolist()
        
        # Feature importance for each cluster 
        self.get_feature_imp_unsup2sup(df)
        # Save feature importances 
        for label in set(self.labels):
            pd.DataFrame(self.feature_importances[label]).to_csv(f'{self.data_path}/output/{self.filename}_feature_imprtnc{label}.csv', index=False)
            
        # Overall feature importance, scale by cluster weight (% population)
        unique_elements, counts_elements = np.unique(self.labels, return_counts=True)
        
        label_n_weight = pd.DataFrame(self.labels)
        label_n_weight = label_n_weight.rename(columns={0: "label"})
        label_n_weight['weight_variable'] = self.weights

        weighted_list = []
        for value in unique_elements:
            weighted_list.append(label_n_weight[label_n_weight['label']==value]['weight_variable'].sum() / label_n_weight['weight_variable'].sum())
        weight = dict(zip(unique_elements, weighted_list))        
        wght_feat_imp = pd.DataFrame({'Feature': np.sort(self.ordered_feature_names)})
        for label in set(self.labels):
            wght_feat_imp[label] = (pd.DataFrame(self.feature_importances[label]).sort_values(by=0)[1] * weight[label]).values
        wght_feat_imp.set_index('Feature', inplace=True)
        wght_feat_imp['overall_feature_importance'] = wght_feat_imp.sum(axis=1)
        
        res = pd.DataFrame(wght_feat_imp)
        res = res.sort_values(by='overall_feature_importance', ascending=False)
        res.to_csv(self.data_path + '/output/' + self.filename + '.csv')
        return res

    @time_function
    def feature_importance_keep_vars(
        self, 
        feature_importance_threshold
        ):

        fi_table = pd.read_csv('{0}/output/{1}.csv'.format(self.data_path, self.filename), sep=',')
        return list(fi_table[fi_table['overall_feature_importance'] > feature_importance_threshold]['Feature'])
