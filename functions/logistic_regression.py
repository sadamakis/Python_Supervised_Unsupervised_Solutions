from decorators import time_function
import pandas as pd
import numpy as np
import statsmodels.api as sm
from io import StringIO
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report
from statsmodels.tools.tools import add_constant 

# Dictionary to save each summary table for each sample
glm_bin_summary = {}

@time_function
def glm_bin(
    sample_values_solution, 
    data, 
    final_feats, 
    target_variable_name, 
    weight_variable_name_solution
    ):

    i = sample_values_solution[0]
    df = data['data_{}'.format(i)]
    X = sm.add_constant(df[final_feats])
    Y = df[target_variable_name]
    
    # Build the model and fit the data
    glm_binom = sm.GLM(Y, X, family=sm.families.Binomial(), freq_weights=df[weight_variable_name_solution]).fit()
    
    summary_results = glm_binom.summary()
    results_as_csv = summary_results.tables[1].as_csv()
    results_str = StringIO(results_as_csv)
    
    glm_bin_summary['log_reg_summary_{}'.format(i)] = pd.read_csv(results_str, sep=',', skipinitialspace=True)
    glm_bin_summary['log_reg_summary_{}'.format(i)].columns = ['variable', 'coef', 'std_err', 'z', 'p_value', '[0.025', '0.975]']
    glm_bin_summary['log_reg_summary_{}'.format(i)]['variable'] = glm_bin_summary['log_reg_summary_{}'.format(i)]['variable'].str.strip()
    glm_bin_summary['log_reg_summary_{}'.format(i)]['statistically_significant'] = np.where(glm_bin_summary['log_reg_summary_{}'.format(i)]['p_value'] < 0.05, 'Yes', 'No')
    glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] = np.exp(glm_bin_summary['log_reg_summary_{}'.format(i)]['coef'])
    
    odds_ratio_condition = [
        (glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] < 1), 
        (glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] > 1), 
        (glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] == 1)]
        
    desc_value_rules = [1 - glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'], 
                        glm_bin_summary['log_reg_summary_{}'.format(i)]['odds_ratio'] - 1, 
                        1]
    glm_bin_summary['log_reg_summary_{}'.format(i)]['desc_value'] = np.select(odds_ratio_condition, desc_value_rules)
    return glm_binom, glm_bin_summary

# Dictionary to save each summary table for each sample
sub_summary = {}
disp_report = {}

@time_function
def glm_report(
    data_path, 
    glm_bin_summary
    ): 
    
    for i, j in glm_bin_summary.items():
        temp = glm_bin_summary[i]
        temp = temp.drop(['[0.025', '0.975]'], axis=1)
        display(temp)
        pd.DataFrame(temp).to_csv(data_path + '/output/' + str(i) + '.csv', index=False)
    return temp

@time_function
def get_evaluation(
    model, 
    sample_values_dict, 
    final_feats, 
    target_variable_name, 
    data, 
    data_path
    ):
    
    dataset = []
    TP = []
    FP = []
    TN = []
    FN = []
    auc = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    for i, j in sample_values_dict.items():
        dataset.append(i)
        df = data['data_{}'.format(i)]
        X = sm.add_constant(df[final_feats])
        Y = df[target_variable_name]
        pred = list(map(round, model.predict(X)))
        cm = confusion_matrix(Y, pred)
        auc.append(roc_auc_score(Y, pred))
        accuracies.append(accuracy_score(Y, pred))
        precisions.append(precision_score(Y, pred))
        recalls.append(recall_score(Y, pred))
        f1_scores.append(f1_score(Y, pred))
        TP.append(cm[0][0])
        FP.append(cm[0][1])
        TN.append(cm[1][1])
        FN.append(cm[1][0])
    eval_df = pd.DataFrame(dataset)
    eval_df['auc'] = auc
    eval_df['accuracy'] = accuracies
    eval_df['precision'] = precisions
    eval_df['recall'] = recalls
    eval_df['f1_score'] = f1_scores
    eval_df['true positives'] = TP
    eval_df['true negatives'] = TN
    eval_df['false positives'] = FP
    eval_df['false negatives'] = FN
    eval_df.rename(columns = {0: 'dataset'}, inplace = True)
    eval_df.to_csv(data_path + '/output/evaluation_metrics.csv', index=False)
    display(eval_df)


