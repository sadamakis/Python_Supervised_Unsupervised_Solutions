# Supervised and Unsupervised Modeling solutions
This repository contains implementations of both supervised and unsupervised machine learning solutions for various tasks. The codes are fully automated to perform data cleansing, data transformation, dimensionality reduction, model development, and reports generation.
 
* **Data transformation** steps include: outlier replacement, one hot encoding, impute missing values, data standardization.
* **Dimensionality reduction** steps include: remove variables with high missing values percentage, remove character variables with many levels, drop numeric variables with only one value, drop variables based on low Gini, remove highly correlated features, remove features using VIF, Lasso Logistic Regression for feature selection, remove features based on p-value information, Factor Analysis, PCA transformation, Feature Importance. 
* **Reports** include: data quality report, Supervised and Unsupervised modeling reports that include tables and graphs to assess the quality of the models. 

## Key Contents

* **1_Fake_Data_Generation:** Jupyter Notebook to generate fake data. The code can be accessed in the ./src folder. 
* **2_Supervised_Modeling:** Includes models and code for classification, leveraging labeled datasets. Explore implementations of algorithms like Logistic Regression, Lasso Regression (Logistic_Regression folder) and Machine Learning models, e.g., Random Forests, Gradient Boosting, Neural Networks, lightGBM (Machine_Learning folder).
* **3_Unsupervised_Modeling:** Features algorithms designed to discover patterns and structures in unlabeled data. Find solutions for tasks like clustering, dimensionality reduction, and anomaly detection, with implementations of algorithms such as K-Means, DBSCAN, Principal Component Analysis (PCA), Factor Analysis.

## Purpose

This repository serves as a collection of practical implementations and examples demonstrating different approaches to machine learning problems. It can be used for:

* **Learning and understanding:** Providing clear and concise code examples for various supervised and unsupervised techniques.
* **Experimentation:** Offering a platform to test and compare different algorithms on various datasets.
* **Reference:** Serving as a quick reference for implementing common machine learning tasks.

## Structure

The repository is organized into logical directories (e.g., '1_Fake_Data_Generation/', '2_Supervised_Modeling/Logistic_Regression/', '2_Supervised_Modeling/Machine_Learning/', '3_Unsupervised_Modeling/') to facilitate easy navigation and understanding of the codebase. Each solution typically includes:

* 'data' folder that includes subfolders related to data that are used as input to the solutions, and data that are the outcome of the solutions. 
* 'environment_setup' folder that includes instructions how to set up the Python environment to execute the Jupyter Notebook files. 
* 'src' folder that includes Jupyter Notebooks for more detailed explanations, visualizations, and experimentation with the codes.

## Contributions

Contributions, including bug fixes, new implementations, and improvements to existing code, are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

**Keywords:** machine learning, supervised learning, unsupervised learning, classification, clustering, dimensionality reduction, Python, scikit-learn, lightGBM, TensorFlow, Keras, Optuna, H2O.