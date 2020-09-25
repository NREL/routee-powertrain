'''
# RouteE Tutorial: Feature_Selection Module
This notebook details the functionalities of the Feature_Selection module. The Feature_Selection module is a preprocessor which enables the user to analyze and select the best set of training features. This module has three main submodules to select the desired feature:
1. Filter Method: The user will be able to investigate the feature correlation of the training data and select the highly correlated set of features.
2. Iterative Method: The user will be able to request the set of "n" number of features that yields highest performance (e.g., cv score) with a given training model.
3. Optimal Method: Based on the training data and intendended training model, the user can request the optimal set of features that maximizes the training performance.
Each of the above submodules are associated with hyperparameters that can be modified to get the intended analysis and result. In this notebook, a detailed walkthrough of all of the functionalities of the above modules and submodules are documented.
'''

#-----------------------------------------------------------------------------
'''
# Loading the Modules and Reading Data
The Feature_Selection module resides inside the feature_engineering folder. To import the module, use--> from routee.core.model import Feature_Selection
'''
PATH_TO_ROUTEE = '../../../routee-powertrain/'
#PATH_TO_ROUTEE = '/projects/aes4t/amahbub/RouteE_git/routee/routee/'
PICKLE_OUT_PATH = "data/"

import pandas as pd
import sys
sys.path.append(PATH_TO_ROUTEE)
import sklearn
import numpy as np
import sqlalchemy as sql
import matplotlib.pyplot as plt
import time
%matplotlib inline
import seaborn as sn

import warnings
warnings.filterwarnings('ignore')

import routee
from routee.estimators import ExplicitBin, RandomForest, XGBoost
from routee.core.model import Feature, Distance, Energy
from routee.core.feature_selection import Feature_Selection #importing the Feature_Selection Module

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
#---------------------------------------------------------------------------
'''
## Load FastSIM Data
1. Based on the repo location, activate either of the following paths:
-    FASTSIM_RESULTS (local repo) = "C:/Users/amahbub/Documents/Data_RouteE/data_tomtom_new/2016_TOYOTA_Camry_4cyl_2WD_fs_results.csv"
-   FASTSIM_RESULTS (Eagle HPC repo) = "/projects/aes4t/jholden/data/fastsim_results/2016_TOYOTA_Camry_4cyl_2WD_fs_results.csv"
2. Provide the feature list --> Training + Target feature
3. Provide the # of samples to read. Provide "None" if all samples are to be read
'''
def get_df(nrows=None, feature_list=None):
    #FASTSIM_RESULTS = "C:/Users/amahbub/Documents/Data_RouteE/data_tomtom_new/2016_TOYOTA_Camry_4cyl_2WD_fs_results.csv"
    FASTSIM_RESULTS = "/projects/aes4t/jholden/data/fastsim_results/2016_TOYOTA_Camry_4cyl_2WD_fs_results.csv"
    df = pd.read_csv(FASTSIM_RESULTS, nrows= nrows, usecols=feature_list)
    return df

feature_list=['gge','miles', 'meters', 'seconds', 'grade', 'frc', 'gpsspeed', 'minutes',
                   'net2class','kph','speedcat','fow']
start_time = time.time()
df_passes = get_df(nrows = 1000, feature_list=feature_list)
end_time = time.time()
print(f"Function execution time: {end_time-start_time}.")

#----------------------------------------------------------------------------------

'''
## Preprocessing the input dataframe
'''

df_mod = df_passes[feature_list]
def remove_nan(df_passes, nan_threshold=1000000):
    columns = df_passes.columns
    nan_list = df_passes.isna().sum() #get a df of nan values sums
    val = (nan_list>nan_threshold).values #make a list of nan val greater than the threshold, i.e. inadmissible features --> 1M
    df_passes = df_passes.drop(columns=columns[val]) #column list defined above
    df_passes = df_passes.dropna()
    print(columns[val])
    return df_passes
df_mod = remove_nan(df_mod)

print("Samples x Features")
print("----------------------")
print(df_mod.shape)

print("Checking NaN values")
print("----------------------")
print(df_mod.isna().sum())
df_mod.head()

#-------------------------------------------------------------------------------

'''
# Module: Feature_Selection
This is the preprocessing module for feature engineering.
1. Args:
    - df:
       Pandas dataframe containing the sample data.
    - feature_list:
        List of all relevant features --> Training + target features.
2. Returns:
    - Object to access the submodules (feature selectors) --> get_correlation, get_n_features, get_optimal_features
'''
feat_obj = Feature_Selection(df_mod, feature_list)
#-------------------------------------------------------------------------------

'''
# 1. Filter Method: Feature_Selector --> Correlation
The get_correlation(args) method can be called to investigate the correlation among different features. This analysis provides a preliminary intuition about the set of possible features to train the core RouteE model.
The arguments/parameters of the get_correlation(args) method can be used obtain desired result.
1. Args:
    - method [str]--> 'pearson', 'kendall', 'spearman'
    - target_feature [list of str]-->
        List of all relevant features --> Training + target features.
    - threshold [float, 0~1]--> Set the threshold of the colinearity. E.g., for filtering features with 50% or more colinearity, provide threshold = 0.5. The default value is set to be 0.4.
    - multicolinearity [Bool]--> True/ False
        If True, prints the number of highly colinear features. Provides the information as to whether a feature has other colinear features. Highly colinear features can be reduces to independent set of features to improve model performance and computational complexity.
    - show_correlation [bool]--> True/False
        If True, shows the correlation table of the provided features.
'''

'''
### Example 1: Find the pearson correlation of all the features against the target feature 'gge' with >=0.4 correlation. 
- Note, the multicolinearity and show_correlation parameters are turned off.
- The target feature 'gge' has correlation 1.0 with itself.
'''
feat_obj.get_correlation(method = 'pearson', target_feature = 'gge', threshold = 0.4, multicolinearity = False, show_correlation = False)

#--------------------------------------

'''
### Example 2: If mulcolinearity is to be investigated, put multicolinearity = True. We can then get the multicolinear feature sets by setting the appropriate target_feature.
- Note, the feature 'frc' has 4 highly colinear features (>=0.4). The highly colinear set of features are = ['frc', 'gpsspeed', 'net2class', 'kph', 'speedcat'], which can be obtained by setting target_feature = 'frc'
- To improve the model performance and computational effort, we can reduce the above set to a number of features that has high correlation with the target_feature (in this case, 'gge').
'''
feat_obj.get_correlation(method = 'pearson', target_feature = 'frc', threshold = 0.4, multicolinearity = True, show_correlation = False)

#-------------------------------------------------
'''
Example 3: We can obain the correlation table of all the features by turning show_correlation = True.¶
Here, the 'kendall' correlation table is shown.
Based on the number of features, the figure size can be changed in the feature_selection.py script manually.
'''
feat_obj.get_correlation(method = 'kendall', target_feature = 'frc', threshold = 0.4, multicolinearity = False, show_correlation = True)

#--------------------------------------------------
'''
# 2. Wrapper Method --> Sequential_feature_search (SFS)
The "get_n_features(args)" method can be called to deploy a wrapper method that returns the best "n" number of features based on the given training model.
1. Args:
    - target_feature [str] --> The target feature name.
    - n_features [int] --> Number of features to be analysed. E.g., the user may want to find out the "best 3 features" to train the model.
    - estimator [estimator object] --> Estimator object based on scikit-learn regressor models, e.g., RandomForestRegressor() for random forest regressor. Any of the scikit-learn regressors can be provided with a proper preamble of importing the regressor module (see example 2.1).
    - search_method [str] --> The search method to be employed for the sequential feature search. Currently, there are six options.
           1. sfs: Sequential forward selection
           2. fsfs: Floating sequential forward selection
           3. sbe: Sequential backward elimination
           4. fsbe: Floating sequential backward elimination
           5. rfe: Recursive feature elimination
           6. all: Runs all of the above methods 
2. Returns:
    - prints a set of best "n" number of features.
    - prints the corresponding CV score.
    - shows a plot of the recursive search procedure (except 'rfe' method).
'''
'''
## 2.1 SFS method from scikit-learn: Recursive Feature Elimination (RFE)
The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
### Example: Get the best 3 features with target feature as 'gge' and search method as 'rfe'
'''

from sklearn.ensemble import GradientBoostingRegressor
feat_obj.get_n_features(target_feature = 'gge', n_features = 3, estimator = RandomForestRegressor(), search_method = 'rfe')

#------------------------------

'''
## 2.2 SFS methods from mlxtend package
How is this different from Recursive Feature Elimination (RFE) -- e.g., as implemented in sklearn.feature_selection.RFE? RFE is computationally less complex using the feature weight coefficients (e.g., linear models) or feature importance (tree-based algorithms) to eliminate features recursively, whereas SFSs eliminate (or add) features based on a user-defined classifier/regression performance metric.
### Example: Get the best 3 features with target feature as 'gge' and search method as 'sfs'
'''
feat_obj.get_n_features(target_feature = 'gge', n_features = 3, estimator = GradientBoostingRegressor(), search_method = 'sfs')

#----------------------------------------
'''
## 2.3 All SFS/RFE Feature Search
If the user is unsure about which feature search method (sfs/sbe/rfe etc.) to select, setting search_method to 'all' provides a summary of the results of all the available sequential feature search methods. Based on the most recurring set of features, the user can select the preferred set of training features.
### Example: Get the best 3 features with target feature as 'gge' and search method as 'all'
'''
feat_obj.get_n_features(target_feature = 'gge', n_features = 3, estimator = RandomForestRegressor(), search_method = 'all')

#--------------------------------------------
'''
# 3. Optimal Feature Search
Performs sequential feature search on all possible combinations of feature sets and selects the one with highest CV score. The user should call the "get_optimal_features(args)" method of Feature_Selection object to perform this functionality. The parameters of "get_optimal_features(args)" method is provided below:
1. Args:
    - target_feature [str] --> The target feature name.
    - estimator [estimator object] --> Estimator object based on scikit-learn regressor models, e.g., RandomForestRegressor() for random forest regressor. Any of the scikit-learn regressors can be provided with a proper preamble of importing the regressor module (see example 2.1). The default training model is set as RandomForestRegressor() [see example 3.1].
    - search_method [str] --> The search method to be employed for the sequential feature search. Currently, there are six options.
           1. sfs: Sequential forward selection
           2. fsfs: Floating sequential forward selection
           3. sbe: Sequential backward elimination
           4. fsbe: Floating sequential backward elimination
           5. rfe: Recursive feature elimination
           6. all: Runs all of the above methods 
2. Returns:
    - prints a set of best "n" number of features.
    - prints the corresponding CV score.
    - shows a plot of the recursive search procedure (except 'rfe' method).
'''
'''
## 3.1 Optimal RFECV method:
By setting the search method as 'rfe', the user can deploy the scikit-learn's recursive feature selector (rfe) module to obtain the optimal set of features.
### Example: Optimal Feature Search (RFECV)
'''
feat_obj.get_optimal_features(target_feature = 'gge', estimator = RandomForestRegressor(), search_method = 'rfe')
#------------------------------
'''
## 3.2 Optimal SFS method:
By setting the search method to any of the sequential feature selector methods, the user can deploy the mlxtends's the sequential feature selector (sfs) modules to obtain the optimal set of features.
### Example: Optimal Feature Search (SFS)
'''
feat_obj.get_optimal_features(target_feature = 'gge', estimator = RandomForestRegressor(), search_method = 'fsfs')