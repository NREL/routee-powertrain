#importing data processing and visualization modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#random forest regressor
from sklearn.ensemble import RandomForestRegressor

#optimal feature selection
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

#from sklearn.model_selection import train_test_split

#importing utility functions
from . import util_featureSelector as util


#defining utility functions to check feature correlations

class FeatureCorrelation:
    def __init__(self, df, correlation_features = None, method = 'pearson', threshold = 0.4):
        self.df = df
        self.correlation_features = correlation_features
        self.method = method
        self.threshold = threshold
    
        ### getting correlation matrix
        
        #isolating the relevant features
        if self.correlation_features == None:
            print("No correlation features selected!")
            return
        df = self.df[correlation_features]

        #Get correlation with output variable
        methods = ['pearson', 'kendall', 'spearman']
        if self.method in methods:
            self.cor = df.corr(method = self.method)
        else:
            print("Correlation method is not listed.")
            return

    def get_multicolinearity(self):
        #filter correlation by user-defined threshold
        print("----------------------------------------------")
        print("Feature Name       Highly correlated # of features")
        print("----------------------------------------------")
        for feature in self.correlation_features:
            cor_target = abs(self.cor[feature])
            #Selecting highly correlated features
            feat_num = 0
            for i in cor_target:
                if (i> self.threshold) and (i<1):
                    feat_num += 1
            print(f"{feature}:           {feat_num}")

    def get_correlated_features(self,target_feature):    
        #To see exactly which features:
        cor_target = abs(self.cor[target_feature])
        #Selecting highly correlated features
        relevant_features=cor_target[cor_target>self.threshold]
        print("----------------------------------------------")
        print(f"Features with high correlation (>={self.threshold}) are: {relevant_features}.")
        print("----------------------------------------------")
        return relevant_features

    def show_corr(self):
        plt.figure(figsize=(16,16))
        sns.heatmap(self.cor, annot=True, cmap=plt.cm.Reds)
        plt.show()
        
        
    #####################################################################################################    
    #####################################################################################################   
    #################################################################################################### 
class N_FeatureSearch:
    
    def __init__(self, df, target_feature = None, n_features=6, search_method = None, training_model = RandomForestRegressor()):
        self.df = df
        self.target_feature = target_feature
        self.n_features = n_features
        self.search_method = search_method
        self.training_model = training_model
        self.target_matrix = df[self.target_feature]
        self.feature_matrix = df.drop(target_feature, 1)
        
    def sequential_feature_search(self):  
        forward_val, floating_val = util.get_sfs_vals(self.search_method)

        sfs_rf = SFS(self.training_model,
                   k_features = self.n_features,
                   forward = forward_val,
                   floating = floating_val,
                   scoring = 'r2',#for regressions, 'f1'/'accuracy' for classification or use='neg_mean_squared_error'
                   cv = 0,
                    n_jobs = -1)
        sfs_rf.fit(self.feature_matrix, self.target_matrix)
        
        #output
        print(f"Training Model: {self.training_model.__class__.__name__}.")
        print(f"Sequential Feature Selection Method: {self.search_method}")
        print("---------------------------------------------")
        print(f"Best features: {sfs_rf.k_feature_names_}")
        print(f"CV Score: {sfs_rf.k_score_}")

        fig = plot_sfs(sfs_rf.get_metric_dict(), kind='std_err')
        plt.title(f"{self.search_method} (w. StdErr)")
        plt.grid()
        plt.show()

    def sfs_rfe(self):
        selector = RFE(self.training_model, n_features_to_select=self.n_features, step=1)
        selector = selector.fit(self.feature_matrix, self.target_matrix)
        feat_idx = selector.support_
        best_features = self.feature_matrix.columns[feat_idx]
        best_score = selector.score(self.feature_matrix, self.target_matrix)  #note: check it again!S
        #output
        print(f"Training Model: {self.training_model.__class__.__name__}.")
        print(f"Sequential Feature Selection Method: {self.search_method}")
        print("---------------------------------------------")
        print(f"Best features: {best_features}")
        print(f"CV Score: {best_score}")
        
    def all_sfs(self):
        selection_methods = ['sfs', 'sbe', 'fsfs', 'fsbe']
        best_features = []

        counter = 0
        for selection in selection_methods:
            forward_val, floating_val = util.get_sfs_vals(selection)
            sfs_rf = SFS(self.training_model,
                       k_features = self.n_features,
                       forward = forward_val,
                       floating = floating_val,
                       scoring = 'r2',#for regressions, 'f1'/'accuracy' for classification or use='neg_mean_squared_error'
                       cv = 0,
                        n_jobs = -1)
            sfs_rf.fit(self.feature_matrix, self.target_matrix)
            best_features.append(sfs_rf.k_feature_names_)
            
            #output
            print(f"Training Model: {self.training_model.__class__.__name__}.")
            print(f"Sequential Feature Selection Method: {selection}")
            print("---------------------------------------------")
            print(f"Best features: {sfs_rf.k_feature_names_}")
            print(f"CV Score: {sfs_rf.k_score_}")

            fig = plot_sfs(sfs_rf.get_metric_dict(), kind='std_err')
            plt.title(f"{selection} (w. StdErr)")
            plt.grid()
            plt.show()

            counter += 1
        print(f"Best features: {set(best_features)}.")
     #####################################################################################################   
     ##################################################################################################### 
    ## Optimal feature selector
class Optimal_FeatureSearch(N_FeatureSearch):
#     def __init__(self):
#         pass
        
    def optimal_feature_selector_RFE(self):
        features = self.feature_matrix.columns

        rfe = RFECV(self.training_model,cv = 5)      #default 5-fold cross-validation
        rfe.fit(self.feature_matrix, self.target_matrix)
        best_features = features[rfe.support_]
        n_best_features = len(best_features) # can also use--> rfe.n_features_
        score = rfe.score(self.feature_matrix, self.target_matrix) #Reduce X to the selected features and then return the score of the underlying estimator.
        
        #output
        print(f"Training Model: {self.training_model.__class__.__name__}.")
        print(f"Sequential Feature Selection Method: {self.search_method}")
        print("---------------------------------------------")
        print(f"Best Features: {best_features}")
        print(f"Best number of feature: {n_best_features}")
        print(f"Best score: {score}")

    def optimal_feature_selector_SFS(self):
        n_features = len(self.feature_matrix.columns)
        high_score = 0
        best_feature_number = 0
        best_features = []
        score_list = []
        forward_val, floating_val = util.get_sfs_vals(self.search_method)
#         for n in range(n_features):
#             sfs_rf = SFS(self.training_model,
#                        k_features = n_features,
#                        forward = forward_val,
#                        floating = floating_val,
#                        scoring = 'r2',#for regressions or use='neg_mean_squared_error'
#                        cv = 0,
#                         n_jobs = -1)
#             sfs_rf.fit(self.feature_matrix, self.target_matrix)
#             score = sfs_rf.k_score_ 
#             score_list.append(score)
#             if score>high_score:
#                 high_score = score
#                 best_model = sfs_rf

#         if len(best_model.k_feature_names_) ==0:
#             print("0 best features found!")
#             return
        
#         #output
#         print(f"Training Model: {self.training_model.__class__.__name__}.")
#         print(f"Sequential Feature Selection Method: {self.search_method}")
#         print("---------------------------------------------")
#         print(f"Best number of feature: {len(best_model.k_feature_names_)}")
#         print(f"Best Features: {best_model.k_feature_names_}")
#         print(f"Best score: {high_score}")

#         fig = plot_sfs(best_model.get_metric_dict(), kind='std_err')
#         plt.title(f"Optimal Backward Elimination (w. StdErr)")
#         plt.grid()
#         plt.show()
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        sfs_rf = SFS(self.training_model,
                       k_features = (1,n_features),
                       forward = forward_val,
                       floating = floating_val,
                       scoring = 'r2',#for regressions or use='neg_mean_squared_error'
                       cv = 0,
                        n_jobs = -1)
        
        pipe = make_pipeline(StandardScaler(), sfs_rf)

        pipe.fit(self.feature_matrix, self.target_matrix)


        #print('best combination (ACC: %.3f): %s\n' % (sfs_rf.k_score_, sfs_rf.k_feature_idx_))
        #output
        print(f"Training Model: {self.training_model.__class__.__name__}.")
        print(f"Sequential Feature Selection Method: {self.search_method}")
        print("---------------------------------------------")
        print(f"Best Features: {self.feature_matrix.columns[[sfs_rf.k_feature_idx_]]}")
        print(f"Best number of feature: {len(self.feature_matrix.columns[[sfs_rf.k_feature_idx_]])}")
        print(f"Best score: {sfs_rf.k_score_}.")

        
        
        
        plot_sfs(sfs_rf.get_metric_dict(), kind='std_err');
     #####################################################################################################        
    