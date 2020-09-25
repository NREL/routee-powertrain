import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from routee.estimators.base import BaseEstimator

import xgboost as xgb


class XGBoost(BaseEstimator):
    """This estimator uses a xgboost tree to select an optimal decision tree,
    meant to serve as an automated construction of a lookup table.

    Example application:
        > import routee
        > from routee.estimators import RandomForest
        >
        >
        > model_rf = routee.Model(
        >                '2016 Ford Explorer',
        >                estimator = RandomForest(cores = 2),
        >                )
        >
        > model_rf.train(fc_data, # fc_data = link attributes + fuel consumption
        >               energy='gallons',
        >               distance='miles',
        >               trip_ids='trip_ids')
        >
        > model_rf.predict(route1) # returns route1 with energy appended to each link
        
    Args:
        cores (int):
            Number of cores to use during traing.
            
    """

    def __init__(self, cores):
        self.cores = cores

    def train(self, x, y):
        """Method to train the estimator over a specific dataset.
        Overrides BaseEstimatortrain method.

        Args:
            train (pandas.DataFrame):
                Data frame of the training data including features and target.
            test (pandas.DataFrame):
                Data frame of testing data for validation and error calculation.
            metadata (dict):
                Dictionary of metadata including features, target, distance column,
                energy column, and trip_ids column.

        Returns
            errors (dict):
                Dictionary with error metrics.
                
        """

        regmod = xgb.XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3
        )

        self.model = regmod.fit(np.array(x), np.array(y)) #trained regressor model


    #add feature importance
    def feature_importance(self):
        return self.model.feature_importances_
        
    def plot_feature_importance(self, features = None):
        self.model.feature_names = features
        xgb.plot_importance(self.model)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.show()
        
        
        (pd.Series(self.model.feature_importances_, index=features)
        .nlargest(10)
        .plot(kind='barh'))
