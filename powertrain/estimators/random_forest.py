import numpy as np
import matplotlib.pyplot as plt

from powertrain.estimators.base import BaseEstimator

from sklearn.ensemble import RandomForestRegressor


class RandomForest(BaseEstimator):
    """This estimator uses a random forest to select an optimal decision tree,
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
        
        # Number of trees in random forest
        n_estimators = [int(est) for est in np.linspace(start=50, stop=1000, num=10)]

        # Maximum number of levels in tree
        max_depth = [int(d) for d in np.linspace(10, 110, num=11)]
        max_depth.append(None)

        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]

        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf}

        regmod = RandomForestRegressor(n_estimators=20,
                                       max_features='auto',
                                       max_depth=10,
                                       min_samples_split=10,
                                       n_jobs=self.cores,
                                       random_state=52)

        self.model = regmod.fit(np.array(x), np.array(y))

        #add feature importance
    def feature_importance(self):
        return self.model.feature_importances_
