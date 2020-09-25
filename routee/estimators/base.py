from sklearn import linear_model
import numpy as np


class BaseEstimator:
    """Base class for a RouteE estimator. 
    
    This estimator uses a linear model to predict
    route energy usage.
    
    """

    def __init__(self):
        self.model = None

    def train(self, x, y):
        """Method to train the estimator over a specific dataset

        Args:
            train (pandas.DataFrame):
                Data frame of the training data including features and target.
            test (pandas.DataFrame):
                Data frame of testing data for validation and error calculation.

        Returns:
            errors (dict):
                Dictionary with error metrics.
                
        """
        regmod = linear_model.LinearRegression()
        self.model = regmod.fit(x, y)

    def predict(self, x):
        """Apply the estimator to to predict consumption.

        Args:
        links_df (pandas.DataFrame):
            Columns that match self.features and self.distance that 
            describe vehicle passes over links in the road network.

        Returns:
            target_pred (float): 
                Predicted target for every row in links_df. 
                
        """
        target_pred = self.model.predict(np.array(x)) #modified: feed np.array(x) to predict()
        target_pred = target_pred.clip(min=0)

        return target_pred
