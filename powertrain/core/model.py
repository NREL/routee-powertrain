import pickle

import numpy as np
from pandas import DataFrame

from powertrain.core.utils import test_train_split, get_version
from powertrain.estimators.estimator_interface import EstimatorInterface
from powertrain.validation import errors


class Model:
    """This is the core model for interaction with the routee engine.

    Args:
        veh_desc (str):
            Unique description of the vehicle to be modeled.
        estimator (routee.estimator.base.BaseEstimator):
            Estimator to use for predicting route energy usage.
            
    """

    def __init__(self, veh_desc: str, estimator: EstimatorInterface):
        self.metadata = {'veh_desc': veh_desc}
        self._estimator = estimator
        self.errors = None

    def train(
            self,
            data: DataFrame,
    ):
        """
        Train a model

        Args:
            data:

        Returns:

        """
        print(f"training estimator {self._estimator} with option {self._estimator.predict_type}.")

        self.metadata['estimator'] = self._estimator.__class__.__name__
        self.metadata['routee_version'] = get_version("powertrain/__init__.py")

        pass_data = data.copy(deep=True)
        pass_data = pass_data[~pass_data.isin([np.nan, np.inf, -np.inf]).any(1)]

        # splitting test data between train and validate --> 20% here
        train, test = test_train_split(pass_data.dropna(), 0.2)

        self._estimator.train(pass_data)

        self.validate(test)

    def validate(self, test):
        """Validate the accuracy of the estimator.

        Args:
            test (pandas.DataFrame):
                Holdout test dataframe for validating performance.
                
        """

        _target_pred = self.predict(test)
        test['target_pred'] = _target_pred
        self.errors = errors.all_error(
            test[self._estimator.feature_pack.energy.name],
            _target_pred,
            test[self._estimator.feature_pack.distance.name],
        )

    def predict(self, links_df):
        """Apply the trained energy model to to predict consumption.

        Args:
            links_df (pandas.DataFrame):
                Columns that match self.features and self.distance that describe
                vehicle passes over links in the road network.

        Returns:
            energy_pred (pandas.Series):
                Predicted energy consumption for every row in links_df.
                
        """
        return self._estimator.predict(links_df)

    def dump_model(self, outfile):
        """Dumps a routee.Model to a pickle file for persistance and sharing.

        Args:
            outfile (str):
                Filepath for location of dumped model.

        """
        out_dict = {
            'metadata': self.metadata,
            'estimator': self._estimator,
            'errors': self.errors,
        }

        pickle.dump(out_dict, open(outfile, 'wb'))
