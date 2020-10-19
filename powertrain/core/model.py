import pickle
import numpy as np
import matplotlib.pyplot as plt

from powertrain import __version__
from powertrain.validation import errors
from powertrain.core.utils import test_train_split
from powertrain.estimators.base import BaseEstimator
from powertrain.core.features import Feature, Distance, Energy



class Model:
    """This is the core model for interaction with the routee engine.

    Args:
        veh_desc (str):
            Unique description of the vehicle to be modeled.
        estimator (routee.estimator.base.BaseEstimator):
            Estimator to use for predicting route energy usage.
            
    """

    def __init__(self, veh_desc, option=2, estimator=BaseEstimator()):
        self.metadata = {'veh_desc': veh_desc}
        self._estimator = estimator
        self.errors = None
        self.option = option

    def train(
            self,
            data,
            features,
            distance,
            energy,
    ):
        """
        Train a model

        Args:
            data:
            features:
            distance:
            energy:

        Returns:

        """
        print(f"training estimator {self._estimator} with option {self.option}.")

        self.metadata['features'] = features
        self.metadata['distance'] = distance
        self.metadata['energy'] = energy
        self.metadata['estimator'] = self._estimator.__class__.__name__
        self.metadata['routee_version'] = __version__

        train_features = [feat.name for feat in features]

        # reduced local copy of the data based on features
        pass_data = data[train_features + [distance.name] + [energy.name]].copy()

        # convert absolute consumption for FC RATE
        pass_data[energy.name + '_per_' + distance.name] = (pass_data[energy.name]/pass_data[distance.name])

        pass_data = pass_data[~pass_data.isin([np.nan, np.inf, -np.inf]).any(1)]

        # splitting test data between train and validate --> 20% here
        train, test = test_train_split(pass_data.dropna(), 0.2)

        #training the models--> randomForest will have two options of feature selection

        # option 1: distance is not a feature and fc/dist is the target column,
        # option 2: distance is a feature, and fc is the target column
        if (self.metadata['estimator'] == 'ExplicitBin') or (self.option == 2):
            self._estimator.train(
                x=train[train_features+[distance.name]],
                y=train[energy.name],
            )

        else:
            self._estimator.train(
                x=train[train_features],
                y=train[energy.name + '_per_' + distance.name],
            )

        self.validate(test)

        #saving feature_importance
        if (self.metadata['estimator']=='RandomForest') or (self.metadata['estimator']=='XGBoost'):
            self.metadata['feature_importance'] = self._estimator.feature_importance()

    def plot_feature_importance(self):
        if (self.metadata['estimator'] == 'ExplicitBin'):
            print ("Feature importance not available for visualization.")
        else:
            features = [feat.name for feat in self.metadata['features']]
            if self.option == 2: features.append(self.metadata['distance'].name)
            self._estimator.plot_feature_importance(features)

    def validate(self, test):
        """Validate the accuracy of the estimator.

        Args:
            test (pandas.DataFrame):
                Holdout test dataframe for validating performance.
                
        """

        _target_pred = self.predict(test)
        test['target_pred'] = _target_pred
        self.errors = errors.all_error(
            test[self.metadata['energy'].name],
            _target_pred,
            test[self.metadata['distance'].name],
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

        for feat in self.metadata['features']:
            assert feat.name in links_df.columns, f"Missing expected column {feat.name} in links input"

        predict_features = [feat.name for feat in self.metadata['features']]

        if (self.metadata['estimator'] == 'ExplicitBin') or (self.option == 2):
            _energy_pred = self._estimator.predict(links_df[predict_features+[self.metadata['distance'].name]])

        else:
            _energy_pred_rates = self._estimator.predict(links_df[predict_features])
            _energy_pred = _energy_pred_rates * links_df[self.metadata['distance'].name]

        return _energy_pred

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
            'option': self.option,
        }

        pickle.dump(out_dict, open(outfile, 'wb'))