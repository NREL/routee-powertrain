from __future__ import annotations

import numpy as np
from pandas import DataFrame, Series
from sklearn import linear_model

from powertrain.core.features import FeaturePack
from powertrain.estimators.estimator_interface import EstimatorInterface


class LinearRegression(EstimatorInterface):
    """linear regression routee estimator.
    
    This estimator uses a linear model to predict
    route energy usage.
    
    """

    def __init__(
            self,
            feature_pack: FeaturePack,
            model: linear_model.LinearRegression = linear_model.LinearRegression()
    ):
        self.feature_pack = feature_pack
        self.model = model

    def train(self,
              data: DataFrame,
              ):
        """
        train method for the base estimator (linear regression)
        Args:
            data:

        Returns:

        """
        # convert absolute consumption to rate consumption
        energy_rate = data[self.feature_pack.energy.name] / data[self.feature_pack.distance.name]

        x = data[self.feature_pack.feature_list]
        y = energy_rate

        self.model = self.model.fit(x.values, y.values)

    def predict(self, data: DataFrame) -> Series:
        """Apply the estimator to to predict consumption.

        Args:
        data (pandas.DataFrame):
            Columns that match self.features and self.distance that 
            describe vehicle passes over links in the road network.

        Returns:
            target_pred (float): 
                Predicted target for every row in links_df. 
                
        """
        x = data[self.feature_pack.feature_list]
        _energy_pred_rates = self.model.predict(x.values)
        _energy_pred = _energy_pred_rates * data[self.feature_pack.distance.name]

        energy_pred = Series(_energy_pred)

        return energy_pred

    def to_json(self) -> dict:
        serialized_model = {
            'meta': self.model.__class__.__name__,
            'coef_': self.model.coef_.tolist(),
            'intercept_': self.model.intercept_.tolist(),
            'params': self.model.get_params()
        }
        out_json = {
            'model': serialized_model,
            'feature_pack': self.feature_pack.to_json(),
        }

        return out_json

    @classmethod
    def from_json(cls, json: dict) -> LinearRegression:
        model_dict = json['model']
        model = linear_model.LinearRegression(**model_dict['params'])
        model.coef_ = np.array(model_dict['coef_'])
        model.intercept_ = np.array(model_dict['intercept_'])

        feature_pack = FeaturePack.from_json(json['feature_pack'])

        return LinearRegression(feature_pack=feature_pack, model=model)
