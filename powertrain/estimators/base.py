from __future__ import annotations
from typing import Union

import numpy as np

from numpy import clip
from pandas import DataFrame, Series
from sklearn import linear_model

from powertrain.core.features import FeaturePack, PredictType
from powertrain.estimators.estimator_interface import EstimatorInterface


class BaseEstimator(EstimatorInterface):
    """Base class for a RouteE estimator. 
    
    This estimator uses a linear model to predict
    route energy usage.
    
    """

    def __init__(
            self,
            feature_pack: FeaturePack,
            predict_type: Union[str, int, PredictType] = PredictType.ENERGY_RAW,
            model: linear_model.LinearRegression = linear_model.LinearRegression()
    ):
        if isinstance(predict_type, str):
            ptype = PredictType.from_string(predict_type)
        elif isinstance(predict_type, int):
            ptype = PredictType.from_int(predict_type)
        elif isinstance(predict_type, PredictType):
            ptype = predict_type
        else:
            raise TypeError(f"predict_type {predict_type} of python type {type(predict_type)} not supported")

        self.predict_type = ptype
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
        if self.predict_type == PredictType.ENERGY_RATE:  # convert absolute consumption to rate consumption
            energy_rate_name = self.feature_pack.energy.name + "_per_" + self.feature_pack.distance.name
            energy_rate = data[self.feature_pack.energy.name] / data[self.feature_pack.distance.name]
            data[energy_rate_name] = energy_rate

            x = data[self.feature_pack.feature_list]
            y = data[energy_rate_name]
        elif self.predict_type == PredictType.ENERGY_RAW:
            x = data[self.feature_pack.feature_list + [self.feature_pack.distance.name]]
            y = data[self.feature_pack.energy.name]
        else:
            raise NotImplemented(f"{self.predict_type} not supported by BaseEstimator")

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
        if self.predict_type == PredictType.ENERGY_RATE:
            x = data[self.feature_pack.feature_list]
            _energy_pred_rates = self.model.predict(x.values)
            _energy_pred = _energy_pred_rates * data[self.feature_pack.distance.name]
        elif self.predict_type == PredictType.ENERGY_RAW:
            x = data[self.feature_pack.feature_list + [self.feature_pack.distance.name]]
            _energy_pred = self.model.predict(x.values)
        else:
            raise NotImplemented(f"{self.predict_type} not supported by BaseEstimator")

        energy_pred = Series(clip(_energy_pred, a_min=0, a_max=None), name=self.predict_type.name)

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
            'predict_type': self.predict_type.name
        }

        return out_json

    @classmethod
    def from_json(cls, json: dict) -> BaseEstimator:
        model_dict = json['model']
        model = linear_model.LinearRegression(model_dict['params'])
        model.coef_ = np.array(model_dict['coef_'])
        model.intercept_ = np.array(model_dict['intercept_'])

        predict_type = PredictType.from_string(json['predict_type'])
        feature_pack = FeaturePack.from_json(json['feature_pack'])

        return BaseEstimator(feature_pack=feature_pack, predict_type=predict_type, model=model)






