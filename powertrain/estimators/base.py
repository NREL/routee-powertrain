from typing import Union

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
            predict_type: Union[str, int, PredictType] = PredictType.ENERGY_RAW):
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
        self.model: linear_model.LinearRegression = linear_model.LinearRegression()

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
            energy_rate_name = self.feature_pack.energy_name + "_per_" + self.feature_pack.distance_name
            energy_rate = data[self.feature_pack.energy_name] / data[self.feature_pack.distance_name]
            data[energy_rate_name] = energy_rate

            x = data[self.feature_pack.feature_list]
            y = data[energy_rate_name]
        elif self.predict_type == PredictType.ENERGY_RAW:
            x = data[self.feature_pack.feature_list + [self.feature_pack.distance_name]]
            y = data[self.feature_pack.energy_name]
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
            _energy_pred = _energy_pred_rates * data[self.feature_pack.distance_name]
        elif self.predict_type == PredictType.ENERGY_RAW:
            x = data[self.feature_pack.feature_list + [self.feature_pack.distance_name]]
            _energy_pred = self.model.predict(x.values)
        else:
            raise NotImplemented(f"{self.predict_type} not supported by BaseEstimator")

        energy_pred = Series(clip(_energy_pred, a_min=0), name=self.predict_type.name)

        return energy_pred
