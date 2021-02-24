from typing import Union

from numpy import clip
from pandas import DataFrame, Series
from xgboost import XGBRegressor

from powertrain.core.features import PredictType, FeaturePack
from powertrain.estimators.estimator_interface import EstimatorInterface


class XGBoost(EstimatorInterface):
    """This estimator uses a xgboost tree to select an optimal decision tree,
    meant to serve as an automated construction of a lookup table.
    """

    def __init__(
            self,
            feature_pack: FeaturePack,
            predict_type: Union[str, int, PredictType] = PredictType.ENERGY_RAW,
    ):
        if isinstance(predict_type, str):
            ptype = PredictType.from_string(predict_type)
        elif isinstance(predict_type, int):
            ptype = PredictType.from_int(predict_type)
        elif isinstance(predict_type, PredictType):
            ptype = predict_type
        else:
            raise TypeError(f"predict_type {predict_type} of python type {type(predict_type)} not supported")
        self.predict_type: PredictType = ptype

        mod = XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3
        )
        self.model: XGBRegressor = mod

        self.feature_pack: FeaturePack = feature_pack

    def train(self, data: DataFrame):
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
            raise NotImplementedError(f"{self.predict_type} not supported by XGBoost")
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
            raise NotImplementedError(f"{self.predict_type} not supported by XGBoost")

        energy_pred = Series(clip(_energy_pred, a_min=0, a_max=None), name=self.predict_type.name)

        return energy_pred

    def to_json(self) -> dict:
        raise NotImplementedError("to_json() not implemented for XGBoost estimator")

    @classmethod
    def from_json(cls, json: dict) -> EstimatorInterface:
        raise NotImplementedError("from_json() not implemented for XGBoost estimator")

