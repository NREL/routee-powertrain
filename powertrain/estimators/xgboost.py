from pandas import DataFrame, Series
from xgboost import XGBRegressor

from powertrain.core.features import FeaturePack
from powertrain.estimators.estimator_interface import EstimatorInterface


class XGBoost(EstimatorInterface):
    """This estimator uses a xgboost tree to select an optimal decision tree,
    meant to serve as an automated construction of a lookup table.
    """

    def __init__(
            self,
            feature_pack: FeaturePack,
    ):
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

        energy_pred = Series(_energy_pred, index=data.index)

        return energy_pred

    def to_json(self) -> dict:
        raise NotImplementedError("to_json() not implemented for XGBoost estimator")

    @classmethod
    def from_json(cls, json: dict) -> EstimatorInterface:
        raise NotImplementedError("from_json() not implemented for XGBoost estimator")
