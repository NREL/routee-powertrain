from __future__ import annotations
import json

from pandas import DataFrame, Series
from powertrain.core.features import FeaturePack
from powertrain.estimators.estimator_interface import EstimatorInterface


class SmartCoreRandomForest(EstimatorInterface):
    """
    This estimator uses a rust smartcore random forest to select an optimal decision tree,
    """

    def __init__(
        self,
        feature_pack: FeaturePack,
    ):
        self.feature_pack: FeaturePack = feature_pack

    def train(self, data: DataFrame, **kwargs):
        """
        train method for the base estimator (linear regression)
        Args:
            data:

        Returns:

        """
        try:
            from powertrain_rust import RustRandomForest
        except ImportError:
            raise ImportError(
                "Please install powertrain_rust to use the SmartCoreRandomForest estimator."
            )

        x = data[self.feature_pack.feature_list]
        x = x.values.tolist()
        y = list(data.energy_rate.values)

        self.model = RustRandomForest().train(x, y)

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
        _energy_pred_rates = self.model.predict(x.values.tolist())

        energy_pred = Series(_energy_pred_rates, index=data.index)

        return energy_pred

    def to_json(self) -> dict:
        out_json = {
            "model": json.loads(self.model.to_json()),
            "feature_pack": self.feature_pack.to_json(),
        }

        return out_json

    @classmethod
    def from_json(cls, j: dict) -> SmartCoreRandomForest:
        model_dict = j["model"]
        try:
            from powertrain_rust import RustRandomForest
        except ImportError:
            raise ImportError(
                "Please install powertrain_rust to use the SmartCoreRandomForest estimator."
            )
        model = RustRandomForest.from_json(json.dumps(model_dict))

        feature_pack = FeaturePack.from_json(j["feature_pack"])

        e = SmartCoreRandomForest(feature_pack=feature_pack)
        e.model = model

        return e
