from __future__ import annotations

from typing import Optional

from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor

from powertrain.core.core_utils import serialize_random_forest_regressor, deserialize_random_forest_regressor
from powertrain.core.features import FeaturePack
from powertrain.estimators.estimator_interface import EstimatorInterface


class RandomForest(EstimatorInterface):
    """This estimator uses a random forest to select an optimal decision tree,
    meant to serve as an automated construction of a lookup table.

    Args:
        cores (int):
            Number of cores to use during training.
            
    """

    def __init__(
            self,
            feature_pack: FeaturePack,
            cores: int = 4,
            model: Optional[RandomForestRegressor] = None,
    ):
        if not model:
            model = RandomForestRegressor(n_estimators=20,
                                          max_features='auto',
                                          max_depth=10,
                                          min_samples_split=10,
                                          n_jobs=cores,
                                          random_state=52)

        self.model: RandomForestRegressor = model

        self.feature_pack: FeaturePack = feature_pack
        self.cores = cores

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
        out_json = {
            'model': serialize_random_forest_regressor(self.model),
            'feature_pack': self.feature_pack.to_json(),
            'cores': self.cores
        }

        return out_json

    @classmethod
    def from_json(cls, json: dict) -> RandomForest:
        model_dict = json['model']
        model = deserialize_random_forest_regressor(model_dict)

        feature_pack = FeaturePack.from_json(json['feature_pack'])
        cores = json['cores']

        e = RandomForest(
            feature_pack=feature_pack,
            cores=cores,
            model=model
        )
        return e
