from __future__ import annotations

from typing import Optional, NamedTuple, Tuple

import numpy as np
from pandas import DataFrame, Series
from scipy.interpolate import interpn
from sklearn.ensemble import RandomForestRegressor

from powertrain.core.core_utils import serialize_random_forest_regressor, deserialize_random_forest_regressor
from powertrain.core.features import FeaturePack
from powertrain.estimators.estimator_interface import EstimatorInterface


class LookupMatrix(NamedTuple):
    points: Tuple[np.ndarray, ...]
    energy_matrix: np.ndarray

    @classmethod
    def from_json(cls, d: dict) -> LookupMatrix:
        points = tuple(np.array(p) for p in d['points'])
        energy_matrix = np.array(d['energy_matrix'])

        return LookupMatrix(points, energy_matrix)

    def to_json(self) -> dict:
        out = {
            'points': tuple(p.tolist() for p in self.points),
            'energy_matrix': self.energy_matrix.tolist(),
        }
        return out


class RandomForestLookup(EstimatorInterface):
    """
    This estimator trains a random forest and then builds a lookup table for predictions.

    This was built for a tradeoff between high accuracy and ease of transfer between programming
    paradigms
    """

    def __init__(
            self,
            feature_pack: FeaturePack,
            model: Optional[LookupMatrix] = None,
    ):
        self.model = model
        for f in feature_pack:
            if not f.range:
                raise ValueError(f"must specify a min and max range for feature {f.name}")

        self.feature_pack: FeaturePack = feature_pack

    def train(
            self,
            data: DataFrame,
            cores: int = 4,
            grid_points: int = 50
    ):
        """
        trains the model

        Args:
            data:
            cores:
            grid_points:

        Returns:

        """
        rf_model = RandomForestRegressor(n_estimators=20,
                                         max_features='auto',
                                         max_depth=10,
                                         min_samples_split=10,
                                         n_jobs=cores,
                                         random_state=52)

        x = data[self.feature_pack.feature_list]
        y = data.energy_rate

        rf_model.fit(x.values, y.values)

        points = tuple(np.linspace(f.range.lower, f.range.upper, grid_points) for f in self.feature_pack.features)
        predictions = rf_model.predict(np.stack(list(map(np.ravel, np.meshgrid(*points))), axis=1))

        outshape = tuple(grid_points for _ in range(len(self.feature_pack.features)))
        energy_matrix = np.reshape(predictions, outshape)

        self.model = LookupMatrix(points, energy_matrix)

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
        raw_x = []
        for f in self.feature_pack.features:
            clipped_f = data[f.name].clip(f.range.lower, f.range.upper).values
            raw_x.append(clipped_f)

        x = np.array(raw_x).T

        _energy_pred_rates = interpn(self.model.points, self.model.energy_matrix, x)
        _energy_pred = _energy_pred_rates * data[self.feature_pack.distance.name].values

        energy_pred = Series(_energy_pred, index=data.index)

        return energy_pred

    def to_json(self) -> dict:
        out_json = {
            'model': self.model.to_json(),
            'feature_pack': self.feature_pack.to_json(),
        }

        return out_json

    @classmethod
    def from_json(cls, json: dict) -> RandomForestLookup:
        model_dict = json['model']
        model = LookupMatrix.from_json(model_dict)

        feature_pack = FeaturePack.from_json(json['feature_pack'])

        e = RandomForestLookup(
            feature_pack=feature_pack,
            model=model
        )
        return e
