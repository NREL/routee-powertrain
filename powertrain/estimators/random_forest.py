from __future__ import annotations

from typing import Union, Optional

from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor

from powertrain.core.core_utils import serialize_random_forest_regressor, deserialize_random_forest_regressor
from powertrain.core.features import PredictType, FeaturePack
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
            predict_type: Union[str, int, PredictType] = PredictType.ENERGY_RAW,
            model: Optional[RandomForestRegressor] = None,
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
            raise NotImplementedError(f"{self.predict_type} not supported by RandomForest")
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
            raise NotImplementedError(f"{self.predict_type} not supported by RandomForest")

        # energy_pred = Series(clip(_energy_pred, a_min=0, a_max=None), name=self.feature_pack.energy.name)
        energy_pred = Series(_energy_pred)

        return energy_pred

    def to_json(self) -> dict:
        out_json = {
            'model': serialize_random_forest_regressor(self.model),
            'feature_pack': self.feature_pack.to_json(),
            'predict_type': self.predict_type.name,
            'cores': self.cores
        }

        return out_json

    @classmethod
    def from_json(cls, json: dict) -> RandomForest:
        model_dict = json['model']
        model = deserialize_random_forest_regressor(model_dict)

        predict_type = PredictType.from_string(json['predict_type'])
        feature_pack = FeaturePack.from_json(json['feature_pack'])
        cores = json['cores']

        e = RandomForest(
            feature_pack=feature_pack,
            predict_type=predict_type,
            cores=cores,
            model=model
        )
        return e
