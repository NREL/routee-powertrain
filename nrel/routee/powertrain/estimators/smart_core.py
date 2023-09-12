from __future__ import annotations

import json
import pandas as pd

from nrel.routee.powertrain.core.metadata import Metadata
from nrel.routee.powertrain.estimators.estimator import Estimator


class SmartCoreEstimator(Estimator):
    def __init__(self, smartcore_rf) -> None:
        self.model = smartcore_rf

    @classmethod
    def from_dict(cls, in_dict: dict) -> SmartCoreEstimator:
        try:
            from powertrain_rust import RustRandomForest
        except ImportError:
            raise ImportError(
                "Please install powertrain_rust to use the SmartCoreRandomForest estimator."
            )
        smartcore_model_raw = in_dict.get("smartcore_model")
        if smartcore_model_raw is None:
            raise ValueError(
                "Model file must contain smartcore model at key: 'smartcore_model'"
            )
        if isinstance(smartcore_model_raw, str):
            smartcore_model = RustRandomForest.from_json(smartcore_model_raw)
        elif isinstance(smartcore_model_raw, dict):
            input_json = json.dumps(smartcore_model_raw)
            smartcore_model = RustRandomForest.from_json(input_json)
        else:
            raise ValueError("Smartcore input must be a string or a dictionary")

        return cls(smartcore_model)

    def to_dict(self) -> dict:
        out_dict = {
            "smartcore_model": json.loads(self.model.to_json()),
        }
        return out_dict

    def predict(self, links_df: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
        config = metadata.config

        if len(config.feature_pack.energy) != 1:
            raise ValueError(
                "SmartCore only supports a single energy rate. "
                "Please use a different estimator."
            )
        energy = config.feature_pack.energy[0]

        distance_col = config.feature_pack.distance.name

        x = links_df[config.feature_pack.feature_name_list].values

        energy_pred_rates = self.model.predict(x.tolist())

        energy_df = pd.DataFrame(index=links_df.index)

        energy_pred = energy_pred_rates * links_df[distance_col]
        energy_df[energy.name] = energy_pred

        return energy_df
