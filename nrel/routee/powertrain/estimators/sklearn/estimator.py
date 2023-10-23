from __future__ import annotations
import json
import pickle
from typing import TYPE_CHECKING
from pathlib import Path

import pandas as pd
from nrel.routee.powertrain.core.features import DataColumn, FeatureSet, TargetSet
from nrel.routee.powertrain.estimators.estimator_interface import Estimator

from .port_to_c import (
    c_header_from_random_forest,
    c_source_from_random_forest,
    parse_port_name,
)
from .utils import (
    deserialize_random_forest_regressor,
    serialize_random_forest_regressor,
)

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestRegressor


class SKLearnEstimator(Estimator):
    # only supports random forest for now but we
    # could extend this to other sklearn models
    sklearn_model: RandomForestRegressor

    def __init__(self, sklearn_model: RandomForestRegressor) -> None:
        self.sklearn_model = sklearn_model

    @classmethod
    def from_dict(cls, in_dict: dict) -> SKLearnEstimator:
        rf_model_dict = in_dict.get("rf_regressor")
        if rf_model_dict is None:
            raise ValueError(
                "Model file must contain random forest model at key: 'rf_regressor'"
            )
        rf_model = deserialize_random_forest_regressor(rf_model_dict)
        return cls(rf_model)

    def to_dict(self) -> dict:
        out_dict = {
            "rf_regressor": serialize_random_forest_regressor(self.sklearn_model)
        }
        return out_dict

    @classmethod
    def from_file(cls, filepath: str | Path) -> SKLearnEstimator:
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            with filepath.open("r") as f:
                in_dict = json.load(f)
                rf_model = deserialize_random_forest_regressor(in_dict["rf_regressor"])
        elif filepath.suffix == ".pickle":
            with filepath.open("rb") as f:
                rf_model = pickle.load(f)
        return cls(rf_model)

    def to_file(self, filepath: str | Path):
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            with filepath.open("w") as f:
                out_dict = {
                    "rf_regressor": serialize_random_forest_regressor(
                        self.sklearn_model
                    )
                }
                f.write(json.dumps(out_dict))
        elif filepath.suffix == ".pickle":
            with filepath.open("wb") as f:
                pickle.dump(self.sklearn_model, f)
        else:
            raise ValueError(
                "SkLearnEstimator must be saved as a .json or .pickle file"
            )

    def to_c_code(self, outdir: str | Path, name: str):
        outpath = Path(outdir)

        name = parse_port_name(name)

        header_file = outpath / f"{name}.h"
        source_file = outpath / f"{name}.c"

        header_str = c_header_from_random_forest(self.sklearn_model, name)
        source_str = c_source_from_random_forest(self.sklearn_model, name)

        with header_file.open("w") as hf:
            hf.write(header_str)

        with source_file.open("w") as sf:
            sf.write(source_str)

    def predict(
        self,
        links_df: pd.DataFrame,
        feature_set: FeatureSet,
        distance: DataColumn,
        target_set: TargetSet,
    ) -> pd.DataFrame:
        distance_col = distance.name

        x = links_df[feature_set.feature_name_list].values

        raw_energy_pred_rates = self.sklearn_model.predict(x)

        energy_df = pd.DataFrame(index=links_df.index)

        for i, energy in enumerate(target_set.targets):
            energy_pred_rates = pd.Series(
                raw_energy_pred_rates[:, i], index=links_df.index
            )

            energy_pred = energy_pred_rates * links_df[distance_col]
            energy_df[energy.name] = energy_pred

        return energy_df
