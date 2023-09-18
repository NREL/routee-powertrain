from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib import request

import pandas as pd
from nrel.routee.powertrain.core.features import (
    FeatureSetId,
    feature_id_to_names,
    feature_names_to_id,
)

from nrel.routee.powertrain.core.metadata import Metadata
from nrel.routee.powertrain.core.real_world_adjustments import ADJUSTMENT_FACTORS
from nrel.routee.powertrain.estimators.estimator_interface import Estimator
from nrel.routee.powertrain.estimators.onnx import ONNXEstimator
from nrel.routee.powertrain.estimators.smart_core import SmartCoreEstimator

REGISTERED_ESTIMATORS = {
    "ONNXEstimator": ONNXEstimator,
    "SmartCoreEstimator": SmartCoreEstimator,
}

METADATA_SERIALIZATION_KEY = "metadata"
ALL_ESTIMATOR_SERIALIZATION_KEY = "all_estimators"
ESTIMATOR_SERIALIZATION_KEY = "estimator"
CONSTRUCTOR_TYPE_SERIALIZATION_KEY = "estimator_constructor_type"


@dataclass
class Model:
    """
    A RouteE-Powertrain vehicle model represents a single vehicle
    (i.e. a 2016 Toyota Camry with a 1.5 L gasoline engine).
    """

    estimators: Dict[FeatureSetId, Estimator]
    metadata: Metadata

    @property
    def feature_sets(self):
        return self.metadata.config.feature_sets

    @property
    def feature_set_lists(self) -> List[List[str]]:
        return [feature_id_to_names(fid) for fid in self.estimators.keys()]

    @classmethod
    def from_dict(cls, input_dict: dict) -> Model:
        """
        Load a vehicle model from a python dictionary
        """
        metadata_dict = input_dict.get(METADATA_SERIALIZATION_KEY)
        if metadata_dict is None:
            raise ValueError(
                "Model file must contain metadata at key: "
                f"'{METADATA_SERIALIZATION_KEY}'"
            )
        metadata = Metadata.from_dict(metadata_dict)

        all_estimators_dict = input_dict.get(ALL_ESTIMATOR_SERIALIZATION_KEY)
        if all_estimators_dict is None:
            raise ValueError(
                "Model file must contain estimator data at key: "
                f"'{ALL_ESTIMATOR_SERIALIZATION_KEY}'"
            )

        estimator_constructor_type = input_dict.get("estimator_constructor_type")

        estimators = {}
        for feature_set_id, ed in all_estimators_dict.items():
            constructor_type = ed.get(CONSTRUCTOR_TYPE_SERIALIZATION_KEY)
            if estimator_constructor_type is None:
                raise ValueError(
                    "Model file must contain estimator constructor at key: "
                    f"'{CONSTRUCTOR_TYPE_SERIALIZATION_KEY}'"
                )

            estimator_constructor = REGISTERED_ESTIMATORS.get(constructor_type)
            if estimator_constructor is None:
                raise ValueError(
                    f"Estimator constructor type '{estimator_constructor_type}' "
                    "is not registered"
                )

            estimator_input_dict = ed.get(ESTIMATOR_SERIALIZATION_KEY)
            if estimator_input_dict is None:
                raise ValueError(
                    "Model file must contain estimator data at key: "
                    f"'{ESTIMATOR_SERIALIZATION_KEY}'"
                )

            estimator = estimator_constructor.from_dict(estimator_input_dict)
            estimators[feature_set_id] = estimator

        return cls(estimators, metadata)

    def to_dict(self) -> dict:
        """
        Convert model to a dictionary
        """
        estimator_dict = {}
        for feature_set_id, estimator in self.estimators.items():
            estimator_dict[feature_set_id] = {
                ESTIMATOR_SERIALIZATION_KEY: estimator.to_dict(),
                CONSTRUCTOR_TYPE_SERIALIZATION_KEY: estimator.__class__.__name__,
            }

        return {
            METADATA_SERIALIZATION_KEY: self.metadata.to_dict(),
            ALL_ESTIMATOR_SERIALIZATION_KEY: estimator_dict,
            CONSTRUCTOR_TYPE_SERIALIZATION_KEY: self.estimators.__class__.__name__,
        }

    @classmethod
    def from_file(cls, file: Union[str, Path]):
        """
        Load a vehicle model from a file.
        """
        path = Path(file)
        if path.suffix != ".json":
            raise ValueError("Model file must be a .json file")
        with path.open("r") as f:
            input_dict = json.load(f)
        return cls.from_dict(input_dict)

    @classmethod
    def from_url(cls, url: str) -> Model:
        """
        attempts to read a file from a url
        Args:
            url: the url to download the file from
            filetype: the type of file to expect

        Returns: a powertrain vehicle
        """
        with request.urlopen(url) as u:
            in_dict = json.load(u)
            vehicle = cls.from_dict(in_dict)

        return vehicle

    def to_file(self, file: Union[str, Path]):
        """
        Save a vehicle model to a file.
        """
        path = Path(file)
        if path.suffix != ".json":
            raise ValueError("Model file must be a .json file")

        output_dict = self.to_dict()
        with path.open("w") as f:
            json.dump(output_dict, f)

    def predict(
        self,
        links_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        distance_column: Optional[str] = None,
        apply_real_world_adjustment: bool = False,
    ) -> pd.DataFrame:
        """
        Predict absolute energy consumption for each link
        """
        config = self.metadata.config

        if distance_column is None:
            distance_column = config.distance.name
            if distance_column not in links_df.columns:
                raise ValueError(
                    f"links_df must contain a distance column named: '{distance_column}'"
                )
        else:
            links_df = links_df.rename(columns={distance_column: config.distance.name})


        # if we only have one estimator, just use that
        if len(self.estimators) == 1:
            feature_set_id = list(self.estimators.keys())[0]
            estimator = self.estimators.get(feature_set_id)
            if estimator is None:
                raise ValueError("Could not find estimator")

        # if no explicit feature names are supplied we assume that the
        # dataframe contains all the features needed for prediction;
        # if that isn't the case, we throw an error
        elif feature_columns is None:
            feature_columns = [c for c in links_df.columns if c != distance_column]
            feature_set_id = feature_names_to_id(feature_columns)
            estimator = self.estimators.get(feature_set_id)
            if estimator is None:
                raise ValueError(
                    "This model has multiple feature sets and no features were "
                    "explicitly provided. "
                    "We attempted to just use the columns in the incoming dataframe "
                    "but we couldn't find an estiamtor that matches the features: "
                    f"{feature_columns}. "
                    "Please provide an explicit list of feature names to the features "
                    "paramter of the predict method or provide a dataframe that only "
                    "contains the features you want to use. "
                    "Here are the feature sets that can be used: "
                    f"{self.feature_set_lists}"
                )
        else:
            feature_set_id = feature_names_to_id(feature_columns)
            estimator = self.estimators.get(feature_set_id)
            if estimator is None:
                raise ValueError(
                    "Could not find an estimator that matches the provided "
                    f"feature columns {feature_columns}. Here are the feature "
                    f"sets that can be used: {self.feature_set_lists}"
                )

        feature_set = self.metadata.config.feature_set_map.get(feature_set_id)
        if feature_set is None:
            raise ValueError(
                f"Could not find a feature set {feature_set_id} in model config"
            )

        pred_energy_df = estimator.predict(
            links_df,
            feature_set,
            self.metadata.config.distance,
            self.metadata.config.target,
        )

        for energy in config.target.targets:
            if apply_real_world_adjustment:
                adjustment_factor = ADJUSTMENT_FACTORS.get(config.powertrain_type)
                if adjustment_factor is None:
                    raise ValueError(
                        f"Could not find an adjustment factor for powertrain type "
                        f"{config.powertrain_type}"
                    )
                pred_energy_df[energy.name] = (
                    pred_energy_df[energy.name] * adjustment_factor
                )

        return pred_energy_df

    def set_errors(self, errors: dict) -> Model:
        new_meta = self.metadata.set_errors(errors)
        return Model(estimators=self.estimators, metadata=new_meta)
