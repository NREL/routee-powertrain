from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Union
from urllib import request

import numpy as np
import pandas as pd

from nrel.routee.powertrain.core.metadata import Metadata
from nrel.routee.powertrain.core.real_world_adjustments import ADJUSTMENT_FACTORS
from nrel.routee.powertrain.estimators.estimator import Estimator
from nrel.routee.powertrain.estimators.onnx import ONNXEstimator
from nrel.routee.powertrain.estimators.smart_core import SmartCoreEstimator

REGISTERED_ESTIMATORS = {
    "ONNXEstimator": ONNXEstimator,
    "SmartCoreEstimator": SmartCoreEstimator,
}

METADATA_SERIALIZATION_KEY = "metadata"
ESTIMATOR_SERIALIZATION_KEY = "estimator"
CONSTRUCTOR_TYPE_SERIALIZATION_KEY = "estimator_constructor_type"


@dataclass
class Model:
    """
    A RouteE-Powertrain vehicle model represents a single vehicle
    (i.e. a 2016 Toyota Camry with a 1.5 L gasoline engine).
    """

    estimator: Estimator
    metadata: Metadata

    @property
    def feature_pack(self):
        return self.metadata.config.feature_pack

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

        estimator_dict = input_dict.get(ESTIMATOR_SERIALIZATION_KEY)
        if estimator_dict is None:
            raise ValueError(
                "Model file must contain estimator data at key: "
                f"'{ESTIMATOR_SERIALIZATION_KEY}'"
            )

        estimator_constructor_type = input_dict.get("estimator_constructor_type")
        if estimator_constructor_type is None:
            raise ValueError(
                "Model file must contain estimator constructor at key: "
                f"'{CONSTRUCTOR_TYPE_SERIALIZATION_KEY}'"
            )

        estimator_constructor = REGISTERED_ESTIMATORS.get(estimator_constructor_type)
        if estimator_constructor is None:
            raise ValueError(
                f"Estimator constructor type '{estimator_constructor_type}' "
                "is not registered"
            )

        estimator = estimator_constructor.from_dict(estimator_dict)
        return cls(estimator, metadata)

    def to_dict(self) -> dict:
        """
        Convert model to a dictionary
        """
        return {
            METADATA_SERIALIZATION_KEY: self.metadata.to_dict(),
            ESTIMATOR_SERIALIZATION_KEY: self.estimator.to_dict(),
            CONSTRUCTOR_TYPE_SERIALIZATION_KEY: self.estimator.__class__.__name__,
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
        self, links_df: pd.DataFrame, apply_real_world_adjustment: bool = False
    ) -> pd.DataFrame:
        """
        Predict absolute energy consumption for each link
        """
        config = self.metadata.config

        distance_col = self.metadata.config.feature_pack.distance.name

        if distance_col not in links_df.columns:
            raise ValueError(
                f"links_df must contain a distance column named: '{distance_col}' "
                "according to the model metadata"
            )

        for feature in config.feature_pack.features:
            if feature.name not in links_df.columns:
                raise ValueError(
                    f"links_df must contain a feature column named: '{feature.name}' "
                    "according to the model metadata"
                )

        pred_energy_df = self.estimator.predict(links_df, self.metadata)

        for energy in config.feature_pack.energy:
            if apply_real_world_adjustment:
                adjustment_factor = ADJUSTMENT_FACTORS[config.powertrain_type]
                pred_energy_df[energy.name] = (
                    pred_energy_df[energy.name] * adjustment_factor
                )

        return pred_energy_df

    def to_lookup_table(self, feature_bins: Dict[str, int]) -> pd.DataFrame:
        """
        Returns a lookup table for the model.

        Args:
            feature_bins (dict): A dictionary of features and their number of bins.


        Returns:
            lookup_table (pandas.DataFrame):
                A lookup table for the model.

        """
        if any([f.feature_range is None for f in self.feature_pack.features]):
            raise ValueError("Feature ranges must be set to generate lookup table")
        elif set(feature_bins.keys()) != set(self.feature_pack.feature_name_list):
            raise ValueError("Feature names must match model feature pack")

        # build a grid mesh over the feature ranges
        points = tuple(
            np.linspace(
                f.feature_range.lower,
                f.feature_range.upper,
                feature_bins[f.name],
            )
            for f in self.feature_pack.features
        )

        mesh = np.meshgrid(*points)

        pred_matrix = np.stack(list(map(np.ravel, mesh)), axis=1)

        pred_df = pd.DataFrame(pred_matrix, columns=self.feature_pack.feature_name_list)
        pred_df[self.feature_pack.distance.name] = 1

        predictions = self.predict(pred_df)

        lookup = pred_df.drop(columns=[self.feature_pack.distance.name])
        energy_key = (
            f"{self.feature_pack.energy.units}_per_{self.feature_pack.distance.units}"
        )
        lookup[energy_key] = predictions

        return lookup

    def set_errors(self, errors: dict) -> Model:
        new_meta = self.metadata.set_errors(errors)
        return Model(estimator=self.estimator, metadata=new_meta)
