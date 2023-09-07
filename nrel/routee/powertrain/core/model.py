from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union
from urllib import request

import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd

from nrel.routee.powertrain.core.metadata import (
    METADATA_SERIALIZATION_KEY,
    Metadata,
    add_metadata_to_onnx_model,
)
from nrel.routee.powertrain.core.real_world_adjustments import ADJUSTMENT_FACTORS


@dataclass
class Model:
    """
    A RouteE-Powertrain vehicle model represents a single vehicle (i.e. a 2016 Toyota Camry with a 1.5 L gasoline engine).

    The model uses ONNX to load a pre-trained model and use it for making energy predictions.
    """

    onnx_model: onnx.ModelProto
    metadata: Metadata

    @property
    def feature_pack(self):
        return self.metadata.config.feature_pack

    @classmethod
    def build(cls, onnx_model: onnx.ModelProto, metadata: Metadata):
        """
        Build a vehicle model from an ONNX model.
        This assumes the metadata has not yet been set on the model
        """
        onnx_model = add_metadata_to_onnx_model(onnx_model, metadata)
        return cls.from_onnx_model(onnx_model)

    @classmethod
    def from_onnx_model(cls, onnx_model: onnx.ModelProto):
        """
        Create a vehicle model from an ONNX session
        """
        onnx_meta = {prop.key: prop.value for prop in onnx_model.metadata_props}

        routee_meta = onnx_meta.get(METADATA_SERIALIZATION_KEY)

        if routee_meta is None:
            raise ValueError(
                f"ONNX model does not contain a {METADATA_SERIALIZATION_KEY} key"
            )

        metadata = Metadata.from_json(routee_meta)

        return cls(onnx_model=onnx_model, metadata=metadata)

    @classmethod
    def from_file(cls, file: Union[str, Path]):
        """
        Load a vehicle model from a file.
        """
        onnx_model = onnx.load_model(str(file))
        return cls.from_onnx_model(onnx_model)

    @classmethod
    def from_bytes(cls, in_bytes: bytes) -> Model:
        onnx_model = onnx.load_from_string(in_bytes)
        return cls.from_onnx_model(onnx_model)

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
            in_bytes = u.read()
            vehicle = cls.from_bytes(in_bytes)
        return vehicle

    def to_file(self, file: Union[str, Path]):
        """
        Save a vehicle model to a file.
        """
        path = Path(file)
        onnx_model = add_metadata_to_onnx_model(self.onnx_model, self.metadata)
        with path.open("wb") as f:
            onnx.save_model(onnx_model, f)

    def predict(
        self, links_df: pd.DataFrame, apply_real_world_adjustment: bool = False
    ) -> pd.Series:
        """
        Predict absolute energy consumption for each link
        """
        config = self.metadata.config

        distance_col = self.metadata.config.feature_pack.distance.name

        if distance_col not in links_df.columns:
            raise ValueError(
                f"links_df must contain a distance column named: '{distance_col}'"
                "according to the model metadata"
            )

        for feature in config.feature_pack.features:
            if feature.name not in links_df.columns:
                raise ValueError(
                    f"links_df must contain a feature column named: '{feature.name}'"
                    "according to the model metadata"
                )

        x = links_df[config.feature_pack.feature_list].values

        onnx_session = rt.InferenceSession(self.onnx_model.SerializeToString())

        raw_energy_pred_rates = onnx_session.run(
            None, {config.onnx_input_name: x.astype(config.feature_dtype)}
        )[0]

        energy_pred_rates = pd.Series(
            raw_energy_pred_rates.reshape(-1), index=links_df.index
        )

        if apply_real_world_adjustment:
            adjustment_factor = ADJUSTMENT_FACTORS[config.powertrain_type]
            energy_pred_rates = energy_pred_rates * adjustment_factor

        energy_pred = energy_pred_rates * links_df[distance_col]

        return energy_pred

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
        elif set(feature_bins.keys()) != set(self.feature_pack.feature_list):
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

        pred_df = pd.DataFrame(pred_matrix, columns=self.feature_pack.feature_list)
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
        return Model(onnx_model=self.onnx_model, metadata=new_meta)
