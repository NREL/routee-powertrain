from __future__ import annotations

import json
import warnings
from typing import NamedTuple

import numpy as np

from powertrain.core.features import FeaturePack
from powertrain.core.powertrain_type import PowertrainType
from powertrain.utils.fs import get_version

METADATA_SERIALIZATION_KEY = "routee_metadata"


def add_metadata_to_onnx_model(onnx_model, metadata: Metadata):
    """
    Helper to add routee metadata to an onnx model
    """
    routee_meta = onnx_model.metadata_props.add()

    routee_meta.key = METADATA_SERIALIZATION_KEY
    routee_meta.value = metadata.to_json()

    return onnx_model


class Metadata(NamedTuple):
    """
    A named tuple carrying model metadata information
    """

    model_description: str
    powertrain_type: PowertrainType

    feature_pack: FeaturePack

    energy_rate_low_limit: float = 0.0
    energy_rate_high_limit: float = np.inf

    test_size: float = 0.2

    random_seed: int = 42

    errors: dict = {}

    feature_dtype: str = "float32"
    onnx_input_name: str = "input"

    trip_column: str = "trip_id"

    routee_version: str = get_version()

    def set_errors(self, errors: dict) -> Metadata:
        return self._replace(errors=errors)

    def to_dict(self) -> dict:
        d = self._asdict()
        d["powertrain_type"] = self.powertrain_type.name
        d["feature_pack"] = self.feature_pack.to_dict()

        return d

    def to_json(self) -> str:
        """
        Convert metadata to json string
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, j: dict) -> Metadata:
        v = get_version()
        if j["routee_version"] != v:
            warnings.warn(
                f"this model was trained using routee-powertrain version {j['routee_version']}"
                f" but you're using version {v}"
            )

        j["powertrain_type"] = PowertrainType.from_string(j.get("powertrain_type"))
        j["feature_pack"] = FeaturePack.from_dict(j["feature_pack"])

        return Metadata(**j)

    @classmethod
    def from_json(cls, j: str) -> Metadata:
        return cls.from_dict(json.loads(j))
