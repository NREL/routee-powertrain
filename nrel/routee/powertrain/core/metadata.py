from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from typing import Optional

from nrel.routee.powertrain.core.model_config import ModelConfig
from nrel.routee.powertrain.utils.fs import get_version

METADATA_SERIALIZATION_KEY = "routee_metadata"


def add_metadata_to_onnx_model(onnx_model, metadata: Metadata):
    """
    Helper to add routee metadata to an onnx model
    """
    routee_meta = onnx_model.metadata_props.add()

    routee_meta.key = METADATA_SERIALIZATION_KEY
    routee_meta.value = metadata.to_json()

    return onnx_model


@dataclass
class Metadata:
    """
    A named tuple carrying model metadata information that gets set post training
    """

    config: ModelConfig

    errors: Optional[dict] = None

    routee_version: str = get_version()

    def set_errors(self, errors: dict) -> Metadata:
        return replace(self, errors=errors)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["config"] = self.config.to_dict()

        return d

    def to_json(self) -> str:
        """
        Convert metadata to json string
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> Metadata:
        v = get_version()
        if d["routee_version"] != v:
            warnings.warn(
                "this model was trained using routee-powertrain version "
                f"{d['routee_version']} but you're using version {v}"
            )

        d["config"] = ModelConfig.from_dict(d["config"])

        return Metadata(**d)

    @classmethod
    def from_json(cls, j: str) -> Metadata:
        return cls.from_dict(json.loads(j))
