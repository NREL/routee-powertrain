from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Tuple, List


class Feature(NamedTuple):
    name: str
    units: str

    @classmethod
    def from_dict(cls, d: dict) -> Feature:
        return Feature(name=d['name'], units=d['units'])


class FeaturePack(NamedTuple):
    features: Tuple[Feature]
    distance: Feature
    energy: Feature

    @property
    def feature_list(self) -> List[str]:
        return [f.name for f in self.features]

    def to_json(self) -> dict:
        return {
            'features': [f._asdict() for f in self.features],
            'distance': self.distance._asdict(),
            'energy': self.energy._asdict(),
        }

    @classmethod
    def from_json(cls, json: dict) -> FeaturePack:
        features = tuple(Feature.from_dict(d) for d in json['features'])
        distance = Feature.from_dict(json['distance'])
        energy = Feature.from_dict(json['energy'])
        return FeaturePack(features, distance, energy)


class PredictType(Enum):
    """
    describes the nature of the estimator prediction

    ENERGY_RATE: predicts the fuel consumption rate per distance unit
    ENERGY_RAW: predicts the raw fuel consumption
    """
    ENERGY_RATE = 1
    ENERGY_RAW = 2

    @classmethod
    def from_string(cls, string: str):
        if string.lower() in ['raw_energy', 'energy_raw']:
            return PredictType.ENERGY_RAW
        elif string.lower() in ['rate_energy', 'energy_rate']:
            return PredictType.ENERGY_RATE
        else:
            raise TypeError(f"{string} not a supported predict type.")

    @classmethod
    def from_int(cls, integer: int):
        if integer == 1:
            return PredictType.ENERGY_RATE
        elif integer == 2:
            return PredictType.ENERGY_RAW
        else:
            raise TypeError(f"{integer} not a supported predict type integer option.")
