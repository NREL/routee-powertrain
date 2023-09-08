from __future__ import annotations
from dataclasses import dataclass

from typing import Dict, List

import numpy as np


@dataclass
class FeatureRange:
    lower: float = -np.inf
    upper: float = np.inf

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError("lower bound must be less than upper bound")

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> FeatureRange:
        return FeatureRange(**d)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Feature:
    name: str
    units: str

    feature_range: FeatureRange = FeatureRange()

    @classmethod
    def from_dict(cls, d: dict) -> Feature:
        if "name" not in d:
            raise ValueError("must provide feature name when building from dictionary")
        elif "units" not in d:
            raise ValueError("must provide feature units when building from dictionary")
        elif "feature_range" not in d:
            raise ValueError("must provide feature range when building from dictionary")

        feature_range = FeatureRange.from_dict(d["feature_range"])

        return Feature(name=d["name"], units=d["units"], feature_range=feature_range)

    def to_dict(self) -> dict:
        out = self.__dict__.copy()
        out["feature_range"] = self.feature_range.to_dict()

        return out


@dataclass
class FeaturePack:
    features: List[Feature]
    distance: Feature
    energy: List[Feature]

    def __post_init__(self):
        if isinstance(self.features, Feature):
            self.features = [self.features]
        if isinstance(self.energy, Feature):
            self.energy = [self.energy]

    @property
    def feature_map(self) -> Dict[str, Feature]:
        return {f.name: f for f in self.features}

    @property
    def energy_map(self) -> Dict[str, Feature]:
        return {f.name: f for f in self.energy}

    @property
    def all_names(self) -> List[str]:
        return self.feature_name_list + self.energy_name_list + [self.distance.name]

    @property
    def feature_name_list(self) -> List[str]:
        return [f.name for f in self.features]

    @property
    def energy_name_list(self) -> List[str]:
        return [f.name for f in self.energy]

    @property
    def energy_rate_name_list(self) -> List[str]:
        return [f"{f.name}_rate" for f in self.energy]

    def to_dict(self) -> dict:
        return {
            "features": [f.to_dict() for f in self.features],
            "distance": self.distance.to_dict(),
            "energy": [e.to_dict() for e in self.energy],
        }

    @classmethod
    def from_dict(cls, json: dict) -> FeaturePack:
        features = [Feature.from_dict(d) for d in json["features"]]
        distance = Feature.from_dict(json["distance"])
        energy = [Feature.from_dict(d) for d in json["energy"]]
        return FeaturePack(features, distance, energy)
