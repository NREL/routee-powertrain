from __future__ import annotations

from typing import NamedTuple, Tuple, List, Optional


class Range(NamedTuple):
    lower: float
    upper: float

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional[Range]:
        if not d:
            return None

        return Range(**d)


class Feature(NamedTuple):
    name: str
    units: str

    range: Optional[Range] = None

    @classmethod
    def from_dict(cls, d: dict) -> Feature:
        if 'name' not in d:
            raise ValueError("must provide feature name when building from dictionary")
        elif 'units' not in d:
            raise ValueError("must provide feature units when building from dictionary")

        range = Range.from_dict(d.get('range'))

        return Feature(name=d['name'], units=d['units'], range=range)


class FeaturePack(NamedTuple):
    features: Tuple[Feature, ...]
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
