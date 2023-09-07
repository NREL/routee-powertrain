from __future__ import annotations

from typing import Any, Iterable, List, NamedTuple, Optional


class FeatureRange(NamedTuple):
    lower: float
    upper: float

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional[FeatureRange]:
        if not d:
            return None

        return FeatureRange(**d)

    def to_json(self) -> dict:
        return self._asdict()


class Feature(NamedTuple):
    name: str
    units: str

    feature_range: Optional[FeatureRange] = None

    @classmethod
    def from_dict(cls, d: dict) -> Feature:
        if "name" not in d:
            raise ValueError("must provide feature name when building from dictionary")
        elif "units" not in d:
            raise ValueError("must provide feature units when building from dictionary")

        frange = FeatureRange.from_dict(d.get("feature_range"))

        return Feature(name=d["name"], units=d["units"], feature_range=frange)

    def to_dict(self) -> dict:
        out: dict[Any, Any] = {
            "name": self.name,
            "units": self.units,
        }
        if self.feature_range:
            out["feature_range"] = self.feature_range.to_json()

        return out


class FeaturePack(NamedTuple):
    features: Iterable[Feature]
    distance: Feature
    energy: Feature

    @property
    def all_names(self) -> List[str]:
        return self.feature_list + [self.distance.name, self.energy.name]

    @property
    def feature_list(self) -> List[str]:
        return [f.name for f in self.features]

    def to_dict(self) -> dict:
        return {
            "features": [f.to_dict() for f in self.features],
            "distance": self.distance._asdict(),
            "energy": self.energy._asdict(),
        }

    @classmethod
    def from_dict(cls, json: dict) -> FeaturePack:
        features = [Feature.from_dict(d) for d in json["features"]]
        distance = Feature.from_dict(json["distance"])
        energy = Feature.from_dict(json["energy"])
        return FeaturePack(features, distance, energy)
