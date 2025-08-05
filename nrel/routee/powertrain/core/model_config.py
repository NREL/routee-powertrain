from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


from nrel.routee.powertrain.core.features import (
    DataColumn,
    FeatureSet,
    FeatureSetId,
    TargetSet,
    feature_names_to_id,
)
from nrel.routee.powertrain.core.powertrain_type import PowertrainType


class PredictMethod(Enum):
    # Predict the rate of energy consumption and then multiply it by the distance.
    RATE = "rate"
    # Predict the total energy consumption for the link (including distance as a feature).
    RAW = "raw"

    @classmethod
    def from_string(cls, string: str) -> PredictMethod:
        if string.lower() == "rate":
            return PredictMethod.RATE
        elif string.lower() == "raw":
            return PredictMethod.RAW
        else:
            raise ValueError("Unknown predict method: {}".format(string))


@dataclass
class ModelConfig:
    ## vehicle information
    vehicle_description: str
    powertrain_type: PowertrainType

    ## estimator information

    # the model allows for multiple feature sets which will correspond to
    # different estimators. For example, you might have a feature set for
    # [speed, grade] which uses a specific estimator and another feature set
    # for [speed, grade, acceleration] which uses a different estimator.
    feature_sets: List[FeatureSet]
    distance: DataColumn
    target: TargetSet

    predict_method: PredictMethod = PredictMethod.RATE

    test_size: float = 0.2
    random_seed: int = 42

    trip_column: str = "trip_id"

    apply_real_world_adjustment: bool = True

    def __post_init__(self):
        # convert to the correct types
        if isinstance(self.feature_sets, dict):
            self.feature_sets = [FeatureSet.from_dict(self.feature_sets)]
        elif isinstance(self.feature_sets, list):
            feature_sets = []
            for f in self.feature_sets:
                if isinstance(f, FeatureSet):
                    feature_sets.append(f)
                elif isinstance(f, list):
                    feature_sets.append(FeatureSet(features=f))
                elif isinstance(f, dict):
                    feature_sets.append(FeatureSet.from_dict(f))
                else:
                    raise ValueError(
                        "Feature sets must be a list of FeatureSets, lists, or dicts"
                    )
            self.feature_sets = feature_sets
        elif isinstance(self.feature_sets, FeatureSet):
            self.feature_sets = [self.feature_sets]

        if isinstance(self.distance, dict):
            self.distance = DataColumn.from_dict(self.distance)

        if isinstance(self.target, dict):
            self.target = TargetSet.from_dict(self.target)
        elif isinstance(self.target, DataColumn):
            self.target = TargetSet([self.target])
        elif isinstance(self.target, list):
            self.target = TargetSet(self.target)

        if isinstance(self.powertrain_type, str):
            self.powertrain_type = PowertrainType.from_string(self.powertrain_type)

        if isinstance(self.predict_method, str):
            self.predict_method = PredictMethod.from_string(self.predict_method)

        # check to make sure the feature sets are unique
        feature_set_ids = [f.features_id for f in self.feature_sets]
        if len(feature_set_ids) != len(set(feature_set_ids)):
            raise ValueError(
                "Feature sets must have unique ids. Found duplicate ids: {}".format(
                    feature_set_ids
                )
            )

        # now check all the types
        if not isinstance(self.distance, DataColumn):
            raise ValueError("Distance must be a DataColumn")
        if not isinstance(self.target, TargetSet):
            raise ValueError("Target set must be a TargetSet")
        if not isinstance(self.feature_sets, list):
            raise ValueError("Feature sets must be a list")
        for feature_set in self.feature_sets:
            if not isinstance(feature_set, FeatureSet):
                raise ValueError("Feature sets must be a list of FeatureSets")
        if not isinstance(self.powertrain_type, PowertrainType):
            raise ValueError("Powertrain type must be a PowertrainType")
        if not isinstance(self.predict_method, PredictMethod):
            raise ValueError("Predict method must be a PredictMethod")

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**d)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["powertrain_type"] = self.powertrain_type.name
        d["feature_sets"] = [f.to_dict() for f in self.feature_sets]
        d["distance"] = self.distance.to_dict()
        d["target"] = self.target.to_dict()
        d["predict_method"] = self.predict_method.value

        return d

    @property
    def feature_set_map(self) -> Dict[FeatureSetId, FeatureSet]:
        return {f.features_id: f for f in self.feature_sets}

    def get_feature_set(self, feature_name_list: List[str]) -> Optional[FeatureSet]:
        """
        Get a feature set by a list of a feature names, returning None if it doesn't
        exist in the feature sets
        """
        feature_set_id = feature_names_to_id(feature_name_list)
        return self.feature_set_map.get(feature_set_id)

    @property
    def all_feature_names(self) -> List[str]:
        """
        Returns a list of all unique feature names in the feature sets
        """
        return [f.name for f in self.all_features]

    @property
    def all_features(self) -> List[DataColumn]:
        """
        Returns a list of all the unique features in the feature sets
        """
        all_features = []
        for feature_set in self.feature_sets:
            for feature in feature_set.features:
                if feature not in all_features:
                    all_features.append(feature)

        if self.predict_method == PredictMethod.RAW:
            all_features.append(self.distance)

        return all_features
