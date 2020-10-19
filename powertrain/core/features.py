from enum import Enum
from typing import NamedTuple, Tuple, List


class Feature(NamedTuple):
    name: str
    units: str


class FeaturePack(NamedTuple):
    features: Tuple[Feature]
    distance_name: str
    distance_units: str
    energy_name: str
    energy_units: str

    @property
    def feature_list(self) -> List[str]:
        return [f.name for f in self.features]


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
        if string in ['raw_energy', 'energy_raw']:
            return PredictType.ENERGY_RAW
        elif string in ['rate_energy', 'energy_rate']:
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
