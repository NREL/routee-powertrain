from __future__ import annotations

import warnings
from typing import NamedTuple

from powertrain.core.powertrain_type import PowertrainType
from powertrain.utils.fs import get_version


class Metadata(NamedTuple):
    """
    A named tuple carrying model metadata information
    """

    model_description: str

    estimator_name: str
    estimator_features: dict

    routee_version: str

    powertrain_type: PowertrainType

    errors: dict = {}

    def set_errors(self, errors: dict) -> Metadata:
        return self._replace(errors=errors)

    def to_json(self) -> dict:
        d = self._asdict()
        d['powertrain_type'] = self.powertrain_type.name

        return d

    @classmethod
    def from_json(cls, j: dict) -> Metadata:
        if 'estimator_predict_type' in j:
            warnings.warn(
                "this model contains an estimator_predict_type which has been deprecated and will be ignored.")
            del (j['estimator_predict_type'])

        v = get_version()
        if j['routee_version'] != v:
            warnings.warn(f"this model was trained using routee-powertrain version {j['routee_version']}"
                          f" but you're using version {v}")

        j['powertrain_type'] = PowertrainType.from_string(j.get('powertrain_type'))

        return Metadata(**j)
