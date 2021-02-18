from __future__ import annotations

from typing import NamedTuple


class Metadata(NamedTuple):
    """
    A named tuple carrying model metadata infomration
    """

    model_description: str

    estimator_name: str
    estimator_features: dict
    estimator_predict_type: str

    routee_version: str

    errors: dict = {}

    def set_errors(self, errors: dict) -> Metadata:
        return self._replace(errors=errors)

    def to_json(self) -> dict:
        return self._asdict()

    @classmethod
    def from_json(cls, j: dict) -> Metadata:
        return Metadata(**j)
