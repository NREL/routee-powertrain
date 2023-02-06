from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pandas import DataFrame, Series

from powertrain.core.features import FeaturePack


class EstimatorInterface(ABC):
    """
    abstract base class for a routee-powertrain estimator
    """

    model: Any
    feature_pack: FeaturePack

    @abstractmethod
    def train(self, data: DataFrame, **kwargs):
        """
        abstract train method

        Args:
            data:

        Returns:

        """

    @abstractmethod
    def predict(self, data: DataFrame) -> Series:
        """
        abstract predict method

        Args:
            data:

        Returns:

        """

    @abstractmethod
    def to_json(self) -> dict:
        """
        method to serialize all necessary data to json for model persistence
        Returns:

        """

    @classmethod
    @abstractmethod
    def from_json(cls, json: dict) -> EstimatorInterface:
        """
        method to serialize all necessary data to json for model persistence
        Returns:

        """
