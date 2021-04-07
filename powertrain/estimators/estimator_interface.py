from __future__ import annotations

from abc import abstractmethod

from pandas import DataFrame, Series

from powertrain.core.features import FeaturePack
from powertrain.utils.abc_utils import ABCMeta, abstract_attribute


class EstimatorInterface(metaclass=ABCMeta):
    """
    abstract base class for a routee-powertrain estimator
    """

    @abstract_attribute
    def model(self):
        pass

    @abstract_attribute
    def feature_pack(self) -> FeaturePack:
        pass

    @abstractmethod
    def train(self, data: DataFrame):
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
