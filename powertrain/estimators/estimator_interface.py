from abc import ABC, abstractmethod

from pandas import DataFrame, Series

from powertrain.core.features import FeaturePack


class EstimatorInterface(ABC):
    """
    abstract base class for a routee-powertrain estimator
    """

    @abstractmethod
    def train(self, data: DataFrame, feature_pack: FeaturePack):
        """
        abstract train method

        Args:
            data:
            feature_pack:

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
