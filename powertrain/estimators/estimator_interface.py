from abc import ABC, abstractmethod

from pandas import DataFrame, Series


class EstimatorInterface(ABC):
    """
    abstract base class for a routee-powertrain estimator
    """

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
