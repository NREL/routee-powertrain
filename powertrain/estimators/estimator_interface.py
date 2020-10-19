from pandas import DataFrame, Series

from abc import abstractmethod

from powertrain.utils.abc_utils import ABCMeta, abstract_attribute
from powertrain.core.features import FeaturePack, PredictType


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

    @abstract_attribute
    def predict_type(self) -> PredictType:
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
