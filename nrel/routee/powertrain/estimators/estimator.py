from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from nrel.routee.powertrain.core.metadata import Metadata


class Estimator(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, in_dict: dict) -> Estimator:
        """
        Load an estimator from a bytes object in memory
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Serialize an estimator to a python dictionary
        """

    @abstractmethod
    def predict(self, links_df: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
        """
        Predict absolute energy consumption for each link
        """
        pass
