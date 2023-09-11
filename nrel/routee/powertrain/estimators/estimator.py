from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from nrel.routee.powertrain.core.metadata import Metadata


class Estimator(ABC):
    @classmethod
    @abstractmethod
    def from_bytes(cls, in_bytes: bytes) -> Estimator:
        """
        Load an estimator from a bytes object in memory
        """

    @abstractmethod
    def to_bytes(self) -> bytes:
        """
        Serialize an estimator to a bytes object in memory
        """

    @abstractmethod
    def predict(self, links_df: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
        """
        Predict absolute energy consumption for each link
        """
        pass
