import logging
from abc import ABC, abstractmethod

import pandas as pd

from powertrain.core.metadata import Metadata
from powertrain.core.model import VehicleModel
from powertrain.trainers.utils import test_train_split
from powertrain.validation.errors import compute_errors

ENERGY_RATE_NAME = "energy_rate"

log = logging.getLogger(__name__)


class Trainer(ABC):
    def train(self, data: pd.DataFrame, metadata: Metadata) -> VehicleModel:
        """
        A wrapper for inner train that does some pre and post processing.
        """
        energy_name = metadata.feature_pack.energy.name
        distance_name = metadata.feature_pack.distance.name

        data[ENERGY_RATE_NAME] = data[energy_name] / data[distance_name]

        filtered_data = data[
            (data[ENERGY_RATE_NAME] > metadata.energy_rate_low_limit)
            & (data[ENERGY_RATE_NAME] < metadata.energy_rate_high_limit)
        ]
        filtered_rows = len(data) - len(filtered_data)
        log.info(
            f"filtered out {filtered_rows} rows with energy rates outside "
            f"of the limits of {metadata.energy_rate_low_limit} "
            f"and {metadata.energy_rate_high_limit}"
        )

        train, test = test_train_split(
            filtered_data, test_size=metadata.test_size, seed=metadata.random_seed
        )
        features = train[metadata.feature_pack.feature_list]
        if features.isnull().values.any():
            raise ValueError("Features contain null values")

        target = train[ENERGY_RATE_NAME]
        if target.isnull().values.any():
            raise ValueError(
                "Target contains null values. Try decreasing the energy rate high limit"
            )

        model = self.inner_train(features=features, target=target, metadata=metadata)

        model_errors = compute_errors(test, model)

        model_with_errors = model.set_errors(model_errors)

        return model_with_errors

    @abstractmethod
    def inner_train(
        self, features: pd.DataFrame, target: pd.DataFrame, metadata: Metadata
    ) -> VehicleModel:
        """
        Trains a vehicle model from a set of data.
        """
        pass
