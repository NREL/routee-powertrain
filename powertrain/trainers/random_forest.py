import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor

from powertrain.core.metadata import Metadata
from powertrain.core.model import VehicleModel
from powertrain.trainers.trainer import Trainer


class RandomForestTrainer(Trainer):
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 10,
        n_estimators: int = 20,
        random_state: int = 52,
        cores: int = 4,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.cores = cores

    def inner_train(
        self, features: pd.DataFrame, target: pd.DataFrame, metadata: Metadata
    ) -> VehicleModel:
        """
        Uses a random forest to predict the energy rate values
        """
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_jobs=self.cores,
            random_state=self.random_state,
        )

        rf.fit(features.values, target.values)

        # convert to ONNX
        n_features = len(features.columns)
        initial_type = [(metadata.onnx_input_name, FloatTensorType([None, n_features]))]
        onnx_model = to_onnx(rf, initial_types=initial_type)

        vehicle_model = VehicleModel.build(onnx_model, metadata)

        return vehicle_model
