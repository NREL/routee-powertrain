from __future__ import annotations

import onnx
import onnxruntime as rt
import pandas as pd
from nrel.routee.powertrain.core.metadata import Metadata
from nrel.routee.powertrain.estimators.estimator import Estimator

# we're pinning the onnx opset to 13 since the rust onnxruntime crate
# is built from onnx runtime version 1.8 which only supports opset 13
# see here: https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
ONNX_OPSET_VERSION = 13

class ONNXEstimator(Estimator):
    onnx_model: onnx.ModelProto

    def __init__(self, onnx_model: onnx.ModelProto) -> None:
        self.onnx_model = onnx_model

    @classmethod
    def from_bytes(cls, in_bytes: bytes) -> ONNXEstimator:
        onnx_model = onnx.load_from_string(in_bytes)
        return cls(onnx_model)

    def to_bytes(self) -> bytes:
        return self.onnx_model.SerializeToString()

    def predict(self, links_df: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
        config = metadata.config

        distance_col = config.feature_pack.distance.name

        x = links_df[config.feature_pack.feature_name_list].values

        onnx_session = rt.InferenceSession(self.onnx_model.SerializeToString())

        raw_energy_pred_rates = onnx_session.run(
            None, {config.onnx_input_name: x.astype(config.feature_dtype)}
        )[0]

        energy_df = pd.DataFrame(index=links_df.index)

        for i, energy in enumerate(config.feature_pack.energy):
            energy_pred_rates = pd.Series(
                raw_energy_pred_rates[:, i], index=links_df.index
            )

            energy_pred = energy_pred_rates * links_df[distance_col]
            energy_df[energy.name] = energy_pred

        return energy_df
