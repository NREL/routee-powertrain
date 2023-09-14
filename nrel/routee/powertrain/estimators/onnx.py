from __future__ import annotations
import base64
from pathlib import Path

import onnx
import onnxruntime as rt
import pandas as pd
from nrel.routee.powertrain.core.metadata import Metadata
from nrel.routee.powertrain.estimators.estimator_interface import Estimator


class ONNXEstimator(Estimator):
    onnx_model: onnx.ModelProto

    def __init__(self, onnx_model: onnx.ModelProto) -> None:
        self.onnx_model = onnx_model

    @classmethod
    def from_dict(cls, in_dict: dict) -> ONNXEstimator:
        onnx_model_raw = in_dict.get("onnx_model")
        if onnx_model_raw is None:
            raise ValueError("Model file must contain onnx model at key: 'onnx_model'")
        in_bytes = base64.b64decode(onnx_model_raw)
        onnx_model = onnx.load_from_string(in_bytes)
        return cls(onnx_model)

    def to_dict(self) -> dict:
        out_dict = {
            "onnx_model": base64.b64encode(self.onnx_model.SerializeToString()).decode(
                "utf-8"
            )
        }
        return out_dict

    @classmethod
    def from_file(cls, filepath: str | Path) -> ONNXEstimator:
        filepath = Path(filepath)
        if filepath.suffix != ".onnx":
            raise ValueError("ONNX model must be saved as a .onnx file")
        with filepath.open("rb") as f:
            onnx_model = onnx.load_from_string(f.read())
        return cls(onnx_model)

    def to_file(self, filepath: str | Path):
        filepath = Path(filepath)
        if filepath.suffix != ".onnx":
            raise ValueError("ONNX model must be saved as a .onnx file")
        with filepath.open("wb") as f:
            f.write(self.onnx_model.SerializeToString())

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
