from __future__ import annotations
import base64
from pathlib import Path

import onnx
import onnxruntime as rt
import pandas as pd
from nrel.routee.powertrain.core.features import DataColumn, FeatureSet, TargetSet
from nrel.routee.powertrain.estimators.estimator_interface import Estimator

ONNX_INPUT_NAME = "input"
ONNX_DTYPE = "float32"


class ONNXEstimator(Estimator):
    onnx_model: onnx.ModelProto
    session: rt.InferenceSession

    def __init__(self, onnx_model: onnx.ModelProto) -> None:
        self.onnx_model = onnx_model
        session = rt.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        self.session = session

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

    def predict(
        self,
        links_df: pd.DataFrame,
        feature_set: FeatureSet,
        distance: DataColumn,
        target_set: TargetSet,
    ) -> pd.DataFrame:
        x = links_df[feature_set.feature_name_list].values

        raw_energy_pred_rates = self.session.run(
            None, {ONNX_INPUT_NAME: x.astype(ONNX_DTYPE)}
        )[0]

        energy_df = pd.DataFrame(index=links_df.index)

        for i, energy in enumerate(target_set.targets):
            energy_pred_rates = pd.Series(
                raw_energy_pred_rates[:, i], index=links_df.index
            )

            energy_pred = energy_pred_rates * links_df[distance.name]
            energy_df[energy.name] = energy_pred

        return energy_df
