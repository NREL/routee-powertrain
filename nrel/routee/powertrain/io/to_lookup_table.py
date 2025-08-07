from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

from nrel.routee.powertrain.core.features import feature_names_to_id

if TYPE_CHECKING:
    from nrel.routee.powertrain.core.model import Model


@dataclass
class LookupTableFeatureParameter:
    feature_name: str
    lower_bound: float
    upper_bound: float
    n_samples: int

    @classmethod
    def from_dict(cls, d: dict) -> LookupTableFeatureParameter:
        if "feature_name" not in d:
            raise ValueError("must provide feature name when building from dictionary")
        elif "lower_bound" not in d:
            raise ValueError("must provide lower bound when building from dictionary")
        elif "upper_bound" not in d:
            raise ValueError("must provide upper bound when building from dictionary")
        elif "n_samples" not in d:
            raise ValueError("must provide n_samples when building from dictionary")

        lower_bound = float(d["lower_bound"])
        upper_bound = float(d["upper_bound"])

        if lower_bound >= upper_bound:
            raise ValueError("lower bound must be less than upper bound")

        return LookupTableFeatureParameter(
            feature_name=d["feature_name"],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            n_samples=int(d["n_samples"]),
        )


def to_lookup_table(
    model: "Model",
    feature_parameters: list[dict],
    energy_target: str,
) -> pd.DataFrame:
    """
    Convert the model to a lookup table for the given estimator id and feature parameters.
    """
    if energy_target not in model.metadata.config.target.target_name_list:
        raise KeyError(
            f"Model does not have a target named {energy_target}. "
            f"Here are the available targets: {model.metadata.config.target.target_name_list}"
        )

    parsed_feature_parameters = [
        LookupTableFeatureParameter.from_dict(fp) for fp in feature_parameters
    ]

    feature_names_list = [fp.feature_name for fp in parsed_feature_parameters]
    feature_set_id = feature_names_to_id(feature_names_list)
    feature_set = model.metadata.config.feature_set_map.get(feature_set_id)
    if feature_set is None:
        raise KeyError(
            f"Model does not have a feature set with the features: {feature_names_list}."
            f"Here are the available feature sets: {model.feature_set_lists}"
        )

    estimator = model.estimators.get(feature_set_id)
    if estimator is None:
        raise KeyError(f"Model does not have an estimator for {feature_set_id}")

    points = tuple(
        np.linspace(f.lower_bound, f.upper_bound, f.n_samples)
        for f in parsed_feature_parameters
    )
    mesh = np.meshgrid(*points)

    pred_input = np.stack(list(map(np.ravel, mesh)), axis=1)

    pred_df = pd.DataFrame(pred_input, columns=feature_names_list)

    pred_df[model.metadata.config.distance.name] = 1

    predictions = estimator.predict(
        pred_df,
        feature_set,
        model.metadata.config.distance,
        model.metadata.config.target,
        model.metadata.config.predict_method,
    )

    lookup = pred_df.drop(columns=[model.metadata.config.distance.name])

    energy_column_key = f"{energy_target}_per_{model.metadata.config.distance.name}"

    lookup[energy_column_key] = predictions[energy_target]

    return lookup
