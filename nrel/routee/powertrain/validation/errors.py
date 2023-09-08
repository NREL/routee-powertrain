from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

if TYPE_CHECKING:
    from nrel.routee.powertrain.core.model import Model


def net_energy_error(target: np.ndarray, target_pred: np.ndarray) -> float:
    net_e = np.sum(target)
    net_e_pred = np.sum(target_pred)
    net_error = (net_e_pred - net_e) / net_e
    return net_error


def weighted_relative_percent_difference(
    target: np.ndarray, target_pred: np.ndarray
) -> float:
    epsilon = np.finfo(np.float64).eps

    w = np.array(np.abs(target) / np.sum(np.abs(target)))

    error_norm = np.abs(
        2
        * (
            (target - target_pred)
            / np.maximum((np.abs(target) + np.abs(target_pred)), epsilon)
        )
    )

    mean_error = np.average(error_norm, weights=w)

    return mean_error


def relative_percent_difference(target: np.ndarray, target_pred: np.ndarray) -> float:
    epsilon = np.finfo(np.float64).eps

    error_norm = np.abs(
        2
        * (
            (target - target_pred)
            / np.maximum((np.abs(target) + np.abs(target_pred)), epsilon)
        )
    )

    mean_error = np.average(error_norm)

    return mean_error


def compute_errors(test_df: pd.DataFrame, model: Model) -> Dict[str, float]:
    """
    Computes the error metrics for a set of predictions relative
    to the ground truth data

    Args:
        test_df: the test dataframe
        model: the routee-powertrain model
        trip_column: an optional trip column for computing trip level metrics

    Returns: a dictionary with all of the error values

    """
    test_df = test_df.copy()

    feature_pack = model.feature_pack
    energy_names = feature_pack.energy_name_list

    if len(energy_names) > 1:
        raise NotImplementedError(
            "compute_errors currently only supports models with a single energy target"
        )

    energy_name = energy_names[0]

    target = np.array(test_df[energy_name])
    target_pred = np.array(model.predict(test_df))

    errors = {}

    rmse = np.sqrt(mean_squared_error(target, target_pred))
    errors["link_root_mean_squared_error"] = rmse
    errors["link_norm_root_mean_squared_error"] = rmse / (
        sum(test_df[energy_name]) / len(test_df)
    )

    ew_rpe = weighted_relative_percent_difference(target, target_pred)
    errors["link_weighted_relative_percent_difference"] = ew_rpe

    trip_column = model.metadata.config.trip_column

    if trip_column in test_df.columns:
        test_df["energy_pred"] = target_pred
        gb = test_df.groupby(trip_column).agg(
            {energy_name: "sum", "energy_pred": "sum"}
        )
        t_rpd = relative_percent_difference(gb[energy_name], gb["energy_pred"])
        t_wrpd = weighted_relative_percent_difference(
            gb[feature_pack.energy.name], gb["energy_pred"]
        )
        t_rmse = np.sqrt(mean_squared_error(gb[energy_name], gb["energy_pred"]))

        errors["trip_relative_percent_difference"] = t_rpd
        errors["trip_weighted_relative_percent_difference"] = t_wrpd
        errors["trip_root_mean_squared_error"] = t_rmse
        errors["trip_norm_root_mean_squared_error"] = t_rmse / (
            sum(gb[energy_name]) / len(gb)
        )

    errors["net_error"] = net_energy_error(target, target_pred)

    total_dist = test_df[feature_pack.distance.name].sum()

    pred_energy = np.sum(target_pred)
    actual_energy = np.sum(target)

    errors["actual_dist_per_energy"] = total_dist / actual_energy
    errors["pred_dist_per_energy"] = total_dist / pred_energy

    return errors
