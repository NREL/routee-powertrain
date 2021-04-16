from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

if TYPE_CHECKING:
    from powertrain import Model


def net_energy_error(target: np.ndarray, target_pred: np.ndarray) -> float:
    net_e = np.sum(target)
    net_e_pred = np.sum(target_pred)
    net_error = (net_e_pred - net_e) / net_e
    return net_error


def weighted_relative_percent_difference(target: np.ndarray, target_pred: np.ndarray) -> float:
    epsilon = np.finfo(np.float64).eps

    w = np.array(np.abs(target) / np.sum(np.abs(target)))

    error_norm = np.abs(2 * ((target - target_pred) / np.maximum((np.abs(target) + np.abs(target_pred)), epsilon)))

    mean_error = np.average(error_norm, weights=w)

    return mean_error


def relative_percent_difference(target: np.ndarray, target_pred: np.ndarray) -> float:
    epsilon = np.finfo(np.float64).eps

    error_norm = np.abs(2 * ((target - target_pred) / np.maximum((np.abs(target) + np.abs(target_pred)), epsilon)))

    mean_error = np.average(error_norm)

    return mean_error


def compute_errors(
        test_df: pd.DataFrame,
        model: Model,
        trip_column: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the error metrics for a set of predictions relative to the ground truth data

    Args:
        test_df: the test dataframe
        model: the routee-powertrain model
        trip_column: an optional trip column for computing trip level metrics

    Returns: a dictionary with all of the error values

    """
    feature_pack = model.feature_pack

    target = np.array(test_df[feature_pack.energy.name])
    target_pred = np.array(model.predict(test_df))

    errors = {}

    rmse = np.sqrt(mean_squared_error(target, target_pred))
    errors['link_root_mean_squared_error'] = rmse

    ew_rpe = weighted_relative_percent_difference(target, target_pred)
    errors['link_weighted_relative_percent_difference'] = ew_rpe

    if trip_column:
        test_df['energy_pred'] = target_pred
        gb = test_df.groupby(trip_column).agg({feature_pack.energy.name: sum, 'energy_pred': sum})
        t_rpd = relative_percent_difference(gb[feature_pack.energy.name], gb['energy_pred'])
        t_wrpd = weighted_relative_percent_difference(gb[feature_pack.energy.name], gb['energy_pred'])
        errors['trip_relative_percent_difference'] = t_rpd
        errors['trip_weighted_relative_percent_difference'] = t_wrpd

    errors['net_error'] = net_energy_error(target, target_pred)

    return errors
