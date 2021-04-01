from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

if TYPE_CHECKING:
    from powertrain import Model


def trip_average_error_weight(df, energy, trip_ids):
    trips_df = df.groupby(trip_ids).agg({energy: sum, 'energy_pred': sum})
    rate_error = (trips_df[energy] / trips_df[energy].sum()) * abs(trips_df[energy] - trips_df.energy_pred) / trips_df[
        energy]
    tae_w = rate_error.sum()
    return tae_w


def net_energy_error(target: np.ndarray, target_pred: np.ndarray):
    """Calculate the net energy prediction error over all
    links in the test dataset
    """
    net_e = np.sum(target)
    net_e_pred = np.sum(target_pred)
    net_error = (net_e_pred - net_e) / net_e
    return net_error


def energy_weighted_relative_percent_difference(target: np.ndarray, target_pred: np.ndarray):
    target = target.flatten()
    target_pred = target_pred.flatten()
    denom = np.abs(target) + np.abs(target_pred)
    high_mask = denom > 0.0000001
    low_mask = denom < -0.0000001
    mask = low_mask | high_mask

    y = target[mask]
    y_hat = target_pred[mask]

    w = np.array(np.abs(y) / np.sum(np.abs(y))).flatten()

    error_norm = np.abs(2 * ((y - y_hat) / (np.abs(y) + np.abs(y_hat))))

    ew_rpe = np.sum(error_norm * w)

    return ew_rpe


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

    Returns:

    """
    feature_pack = model.feature_pack
    target = np.array(test_df[feature_pack.energy.name])

    target_pred_series = model.predict(test_df)
    target_pred = np.array(target_pred_series)

    errors = {}

    rmse = np.sqrt(mean_squared_error(target, target_pred))
    errors['root_mean_squared_error'] = rmse

    mae = mean_absolute_error(target, target_pred)
    errors['mean_absolute_error'] = mae

    ew_rpe = energy_weighted_relative_percent_difference(target, target_pred)
    errors['energy_weighted_relative_percent_difference'] = ew_rpe

    if trip_column:
        test_df['energy_pred'] = target_pred_series
        trip_error = trip_average_error_weight(test_df, feature_pack.energy.name, trip_column)
        errors['trip_average_error_weight'] = trip_error

    errors['net_error'] = net_energy_error(target, target_pred)

    return errors
