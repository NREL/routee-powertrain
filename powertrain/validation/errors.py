"""Errors submodule contains the standard functions for calculating errors
on model classes after training.

Each function expects particular variables and column names within the model
classes.

Args:
    model: (model class), 'test' dataframe in model must have columns
            ['rate', energy, 'rate_pred', trip_ids]

Returns:
    model: Energy model class with error variables
    
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def _link_average_error_unweight(target, target_pred):
    """Calculate the median error on links without weighting
    by distance or energy consumption
    """
    rate_error = (target - target_pred) / target
    lae_uw = np.median(np.abs(rate_error))
    return lae_uw


def _trip_average_error_weight(df, energy, trip_ids):
    """Calculate the median error on links without weighting
    by distance or energy consumption
    """
    trips_df = df.groupby(trip_ids).agg({energy: sum, 'energy_pred': sum})
    rate_error = (trips_df[energy] / trips_df[energy].sum()) * abs(trips_df[energy] - trips_df.energy_pred) / trips_df[
        energy]
    tae_w = rate_error.sum()
    return tae_w


def _net_energy_error(target, target_pred):
    """Calculate the net energy prediction error over all
    links in the test dataset
    """
    net_e = np.sum(target)
    net_e_pred = np.sum(target_pred)
    net_error = (net_e_pred - net_e) / net_e
    return net_error


def _distance_weighted_mean_absolute_error(target, target_pred, distance):
    distance_weights = np.array(distance).flatten()

    return mean_absolute_error(target, target_pred, sample_weight=distance_weights)

def _relative_percent_difference(target, target_pred):
    target = target.flatten()
    target_pred = target_pred.flatten()
    denom = np.abs(target) + np.abs(target_pred)
    high_mask = denom > 0.0000001
    low_mask = denom < -0.0000001
    mask = low_mask | high_mask

    y = target[mask]
    y_hat = target_pred[mask]

    error_norm = np.abs(2 * ((y - y_hat) / (np.abs(y) + np.abs(y_hat))))

    rpe = np.mean(error_norm)

    return rpe

def _energy_weighted_relative_percent_difference(target, target_pred):
    target = target.flatten()
    target_pred = target_pred.flatten()
    denom = np.abs(target) + np.abs(target_pred)
    high_mask = denom > 0.0000001
    low_mask = denom < -0.0000001
    mask = low_mask | high_mask

    y = target[mask]
    y_hat = target_pred[mask]

    w = np.array(y / np.sum(y)).flatten()

    error_norm = np.abs(2 * ((y - y_hat) / (np.abs(y) + np.abs(y_hat))))

    ew_rpe = np.sum(error_norm * w)

    return ew_rpe

def _distance_weighted_relative_percent_difference(target, target_pred, distance):
    distance_weights = np.array(distance / np.sum(distance)).flatten()
    target = target.flatten()
    target_pred = target_pred.flatten()
    denom = np.abs(target) + np.abs(target_pred)
    high_mask = denom > 0.0000001
    low_mask = denom < -0.0000001
    mask = low_mask | high_mask

    y = target[mask]
    y_hat = target_pred[mask]
    w = distance_weights[mask]

    error_norm = np.abs(2 * ((y - y_hat) / (np.abs(y) + np.abs(y_hat))))

    dw_rpe = np.sum(error_norm * w)

    return dw_rpe


def _distance_weighted_mean_absolute_percentage_error(target, target_pred, distance):
    distance_weights = np.array(distance / np.sum(distance)).flatten()
    target = target.flatten()
    target_pred = target_pred.flatten()
    floor_mask = target > 0.000000000001
    y = target[floor_mask]
    y_hat = target_pred[floor_mask]
    w = distance_weights[floor_mask]
    n = len(y)
    error_norm = np.abs((y - y_hat) / y)
    dw_error_norm = error_norm * w

    dw_mape = np.sum(dw_error_norm)

    return dw_mape


def all_error(target, target_pred, distance):
    '''Wrapper function to calculate all error metrics the trained model.
    
    Args:
        target (list):
            List of floats that represent the ground-truth
            target feature energy data.
        target_pred (list):
            List of floats representing the RouteE model's
            energy consumption predictions.
        distance (numpy.array):
            Array of corresponding distance values.
    
    Returns:
        errors (dict):
            Dictionary of each error output measure.
            
    '''
    # Link average error - unweighted
    target = np.array(target)
    target_pred = np.array(target_pred)
    errors = {}

    rmse = np.sqrt(mean_squared_error(target, target_pred))
    errors['root_mean_squared_error'] = rmse

    mae = mean_absolute_error(target, target_pred)
    errors['mean_absolute_error'] = mae

    dw_mae = _distance_weighted_mean_absolute_error(target, target_pred, distance)
    errors['distance_weighted_mean_absolute_error'] = dw_mae

    dw_mape = _distance_weighted_mean_absolute_percentage_error(target, target_pred, distance)
    errors['distance_weighted_mean_absolute_percentage_error'] = dw_mape
    # lae_uw = _link_average_error_unweight(target, target_pred)
    # errors['_link_average_error_unweight'] = lae_uw

    dw_rpe = _distance_weighted_relative_percent_difference(target, target_pred, distance)
    errors['distance_weighted_relative_percent_difference'] = dw_rpe

    ew_rpe = _energy_weighted_relative_percent_difference(target, target_pred)
    errors['energy_weighted_relative_percent_difference'] = ew_rpe

    rpe = _relative_percent_difference(target, target_pred)
    errors['_relative_percent_difference'] = rpe
    # Link average error - weighted
    # TODO: Think about how to caculate this without explicitly passing trip ids.
    # errors['_trip_average_error_weight'] = _trip_average_error_weight(test_df,
    #                                                             metadata['distance'],
    #                                                             metadata['trip_ids'])

    # Net energy error
    errors['net_error'] = _net_energy_error(target, target_pred)

    return errors
