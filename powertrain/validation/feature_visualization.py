import logging
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from powertrain.core.model import Model

log = logging.getLogger(__name__)


def visualize_features(
        model: Model,
        feature_ranges: Dict[str, dict],
        output_path: Optional[str] = None) -> dict:
    """
    takes a model and generates test links to independently test the model's features
    and creates plots of those predictions

    :param model: the model to be tested
    :param feature_ranges: a dictionary with value ranges to generate test links
    :param output_path: if not none, saves results to this location. Else the plots are displayed rather than saved
    :return: a dictionary containing the predictions where the key is the feature tested
    :raises Exception due to IOErrors, KeyError due to missing features ranges required by the model
    """

    # grab the necessary metadata from the model
    feature_meta = model.metadata.estimator_features['features']
    distance_name = model.metadata.estimator_features['distance']['name']
    distance_units = model.metadata.estimator_features['distance']['units']
    energy_units = model.metadata.estimator_features['energy']['units']
    model_name = model.metadata.model_description
    estimator_name = model.metadata.estimator_name

    feature_units_dict = {}
    for feature in feature_meta: feature_units_dict[feature['name']] = feature['units']

    # check that all features in the metadata are present in the config
    # if any features are missing in config, throw an error
    if not all(feature in feature_ranges.keys() for feature in feature_units_dict.keys()):
        missing_features = set(feature_units_dict.keys()) - set(feature_ranges.keys())
        raise KeyError(f'feature range is missing {missing_features} for model {model_name} {estimator_name}')

    # dict for holding the prediction series
    predictions = {}

    # for each feature test it individually using the values form the visualization feature ranges
    for current_feature, current_units in feature_units_dict.items():

        # setup a set of test links
        # make <num_links> number of links
        # using the feature range config, generate evenly spaced ascending values for the current feature
        sample_points = []
        for feature in feature_units_dict.keys():
            points = np.linspace(feature_ranges[feature]['min'],
                                 feature_ranges[feature]['max'],
                                 feature_ranges[feature]['steps'])
            sample_points.append(points)

        mesh = np.meshgrid(*sample_points)

        pred_input = np.stack(list(map(np.ravel, mesh)), axis=1)

        links_df = DataFrame(pred_input, columns=[f for f in feature_units_dict.keys()])

        # set distance to be a constant and label it with the distance name found in the metadata
        links_df[distance_name] = [.1] * len(links_df)

        # make a prediction using the test links
        try:
            links_df['energy_pred'] = model.predict(links_df)
        except:
            log.error(f'unable to predict {current_feature} with model {model_name} {estimator_name} due to ERROR:')
            log.error(f" {traceback.format_exc()}")
            log.error(f"{current_feature} plot for model {model_name} {estimator_name} skipped..")
            continue

        # plot the prediction and save the figure
        prediction = links_df.groupby(current_feature).energy_pred.mean()

        prediction.plot()
        plt.title(f'{estimator_name} [{current_feature}]')
        plt.xlabel(f'{current_feature} [{current_units}]')
        plt.ylabel(f'{energy_units}/100{distance_units}')

        # if an output filepath is specified, save th results instead of displaying them
        if output_path is not None:
            try:
                Path(output_path).joinpath(f'{model_name}').mkdir(parents=True, exist_ok=True)
                plt.savefig(Path(output_path).joinpath(f'{model_name}/{estimator_name}_[{current_feature}].png'),
                            format='png')
            except:
                log.error(f'unable to save plot for {current_feature} with model {model_name} {estimator_name} due to '
                          f'ERROR:')
                log.error(f" {traceback.format_exc()}")
                log.error(f"{current_feature} plot for model {model_name} {estimator_name} skipped..")
        else:
            plt.show()

        plt.clf()
        predictions[current_feature] = prediction

    return predictions


def contour_plot(model: Model,
                 x_feature: str,
                 y_feature: str,
                 feature_ranges: {str, dict},
                 output_path: Optional[str] = None):
    """
        takes a model and generates a contour plot of the two test features: x_Feature and y_feature.

        :param model: the model to be tested
        :param x_feature: one of the features used to generate the energy matrix and will be the x-axis feature
        :param y_feature: one of the features used to generate the energy matrix and will be the y-axis feature
        :param feature_ranges: a dictionary with value ranges to generate test links
        :param output_path: if not none, saves results to this location. Else the plot is displayed rather than saved
        :raises Exception due to IOErrors, KeyError due to missing features ranges required by the model,
        KeyError due to incompatible x/y features
        """
    # get the necessary information from the metadata
    feature_meta = model.metadata.estimator_features['features']
    distance_name = model.metadata.estimator_features['distance']['name']
    model_name = model.metadata.model_description

    # get all of the feature units from the metadata
    feature_units_dict = {}
    for feature in feature_meta: feature_units_dict[feature['name']] = feature['units']

    # check to make sure feature range has all the features required by the model
    if not all(feature in feature_ranges.keys() for feature in feature_units_dict.keys()):
        missing_features = set(feature_units_dict.keys()) - set(feature_ranges.keys())
        raise KeyError(f'feature range is missing {missing_features} for model {model_name}')

    # check that both of the test features are supported by the model
    if not all(feature in feature_units_dict.keys() for feature in [x_feature, y_feature]):
        missing_features = {x_feature, y_feature} - set(feature_units_dict.keys())
        raise KeyError(f'model {model_name} does not support the feature(s): {missing_features}')

    points = {
        n: np.linspace(
            f['min'],
            f['max'],
            f['steps'],
        ) for n, f in feature_ranges.items()
    }

    mesh = np.meshgrid(*[v for v in points.values()])

    pred_input = np.stack(list(map(np.ravel, mesh)), axis=1)

    df = DataFrame(pred_input, columns=[n for n in feature_ranges.keys()])

    df[distance_name] = 1

    df['energy'] = model.predict(df)

    energy_matrix = df.groupby([y_feature, x_feature]).energy.mean().unstack().values

    xx, yy = np.meshgrid(points[x_feature], points[y_feature])

    plt.figure(figsize=(10, 8))
    g = plt.contourf(xx, yy, energy_matrix, levels=50)
    plt.colorbar(g)
    plt.xlabel(f"{x_feature} ({feature_units_dict[x_feature]})")
    plt.ylabel(f"{y_feature} ({feature_units_dict[y_feature]})")
    plt.title(f"energy consumption rate vs {x_feature} and {y_feature}")

    # if an output filepath is specified, save th results instead of displaying them
    if output_path is not None:
        try:
            plt.savefig(Path(output_path).joinpath(f'{model_name}_[{x_feature}_{y_feature}].png'),
                        format='png')
        except:
            log.error(f'unable to save contour plot for {model_name} due to ERROR:')
            log.error(f" {traceback.format_exc()}")
    else:
        plt.show()

    plt.close()
