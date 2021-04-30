import logging
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pathlib import Path
import traceback
from typing import Dict, Optional, Tuple

from powertrain.core.model import Model

log = logging.getLogger(__name__)


def visualize_features(
        model: Model,
        feature_ranges: Dict[str, dict],
        num_links: int,
        int_features: Optional[Tuple] = (),
        output_path: Optional[str] = None) -> dict:
    """
    takes a model and generates test links to independently test the model's features
    and creates plots of those predictions

    :param model: the model to be tested
    :param feature_ranges: a dictionary with value ranges to generate test links
    :param num_links: the number of test links or data points the model will predict over
    :param int_features: optional tuple of feature names which will have their links generated in integer increments
    :param output_path: if not none, saves results to this location. Else the plots are displayed rather than saved
    :return: a dictionary containing the predictions where the key is the feature tested
    :raises Exception due to IOErrors, missing keys in model, or missing config values
    """

    # grab the necessary metadata from the model
    feature_meta = model.metadata.estimator_features['features']
    distance_name = model.metadata.estimator_features['distance']['name']
    distance_units = model.metadata.estimator_features['distance']['units']
    energy_units = model.metadata.estimator_features['energy']['units']
    model_name = model.metadata.model_description
    estimator_name = model.metadata.estimator_name

    feature_dict = {}
    for feature in feature_meta: feature_dict[feature['name']] = feature['units']

    # check that all features in the metadata are present in the config
    # if any features are missing in config, throw an error
    if not all(feature in feature_ranges.keys() for feature in feature_dict.keys()):
        missing_features = set(feature_dict.keys()) - set(feature_ranges.keys())
        raise KeyError(f'feature range config is missing {missing_features} for model {model_name} {estimator_name}')

    # dict for holding the prediction series
    predictions = {}

    # for each feature test it individually using the values form the visualization feature ranges
    for current_feature, current_units in feature_dict.items():

        # setup a set of test links
        # make <num_links> number of links
        # using the feature range config, generate evenly spaced ascending values for the current feature
        links_df = DataFrame()
        if current_feature in int_features:
            difference = int(feature_ranges[current_feature]['max'] - feature_ranges[current_feature]['min'])
            links_df[current_feature] = np.arange(start=feature_ranges[current_feature]['min'],
                                                  stop=feature_ranges[current_feature]['max'] + 1,
                                                  step=round(difference / min([num_links, difference])))
        else:
            links_df[current_feature] = np.linspace(feature_ranges[current_feature]['min'],
                                                    feature_ranges[current_feature]['max'],
                                                    num=num_links)
        # for every other feature, set it to its default value for the all links
        for other_feature in feature_dict.keys():
            if other_feature != current_feature:
                links_df[other_feature] = [(feature_ranges[other_feature]['default'])] * len(links_df)
        # set distance to be a constant and label it with the distance name found in the metadata
        links_df[distance_name] = [.1] * len(links_df)

        # make a prediction using the test links
        try:
            prediction = model.predict(links_df)
        except:
            log.error(f'unable to predict {current_feature} with model {model_name} {estimator_name} due to ERROR:')
            log.error(f" {traceback.format_exc()}")
            log.error(f"{current_feature} plot for model {model_name} {estimator_name} skipped..")
            continue

        # plot the prediction and save the figure
        plt.plot(links_df[current_feature],
                 prediction * 100 / links_df[distance_name],
                 label=model_name)
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
