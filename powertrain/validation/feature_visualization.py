import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pathlib import Path
from typing import Dict

from powertrain.core.model import Model


def visualize_features(
        model: Model,
        feature_ranges: Dict[str, dict],
        output_filepath: Path,
        num_links: int) -> dict:
    """
    takes a model and generates test links to independently test the model's features
    and creates plots of those predictions

    :param model: the model to be tested
    :param feature_ranges: a dictionary with value ranges to generate test links
    :param output_filepath: where to store the results
    :param num_links: the number of test links or data points the model will predict over
    :return: a dictionary containing the predictions where the key is the feature tested
    :raises Exception due to IOErrors, missing keys in model, or missing config values
    """

    # grab the necessary metadata from the model
    feature_meta = model.metadata.estimator_features['features']
    distance_units = model.metadata.estimator_features['distance']['name']
    energy_units = model.metadata.estimator_features['energy']['units']
    model_name = model.metadata.model_description

    feature_dict = {}
    for feature in feature_meta: feature_dict[feature['name']] = feature['units']

    # check that all features in the metadata are present in the config
    # if any features are missing in config, throw an error
    if not all(feature in feature_ranges.keys() for feature in feature_dict.keys()):
        missing_features = set(feature_dict.keys()) - set(feature_ranges.keys())
        raise KeyError(f"feature range config is missing {missing_features} for model {model_name}")

    # dict for holding the prediction series
    predictions = {}

    # for each feature test it individually using the values form the visualization feature ranges
    for current_feature, current_units in feature_dict.items():

        # setup a set of test links
        # make <num_links> number of links
        # using the feature range config, generate evenly spaced ascending values for the current feature
        links_df = DataFrame()
        links_df[current_feature] = np.linspace(feature_ranges[current_feature]['min'],
                                                feature_ranges[current_feature]['max'],
                                                num=num_links)
        # for every other feature, set it to its default value for the all links
        for other_feature in feature_ranges:
            if other_feature != current_feature:
                links_df[other_feature] = [(feature_ranges[other_feature]['default'])] * len(links_df)
        # set distance to be a constant and label it with the distance name found in the metadata
        links_df[distance_units] = [.1] * len(links_df)

        # make a prediction using the test links
        prediction = model.predict(links_df)

        # plot the prediction and save the figure
        plt.plot(links_df[current_feature],
                 prediction * 100 / links_df[distance_units],
                 label=model_name)
        plt.xlabel(f'{current_feature} [{current_units}]')
        plt.ylabel(f'{energy_units}/100{distance_units}')
        plt.savefig(output_filepath.joinpath(f'{model_name}_{current_feature}.png'),
                    format='png')
        plt.clf()

        predictions[current_feature] = prediction

    return predictions
