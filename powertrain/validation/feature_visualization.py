import matplotlib.pyplot as plt
import yaml

import numpy as np
from pandas import DataFrame

from typing import Dict

from powertrain.core.model import Model

log = logging.getLogger(__name__)


def visualize_features(model: Model, feature_ranges: Dict, output_filepath: str):

    # grab the necessary metadata from the model
    feature_meta = model.metadata.estimator_features['features']
    distance_units = model.metadata.estimator_features['distance']['units']
    energy_units = model.metadata.estimator_features['energy']['units']
    model_name = model.metadata.model_description

    feature_dict = {}
    for feature in feature_meta: feature_dict[feature['name']] = feature['units']

    # check that all features in the metadata are present in the config
    # if any features are missing in config, throw an error
    if not all(feature in feature_ranges.keys() for feature in feature_dict.keys()):
        missing_features = set(feature_dict.keys()) - set(feature_ranges.keys())
        log.info(f"feature range config is missing {missing_features} for model {model_name}. Aborting visualization")
        return

    # for each feature test it individually using the values form the visualization feature ranges
    for current_feature, current_units in feature_dict.items():

        # setup a set of test links
        links_df = DataFrame()
        links_df[current_feature] = np.linspace(feature_ranges[current_feature]['min'],
                                                feature_ranges[current_feature]['max'],
                                                num=15)
        for other_feature in feature_ranges:
            if other_feature != current_feature:
                links_df[other_feature] = [(feature_ranges[other_feature]['default'])] * len(links_df)
        links_df[distance_units] = [.1] * len(links_df)

        prediction = model.predict(links_df)

        plt.plot(links_df[current_feature],
                 prediction * 100 / links_df[distance_units],
                 label=model_name)
        plt.xlabel(f'{current_feature} [{current_units}]')
        plt.ylabel(f'{energy_units}/100{distance_units}')
        plt.savefig(f'{current_feature}_vs_{energy_units}_per_100{distance_units}.png', format='png')
        plt.clf()
