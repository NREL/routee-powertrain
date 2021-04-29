import os

import pandas as pd

from powertrain import load_pretrained_model, FeaturePack, Feature


def mock_route() -> pd.DataFrame:
    route_path = os.path.join("routee-powertrain-test-data", "trip_11394_1_metropia_austin_v2.csv")
    route_df = pd.read_csv(route_path)

    return route_df


def mock_model():
    return load_pretrained_model("2016_Leaf_24_kWh_ExplicitBin")


def mock_data_single_feature():
    data = [
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
        {'distance': 1, 'speed': 1, 'energy': 1},
    ]

    feature_pack = FeaturePack(
        features=[Feature(name="speed", units="")],
        distance=Feature(name="distance", units=""),
        energy=Feature(name="energy", units=""),
    )
    return pd.DataFrame(data), feature_pack
