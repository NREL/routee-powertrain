import os

import pandas as pd

from powertrain import load_pretrained_model, FeaturePack, Feature


def mock_route() -> pd.DataFrame:
    route_path = os.path.join("routee-powertrain-test-data", "trip_11394_1_metropia_austin_v2.csv")
    route_df = pd.read_csv(route_path)

    return route_df


def mock_model():
    return load_pretrained_model("ICE")


def mock_ev_data():
    path = os.path.join("routee-powertrain-test-data", "test_data_2016_Nissan_Leaf_30_kWh.csv")
    return pd.read_csv(path)


def mock_ice_data():
    path = os.path.join("routee-powertrain-test-data", "test_data_2016_TOYOTA_Corolla_4cyl_2WD.csv")
    return pd.read_csv(path)


def mock_data_single_feature():
    data = [
        {'miles': 1, 'gpsspeed': 1, 'energy': 1},
        {'miles': 1, 'gpsspeed': 2, 'energy': 1},
        {'miles': 1, 'gpsspeed': 3, 'energy': 1},
        {'miles': 1, 'gpsspeed': 4, 'energy': 1},
        {'miles': 1, 'gpsspeed': 5, 'energy': 1},
        {'miles': 1, 'gpsspeed': 6, 'energy': 1},
        {'miles': 1, 'gpsspeed': 7, 'energy': 1},
        {'miles': 1, 'gpsspeed': 8, 'energy': 1},
        {'miles': 1, 'gpsspeed': 9, 'energy': 1},
        {'miles': 1, 'gpsspeed': 10, 'energy': 1},
        {'miles': 1, 'gpsspeed': 12, 'energy': 1},
    ]

    feature_pack = FeaturePack(
        features=(Feature(name="gpsspeed", units=""),),
        distance=Feature(name="miles", units=""),
        energy=Feature(name="energy", units=""),
    )
    return pd.DataFrame(data), feature_pack
