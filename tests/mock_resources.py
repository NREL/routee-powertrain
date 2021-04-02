import os

import pandas as pd

from powertrain import load_pretrained_model


def mock_route() -> pd.DataFrame:
    route_path = os.path.join("routee-powertrain-test-data", "trip_11394_1_metropia_austin_v2.csv")
    route_df = pd.read_csv(route_path)
    route_df = route_df.rename(columns={
        'mean_mph': 'gpsspeed',
    })

    return route_df


def mock_model():
    return load_pretrained_model("2016_Leaf_24_kWh_ExplicitBin")
