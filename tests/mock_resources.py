import os

import pandas as pd

from powertrain import read_model


def mock_route() -> pd.DataFrame:
    route_path = os.path.join("routee-powertrain-test-data", "trip_11394_1_metropia_austin_v2.csv")
    route_df = pd.read_csv(route_path)
    route_df = route_df.rename(columns={
        'mean_mph': 'gpsspeed',
    })

    return route_df


def mock_model(
        estimator: str,
):
    if estimator == "ExplicitBin":
        model_path = os.path.join(
            "..",
            "trained_models",
            "standard", 
            "2016_Leaf_24_kWh_ExplicitBin.json")
    elif estimator == "Linear":
        model_path = os.path.join(
            "..",
            "trained_models",
            "standard", 
            "2016_Leaf_24_kWh_LinearRegression.json")
    elif estimator == "RandomForest":
        model_path = os.path.join(
            "..",
            "trained_models",
            "standard", 
            "2016_Leaf_24_kWh_RandomForest.json",
        )
    else:
        raise Exception("Incorrect estimator type")

    return read_model(model_path)
