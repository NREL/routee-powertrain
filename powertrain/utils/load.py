import pandas as pd

from powertrain.core.model import Model
from powertrain.resources.default_models import default_model_dir
from powertrain.resources.sample_routes import sample_route_dir


def load_pretrained_model(name: str) -> Model:
    """
    A helper function to load a pretrained model.

    This only loads the default models that are packaged with the repo:
     - powertrain.resources.default_models

    In the future, this might be able to load models from an external source like Box


    Args:
        name: the name of the file to load

    Returns: a pre-trained routee-powertrain model

    """
    default_models = {
        'ICE': default_model_dir() / "2016_TOYOTA_Corolla_4cyl_2WD_ExplicitBin.json",
        'EV': default_model_dir() / "2016_Leaf_24_kWh_ExplicitBin.json",
        'PHEV (Charge Sustain)': default_model_dir() / "2016_CHEVROLET_Volt_Charge_Sustaining_ExplicitBin.json",
        'PHEV (Charge Deplete)': default_model_dir() / "2016_CHEVROLET_Volt_Charge_Depleting_ExplicitBin.json",
    }

    if name not in default_models:
        raise KeyError(f"cannot find default model with name: {name}; try one of {list(default_models.keys())}")

    return Model.from_json(default_models[name])


def load_route(name: str) -> pd.DataFrame:
    """
    A helper function to load sample routes

    Args:
        name: the name of the route

    Returns: a pandas DataFrame representing the route

    """
    routes = {
        "sample_route": sample_route_dir() / "sample_route.csv",
    }

    if name not in routes:
        raise KeyError(f"cannot find route with name: {name}; try one of {list(routes.keys())}")

    df = pd.read_csv(routes[name])

    return df
