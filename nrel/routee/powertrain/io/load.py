import json
from typing import List

import pandas as pd

from nrel.routee.powertrain.core.model import Model
from nrel.routee.powertrain.resources.default_models import default_model_dir
from nrel.routee.powertrain.resources.sample_routes import sample_route_dir

local_models = {
    "2016_TOYOTA_Camry_4cyl_2WD": default_model_dir()
    / "2016_TOYOTA_Camry_4cyl_2WD.json",
    "2017_CHEVROLET_Bolt": default_model_dir() / "2017_CHEVROLET_Bolt.json",
}


def list_available_models(local: bool = True, external: bool = True) -> List[str]:
    """
    returns a list of all the available pretrained models

    Args:
        local: include local models?
        external: include external models?
        print: should we print the results too?

    Returns: a list of model keys
    """
    model_names = []
    if local:
        model_names.extend(list(local_models.keys()))

    if external:
        with open(default_model_dir() / "external_model_links.json", "r") as jf:
            external_models = json.load(jf)

            model_names.extend(list(external_models.keys()))

    return model_names


def load_pretrained_model(name: str) -> Model:
    """
    A helper function to load a pretrained model.

    Args:
        name: the name of the file to load

    Returns: a pre-trained routee-powertrain model

    """

    with open(default_model_dir() / "external_model_links.json", "r") as jf:
        external_models = json.load(jf)

    if name in local_models:
        fp = local_models[name]
        model = Model.from_file(fp)
        return model
    elif name in external_models:
        url = external_models[name]
        model = Model.from_url(url)
        return model
    else:
        all_models = list(local_models.keys()) + list(external_models.keys())
        raise KeyError(f"cannot find model with name: {name}; try one of {all_models}")


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
        raise KeyError(
            f"cannot find route with name: {name}; try one of {list(routes.keys())}"
        )

    df = pd.read_csv(routes[name])

    return df