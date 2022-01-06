import json
from typing import List

import pandas as pd

from powertrain.core.model import Model
from powertrain.resources.default_models import default_model_dir
from powertrain.resources.sample_routes import sample_route_dir

local_models = {
    "ICE": default_model_dir() / "2016_TOYOTA_Camry_4cyl_2WD-speed&grade.json",
    "EV": default_model_dir() / "2016_TESLA_Model_S60_2WD-speed&grade.json",
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

    This only loads the default models that are packaged with the repo:
     - powertrain.resources.default_models

    In the future, this might be able to load models from an external source like Box


    Args:
        name: the name of the file to load

    Returns: a pre-trained routee-powertrain model

    """

    with open(default_model_dir() / "external_model_links.json", "r") as jf:
        external_models = json.load(jf)

    if name in local_models:
        fp = local_models[name]
        model = Model.from_json(fp)
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
