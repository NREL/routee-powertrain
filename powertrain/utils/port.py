import re

from powertrain.estimators.random_forest import RandomForest
from powertrain.utils.jinja import jinja


def parse_port_name(name: str) -> str:
    parsed_name = re.sub(r"[^a-zA-Z0-9 \n\.]", "_", name)
    return parsed_name


def c_header_from_random_forest(random_forest: RandomForest, name: str) -> str:
    return f"double predict_{name}(double distance_miles, double *x);"


def c_source_from_random_forest(random_forest: RandomForest, name: str) -> str:
    return jinja(
        "random_forest.jinja",
        {
            "name": name,
            "n_estimators": random_forest.model.n_estimators,
            "trees": [
                {
                    "left": est.tree_.children_left,
                    "right": est.tree_.children_right,
                    "features": est.tree_.feature,
                    "thresholds": est.tree_.threshold,
                    "values": est.tree_.value,
                }
                for est in random_forest.model.estimators_
            ],
        },
    )
