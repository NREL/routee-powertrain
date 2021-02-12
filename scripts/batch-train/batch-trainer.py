from __future__ import annotations

import io
import logging
import tempfile
import argparse
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Dict, List

import pandas as pd
import yaml

from powertrain.core.features import Feature, PredictType
from powertrain.estimators.estimator_interface import EstimatorInterface
from powertrain.estimators.explicit_bin import ExplicitBin
from powertrain.estimators.linear_regression import LinearRegression
from powertrain.estimators.random_forest import RandomForest
from powertrain.estimators.xgboost import XGBoost

logging.basicConfig(filename='batch_run.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s')

logging.info('RouteE batch run START')

parser = argparse.ArgumentParser(description="batch run for training routee-powertrain models")
parser.add_argument(
    'config_file',
    help='the configuration for this run'
)


class EnergyType(Enum):
    ELECTRIC = 0
    GASOLINE = 1

    @classmethod
    def from_string(cls, string: str) -> EnergyType:
        if string.lower() == "electric":
            return cls.ELECTRIC
        elif string.lower() == "gasoline":
            return cls.GASOLINE
        else:
            raise TypeError(f"energy type {string} not supported by this script; try [electric | gasoline]")


class OutputType(Enum):
    JSON = 0
    PICKLE = 1

    @classmethod
    def from_string(cls, string: str) -> OutputType:
        if string.lower() == "json":
            return OutputType.JSON
        elif string.lower() == "pickle":
            return OutputType.PICKLE
        else:
            raise TypeError(f"output type {string} not supported by this script; try [json | pickle]")


def get_estimator_class(s: str) -> EstimatorInterface:
    registered_estimators = {
        'explicit_bin': ExplicitBin,
        'random_forest': RandomForest,
        'linear_regression': LinearRegression,
        'xgboost': XGBoost,
    }
    if s not in registered_estimators:
        raise TypeError(f"{s} is not a valid estimator type; try one of {list(registered_estimators.keys())}")

    else:
        return registered_estimators[s]


class BatchConfig(NamedTuple):
    training_data_path: Path
    output_path: Path

    energy_targets: Dict[EnergyType, Feature]
    distance: Feature
    features: List[Feature, ...]

    estimators: List[EstimatorInterface, ...]

    n_cores: int
    model_output_type: OutputType
    prediction_type: PredictType

    @classmethod
    def from_dict(cls, d: dict) -> BatchConfig:
        return BatchConfig(
            training_data_path=Path(d['training_data_path']),
            output_path=Path(d['training_data_path']),
            energy_targets={EnergyType.from_string(d['energy_type']): Feature.from_dict(d) for d in d['energy_targets']},
            distance=Feature.from_dict(d['distance']),
            features=[Feature.from_dict(d) for d in d['features']],
            estimators=[get_estimator_class(s) for s in d['estimators']],
            n_cores=int(d['n_cores']),
            model_output_type=OutputType.from_string(d['model_output_type']),
            prediction_type=PredictType.from_string(d['prediction_type'])
        )


def load_config(config_file: str) -> BatchConfig:
    """
    Load the user config file and returns a BatchConfig object
    """
    config_file = Path(config_file)
    if not config_file.is_file():
        raise FileNotFoundError(f"couldn't find config file: {config_file}")

    with open(config_file, 'r') as stream:
        d = yaml.safe_load(stream)
        return BatchConfig.from_dict(d)


def read_sql_inmem_uncompressed(query, db_engine):
    copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
        query=query, head="HEADER"
    )
    conn = db_engine.raw_connection()
    cur = conn.cursor()
    store = io.StringIO()
    cur.copy_expert(copy_sql, store)
    store.seek(0)
    df = pd.read_csv(store)
    return df


def read_sql_tmpfile(query, db_engine):
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
            query=query, head="HEADER"
        )
        conn = db_engine.raw_connection()
        cur = conn.cursor()
        cur.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
        return df


def train_routee_model(tuple_in):
    pass

def run() -> int:
    args = parser.parse_args()

    config = load_config(args.config_file)

    print(config)


if __name__ == '__main__':
    run()

