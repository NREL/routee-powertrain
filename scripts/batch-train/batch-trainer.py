from __future__ import annotations

import argparse
import glob
import logging
import sqlite3
import traceback
from collections import Counter
from datetime import datetime
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple, Dict, List, Union, Type, Tuple

import pandas as pd
import yaml

from powertrain import Model
from powertrain.core.features import Feature, PredictType, FeaturePack
from powertrain.estimators.explicit_bin import ExplicitBin
from powertrain.estimators.linear_regression import LinearRegression
from powertrain.estimators.random_forest import RandomForest

formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
file_handler = logging.FileHandler("batch-trainer.log")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

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


def get_estimator_class(s: str):
    registered_estimators = {
        'explicit_bin': ExplicitBin,
        'random_forest': RandomForest,
        'linear_regression': LinearRegression,
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
    features: Tuple[Feature, ...]

    estimators: List[Type[Union[ExplicitBin, LinearRegression, RandomForest], ...]]

    n_cores: int
    model_output_type: OutputType
    prediction_type: PredictType

    @classmethod
    def from_dict(cls, d: dict) -> BatchConfig:
        return BatchConfig(
            training_data_path=Path(d['training_data_path']),
            output_path=Path(d['output_path']) / datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            energy_targets={EnergyType.from_string(d['energy_type']): Feature.from_dict(d) for d in
                            d['energy_targets']},
            distance=Feature.from_dict(d['distance']),
            features=tuple(Feature.from_dict(d) for d in d['features']),
            estimators=[get_estimator_class(s) for s in d['estimators']],
            n_cores=int(d['n_cores']),
            model_output_type=OutputType.from_string(d['model_output_type']),
            prediction_type=PredictType.from_string(d['prediction_type'])
        )


class ModelConfig(NamedTuple):
    """
    config for a single model
    """
    batch_config: BatchConfig
    training_file: Path


def _err(msg: str) -> int:
    log.error(msg)
    return -1


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


def train_model(mconfig: ModelConfig) -> int:
    if not mconfig.training_file.is_file():
        _err(f"could not find training data at {mconfig.training_file}")

    bconfig = mconfig.batch_config

    # TODO: we assume the filename represents the vehicle name;
    #  perhaps better to come from a metadata table in the training database?
    model_name = mconfig.training_file.stem

    log.info(f"working on training for {model_name}")

    sql_con = sqlite3.connect(mconfig.training_file)

    log.info("reading training data into memory")
    df = pd.read_sql_query("SELECT * FROM links", sql_con)

    # TODO: we should let energy type come from a metadata table in the training database
    if df.gge.sum() > 0:
        energy = bconfig.energy_targets.get(EnergyType.GASOLINE)
        if not energy:
            _err(f"could not find energy target of type gasoline for {model_name} in the config")
    elif df.esskwhoutach.sum() > 0:
        energy = bconfig.energy_targets.get(EnergyType.ELECTRIC)
        if not energy:
            _err(f"could not find energy target of type gasoline for {model_name} in the config")
    else:
        _err(f'there is no energy in the file {mconfig.training_file}')

    train_cols = [f.name for f in bconfig.features] + [bconfig.distance.name] + [energy.name]
    train_df = df[train_cols].dropna()
    feature_pack = FeaturePack(bconfig.features, bconfig.distance, energy)

    for eclass in bconfig.estimators:
        try:
            e = eclass(feature_pack=feature_pack, predict_type=bconfig.prediction_type)
        except Exception:
            _err(f"failed to load estimator type {eclass} \n {traceback.format_exc()}")

        m = Model(e, description=model_name)
        m.train(train_df)

        if bconfig.model_output_type == OutputType.JSON:
            outfile = bconfig.output_path / f"{model_name}_{eclass.__name__}.json"
            log.info(f"writing model to {outfile}")
            m.to_json(outfile)
        elif bconfig.model_output_type == OutputType.PICKLE:
            outfile = bconfig.output_path / f"{model_name}_{eclass.__name__}.pickle"
            log.info(f"writing model to {outfile}")
            m.to_pickle(outfile)
        else:
            _err(f"got unexpected output type: {bconfig.model_output_type}")

    return 1


def run() -> int:
    log.info("🏎  routee-powertrain batch training started!")
    args = parser.parse_args()

    try:
        bconfig = load_config(args.config_file)
    except FileNotFoundError:
        _err(f"could not find {args.config_file}")

    bconfig.output_path.mkdir(parents=True, exist_ok=True)

    train_files = glob.glob(str(bconfig.training_data_path / "*.db"))
    if not train_files:
        log.error(f"no training .db files found at {bconfig.training_data_path}")
        return -1

    with Pool(bconfig.n_cores) as p:
        results = p.map(train_model, [ModelConfig(batch_config=bconfig, training_file=Path(f)) for f in train_files])

    c = Counter(results)
    if c.get(-1):
        log.error(f"{c[-1]} model(s) failed to train; check the logs to see what happened")

    return 1


if __name__ == '__main__':
    run()
