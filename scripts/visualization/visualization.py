import argparse
import glob
import logging

from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Dict, List, Union, Type, Tuple

import yaml

from powertrain.validation.feature_visualization import visualize_features
from powertrain.core.model import Model

formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
file_handler = logging.FileHandler("visualization.log")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

# TODO: update args parser
parser = argparse.ArgumentParser(description="batch run for visualizing routee-powertrain models")
parser.add_argument(
    'config_file',
    help='the configuration for this run'
)


class VisualConfig(NamedTuple):
    models_path: Path

    output_path: Path

    feature_ranges: Dict[str, float]

    @classmethod
    def from_dict(cls, d: dict) -> VisualConfig:
        return VisualConfig(
            models_path=Path(d['models_path']),
            output_path=Path(d['output_path']) / datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            feature_ranges=d['feature_ranges']
        )


def load_config(config_file: str) -> VisualConfig:
    """
    Load the user config file and returns a BatchConfig object
    """
    config_file = Path(config_file)
    if not config_file.is_file():
        raise FileNotFoundError(f"couldn't find config file: {config_file}")

    with open(config_file, 'r') as stream:
        d = yaml.safe_load(stream)
        return VisualConfig.from_dict(d)


def _err(msg: str) -> int:
    log.error(msg)
    return -1


def run():
    log.info("🏎  routee-powertrain visualization started!")
    args = parser.parse_args()

    try:
        vconfig = load_config(args.config_file)
    except FileNotFoundError:
        return _err(f"could not find {args.config_file}")

    vconfig.output_path.mkdir(parents=True, exist_ok=True)

    log.info(f"looking for .json files or .pickle files in {vconfig.models_path}")
    json_model_paths = glob.glob(str(vconfig.models_path / "*.json"))
    pickle_model_paths = glob.glob(str(vconfig.models_path / "*.pickle"))
    if not json_model_paths and not pickle_model_paths:
        return _err(f"no .json files or .pickle files found at {vconfig.models_path}")

    if json_model_paths:
        for model_path in json_model_paths:
            model = Model.from_json(model_path)
            visualize_features(model, vconfig.feature_ranges, vconfig.output_path)

    if pickle_model_paths:
        for model_path in pickle_model_paths:
            model = Model.from_pickle(model_path)
            visualize_features(model, vconfig.feature_ranges, vconfig.output_path)


if __name__ == '__main__':
    run()
