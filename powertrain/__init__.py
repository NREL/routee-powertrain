import logging
from pathlib import Path

from .core.model import Model
from .core.features import Feature, FeaturePack, FeatureRange
from .io.api import read_model
from .io.load import list_available_models, load_pretrained_model

name = "powertrain"
__version__ = "0.6.1"

log = logging.getLogger()
log.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)


def package_root() -> Path:
    return Path(__file__).parent
