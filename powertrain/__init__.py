import logging

from powertrain.io.api import read_model
from powertrain.core.model import Model
from powertrain.core.features import Feature, FeaturePack
from powertrain.io.load import load_pretrained_model, load_route, list_available_models

name = "powertrain"
__version__ = "0.4.0"

log = logging.getLogger()
log.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

