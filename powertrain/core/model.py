from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Union
from urllib import request

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from powertrain.core.metadata import Metadata
from powertrain.estimators.estimator_interface import EstimatorInterface
from powertrain.estimators.explicit_bin import ExplicitBin
from powertrain.estimators.linear_regression import LinearRegression
from powertrain.estimators.random_forest import RandomForest
from powertrain.utils.fs import get_version
from powertrain.validation.errors import compute_errors

_registered_estimators = {
    'LinearRegression': LinearRegression,
    'ExplicitBin': ExplicitBin,
    'RandomForest': RandomForest,
}

log = logging.getLogger(__name__)


def _load_estimator(name: str, json: dict) -> EstimatorInterface:
    if name not in _registered_estimators:
        raise TypeError(f"{name} estimator not registered with routee-powertrain")

    e = _registered_estimators[name]

    return e.from_json(json)


class Model:
    """This is the core model for interaction with the routee engine.

    Args:
        description (str):
            Unique description of the vehicle to be modeled.
        estimator (routee.estimator.base.BaseEstimator):
            Estimator to use for predicting route energy usage.
            
    """

    def __init__(self, estimator: EstimatorInterface, description: Optional[str] = None):
        self.metadata = Metadata(
            model_description=description,
            estimator_name=estimator.__class__.__name__,
            estimator_features=estimator.feature_pack.to_json(),
            routee_version=get_version()
        )
        self._estimator = estimator

    def train(
            self,
            data: DataFrame,
            trip_column: Optional[str] = None,
            random_seed: int = 123,
    ):
        """

        Args:
            data:
            trip_column:
            random_seed:

        Returns:

        """
        log.info(f"training estimator {self._estimator.__class__.__name__}")

        pass_data = data.copy(deep=True)
        pass_data["energy_rate"] = data[self.feature_pack.energy.name] / data[self.feature_pack.distance.name]
        pass_data = pass_data[~pass_data.isin([np.nan, np.inf, -np.inf]).any(1)]

        # splitting test data between train and validate --> 20% here
        train, test = train_test_split(pass_data.dropna(), test_size=0.2, random_state=random_seed)

        self._estimator.train(pass_data)

        model_errors = compute_errors(test, self, trip_column)

        self.metadata = self.metadata.set_errors(model_errors)

    def predict(self, links_df: DataFrame):
        """Apply the trained energy model to to predict consumption.

        Args:
            links_df (pandas.DataFrame):
                Columns that match self.features and self.distance that describe
                vehicle passes over links in the road network.

        Returns:
            energy_pred (pandas.Series):
                Predicted energy consumption for every row in links_df.
                
        """
        return self._estimator.predict(links_df)

    def to_json(self, outfile: Path):
        """Dumps a powertrain model to a json file for persistence and sharing.

        Args:
            outfile (str):
                Filepath for location of dumped model.

        """
        out_dict = {
            'metadata': self.metadata.to_json(),
            '_estimator_json': self._estimator.to_json(),
        }
        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=4)

    def to_pickle(self, outfile: Path):
        """Dumps a powertrain model to a pickle file for persistence and sharing.

        Args:
            outfile (str):
                Filepath for location of dumped model.

        """
        out_dict = {
            'metadata': self.metadata,
            '_estimator': self._estimator,
        }
        with open(outfile, 'wb') as f:
            pickle.dump(out_dict, f)

    @property
    def feature_pack(self):
        return self._estimator.feature_pack

    @classmethod
    def from_json(cls, infile: Union[Path, str]) -> Model:
        infile = Path(infile)
        with infile.open('r', encoding='utf-8') as f:
            in_json = json.load(f)
            metadata = Metadata.from_json(in_json['metadata'])
            estimator = _load_estimator(metadata.estimator_name, json=in_json['_estimator_json'])

            m = Model(estimator=estimator)
            m.metadata = metadata

            return m

    @classmethod
    def from_pickle(cls, infile: Path) -> Model:
        with infile.open('rb') as f:
            in_dict = pickle.load(f)
            metadata = in_dict['metadata']
            estimator = in_dict['_estimator']

            m = Model(estimator=estimator)
            m.metadata = metadata

            return m

    @classmethod
    def from_url(cls, url: str, filetype="json") -> Model:
        """
        attempts to read a file from a url
        Args:
            url: the url to download the file from
            filetype: the type of file to expect

        Returns: a powertrain model
        """
        if filetype.lower() != "json":
            raise NotImplementedError("only json filetypes are supported")

        with request.urlopen(url) as u:
            in_json = json.loads(u.read().decode('utf-8'))
            metadata = Metadata.from_json(in_json['metadata'])
            estimator = _load_estimator(metadata.estimator_name, json=in_json['_estimator_json'])

            m = Model(estimator=estimator)
            m.metadata = metadata

            return m
