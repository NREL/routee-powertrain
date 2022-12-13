from __future__ import annotations

import json
import logging
import pickle
from urllib import request
from pkg_resources import packaging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from powertrain.core.metadata import Metadata
from powertrain.core.powertrain_type import PowertrainType
from powertrain.core.real_world_adjustments import ADJUSTMENT_FACTORS
from powertrain.core.features import Feature
from powertrain.estimators.estimator_interface import EstimatorInterface
from powertrain.estimators.explicit_bin import ExplicitBin
from powertrain.estimators.linear_regression import LinearRegression
from powertrain.estimators.random_forest import RandomForest
from powertrain.utils.fs import get_version
from powertrain.validation.errors import compute_errors

_registered_estimators = {
    "LinearRegression": LinearRegression,
    "ExplicitBin": ExplicitBin,
    "RandomForest": RandomForest,
}

log = logging.getLogger(__name__)

CURRENT_VERSION = get_version()


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
        powertrain_type (str):
            Optional powertrain type (BEV, ICE, HEV) used for applying real world correction factors.


    """

    def __init__(
        self,
        estimator: EstimatorInterface,
        description: Optional[str] = None,
        powertrain_type: Optional[str] = None,
    ):
        ptype = PowertrainType.from_string(powertrain_type)

        self.metadata = Metadata(
            model_description=description,
            estimator_name=estimator.__class__.__name__,
            estimator_features=estimator.feature_pack.to_json(),
            routee_version=get_version(),
            powertrain_type=ptype,
        )
        self._estimator = estimator

    def train(
        self,
        data: DataFrame,
        trip_column: Optional[str] = None,
        random_seed: int = 123,
        **kwargs,
    ):
        """

        Args:
            data:
            trip_column:
            random_seed:

        Returns:

        """
        log.info(f"training estimator {self._estimator.__class__.__name__}")

        if trip_column:
            data_columns = self._estimator.feature_pack.all_names + [trip_column] 
        else:
            data_columns = self._estimator.feature_pack.all_names 
            

        pass_data = data[data_columns].copy(deep=True)
        pass_data["energy_rate"] = (
            data[self.feature_pack.energy.name] / data[self.feature_pack.distance.name]
        )
        pass_data = pass_data[~pass_data.isin([np.nan, np.inf, -np.inf]).any(1)]

        # splitting test data between train and validate --> 20% here
        train, test = train_test_split(
            pass_data.dropna(), test_size=0.2, random_state=random_seed
        )

        self._estimator.train(train, **kwargs)

        model_errors = compute_errors(test, self, trip_column)

        self.metadata = self.metadata.set_errors(model_errors)

    def predict(self, links_df: DataFrame, apply_real_world_adjustment: bool = False):
        """
        Apply the trained energy model to to predict consumption.

        Args:
            links_df (pandas.DataFrame):
                Columns that match self.features and self.distance that describe
                vehicle passes over links in the road network.
            apply_real_world_adjustment (bool):
                If true, applies a real world adjustment factor to correct for environmental variables.
                Useful if the model was trained using FASTSim data.

        Returns:
            energy_pred (pandas.Series):
                Predicted energy consumption for every row in links_df.

        """
        distance_col = self._estimator.feature_pack.distance.name

        energy_pred_rates = self._estimator.predict(links_df)

        if apply_real_world_adjustment:
            adjustment_factor = ADJUSTMENT_FACTORS[self.metadata.powertrain_type]
            energy_pred_rates = energy_pred_rates * adjustment_factor

        energy_pred = energy_pred_rates * links_df[distance_col]

        return energy_pred

    def to_json(self, outfile: Path):
        """
        Dumps a powertrain model to a json file for persistence and sharing.

        Args:
            outfile (str):
                Filepath for location of dumped model.

        """
        out_dict = {
            "metadata": self.metadata.to_json(),
            "_estimator_json": self._estimator.to_json(),
        }
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=4)

    def to_pickle(self, outfile: Path):
        """
        Dumps a powertrain model to a pickle file for persistence and sharing.

        Args:
            outfile (str):
                Filepath for location of dumped model.

        """
        out_dict = {
            "metadata": self.metadata,
            "_estimator": self._estimator,
        }
        with open(outfile, "wb") as f:
            pickle.dump(out_dict, f)

    def to_lookup_table(self, feature_bins: Dict[str, int]) -> DataFrame:
        """
        Returns a lookup table for the model.

        Args:
            feature_bins (dict): A dictionary of features and their number of bins.


        Returns:
            lookup_table (pandas.DataFrame):
                A lookup table for the model.

        """
        if any([f.feature_range is None for f in self.feature_pack.features]):
            raise ValueError("Feature ranges must be set to generate lookup table")
        elif set(feature_bins.keys()) != set(self.feature_pack.feature_list):
            raise ValueError("Feature names must match model feature pack")

        # build a grid mesh over the feature ranges
        points = tuple(
            np.linspace(
                f.feature_range.lower,
                f.feature_range.upper,
                feature_bins[f.name],
            ) for f in self.feature_pack.features 
        )

        mesh = np.meshgrid(*points)
    
        pred_matrix = np.stack(list(map(np.ravel, mesh)), axis=1)

        pred_df = DataFrame(pred_matrix, columns=self.feature_pack.feature_list)
        pred_df[self.feature_pack.distance.name] = 1

        predictions = self.predict(pred_df)

        lookup = pred_df.drop(columns=[self.feature_pack.distance.name])
        energy_key = f"{self.feature_pack.energy.units}_per_{self.feature_pack.distance.units}"
        lookup[energy_key] = predictions

        return lookup

    @property
    def feature_pack(self):
        return self._estimator.feature_pack

    @classmethod
    def from_json(cls, infile: Union[Path, str]) -> Model:
        infile = Path(infile)
        with infile.open("r", encoding="utf-8") as f:
            in_json = json.load(f)
            metadata = Metadata.from_json(in_json["metadata"])

            if packaging.version.parse(
                metadata.routee_version
            ) < packaging.version.parse(CURRENT_VERSION):
                raise Exception(
                    f"This model was trained with routee version {metadata.routee_version} "
                    f"and is incompatible with current version {CURRENT_VERSION}"
                )

            estimator = _load_estimator(
                metadata.estimator_name, json=in_json["_estimator_json"]
            )

            m = Model(estimator=estimator)
            m.metadata = metadata

            return m

    @classmethod
    def from_pickle(cls, infile: Path) -> Model:
        with infile.open("rb") as f:
            in_dict = pickle.load(f)
            metadata = in_dict["metadata"]
            estimator = in_dict["_estimator"]

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
            in_json = json.loads(u.read().decode("utf-8"))
            metadata = Metadata.from_json(in_json["metadata"])
            estimator = _load_estimator(
                metadata.estimator_name, json=in_json["_estimator_json"]
            )

            m = Model(estimator=estimator)
            m.metadata = metadata

            return m
