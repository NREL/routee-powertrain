from __future__ import annotations

from dataclasses import dataclass
import json
from math import isinf
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING, Union
from urllib import request

import pandas as pd
from nrel.routee.powertrain.core.features import (
    FeatureSetId,
    feature_id_to_names,
    feature_names_to_id,
)

from nrel.routee.powertrain.core.metadata import Metadata
from nrel.routee.powertrain.core.real_world_adjustments import ADJUSTMENT_FACTORS
from nrel.routee.powertrain.estimators.estimator_interface import Estimator
from nrel.routee.powertrain.estimators.onnx import ONNXEstimator
from nrel.routee.powertrain.estimators.smart_core import SmartCoreEstimator
from nrel.routee.powertrain.estimators.ngboost_estimator import NGBoostEstimator

from nrel.routee.powertrain.io.to_lookup_table import to_lookup_table
from nrel.routee.powertrain.validation.feature_visualization import (
    contour_plot,
    visualize_features,
)
from nrel.routee.powertrain.validation.errors import ModelErrors

if TYPE_CHECKING:
    from pandas import Series

REGISTERED_ESTIMATORS = {
    "ONNXEstimator": ONNXEstimator,
    "SmartCoreEstimator": SmartCoreEstimator,
    "NGBoostEstimator": NGBoostEstimator,
}

METADATA_SERIALIZATION_KEY = "metadata"
MODEL_ERRORS_SERIALIZATION_KEY = "errors"
ALL_ESTIMATOR_SERIALIZATION_KEY = "all_estimators"
ESTIMATOR_SERIALIZATION_KEY = "estimator"
CONSTRUCTOR_TYPE_SERIALIZATION_KEY = "estimator_constructor_type"


@dataclass
class Model:
    """
    A RouteE-Powertrain vehicle model represents a single vehicle
    (i.e. a 2016 Toyota Camry with a 1.5 L gasoline engine).
    """

    estimators: Dict[FeatureSetId, Estimator]
    metadata: Metadata
    errors: ModelErrors

    @property
    def feature_sets(self):
        return self.metadata.config.feature_sets

    @property
    def feature_set_lists(self) -> List[List[str]]:
        return [feature_id_to_names(fid) for fid in self.estimators.keys()]

    @classmethod
    def from_dict(cls, input_dict: dict) -> Model:
        """
        Load a vehicle model from a python dictionary
        """
        metadata_dict = input_dict.get(METADATA_SERIALIZATION_KEY)
        if metadata_dict is None:
            raise ValueError(
                "Model file must contain metadata at key: "
                f"'{METADATA_SERIALIZATION_KEY}'"
            )
        metadata = Metadata.from_dict(metadata_dict)

        model_errors_dict = input_dict.get(MODEL_ERRORS_SERIALIZATION_KEY)
        if model_errors_dict is None:
            raise ValueError(
                "Model file must contain model errors at key: "
                f"'{MODEL_ERRORS_SERIALIZATION_KEY}'"
            )
        model_errors = ModelErrors.from_dict(model_errors_dict)

        all_estimators_dict = input_dict.get(ALL_ESTIMATOR_SERIALIZATION_KEY)
        if all_estimators_dict is None:
            raise ValueError(
                "Model file must contain estimator data at key: "
                f"'{ALL_ESTIMATOR_SERIALIZATION_KEY}'"
            )

        estimator_constructor_type = input_dict.get("estimator_constructor_type")

        estimators = {}
        for feature_set_id, ed in all_estimators_dict.items():
            constructor_type = ed.get(CONSTRUCTOR_TYPE_SERIALIZATION_KEY)
            if estimator_constructor_type is None:
                raise ValueError(
                    "Model file must contain estimator constructor at key: "
                    f"'{CONSTRUCTOR_TYPE_SERIALIZATION_KEY}'"
                )

            estimator_constructor = REGISTERED_ESTIMATORS.get(constructor_type)
            if estimator_constructor is None:
                raise ValueError(
                    f"Estimator constructor type '{estimator_constructor_type}' "
                    "is not registered"
                )

            estimator_input_dict = ed.get(ESTIMATOR_SERIALIZATION_KEY)
            if estimator_input_dict is None:
                raise ValueError(
                    "Model file must contain estimator data at key: "
                    f"'{ESTIMATOR_SERIALIZATION_KEY}'"
                )

            estimator = estimator_constructor.from_dict(estimator_input_dict)
            estimators[feature_set_id] = estimator

        return cls(estimators, metadata, model_errors)

    def to_dict(self) -> dict:
        """
        Convert model to a dictionary
        """
        estimator_dict = {}
        for feature_set_id, estimator in self.estimators.items():
            estimator_dict[feature_set_id] = {
                ESTIMATOR_SERIALIZATION_KEY: estimator.to_dict(),
                CONSTRUCTOR_TYPE_SERIALIZATION_KEY: estimator.__class__.__name__,
            }

        return {
            METADATA_SERIALIZATION_KEY: self.metadata.to_dict(),
            MODEL_ERRORS_SERIALIZATION_KEY: self.errors.to_dict(),
            ALL_ESTIMATOR_SERIALIZATION_KEY: estimator_dict,
            CONSTRUCTOR_TYPE_SERIALIZATION_KEY: self.estimators.__class__.__name__,
        }

    @classmethod
    def from_file(cls, file: Union[str, Path]):
        """
        Load a vehicle model from a file.

        Args:
            file: the path to the file to load

        Returns: a powertrain vehicle
        """
        path = Path(file)
        if path.suffix != ".json":
            raise ValueError("Model file must be a .json file")
        with path.open("r") as f:
            input_dict = json.load(f)
        return cls.from_dict(input_dict)

    @classmethod
    def from_url(cls, url: str) -> Model:
        """
        Attempts to read a file from a url.

        Args:
            url: the url to download the file from

        Returns: a powertrain vehicle
        """
        with request.urlopen(url) as u:
            in_dict = json.load(u)
            vehicle = cls.from_dict(in_dict)

        return vehicle

    def to_file(self, file: Union[str, Path]):
        """
        Save a vehicle model to a file.

        Args:
            file: the path to the file to save to
        """
        path = Path(file)
        if path.suffix != ".json":
            raise ValueError("Model file must be a .json file")

        output_dict = self.to_dict()
        with path.open("w") as f:
            json.dump(output_dict, f)

    def to_lookup_table(
        self,
        feature_parameters: list[dict],
        energy_target: str,
    ) -> pd.DataFrame:
        """
        Convert the the model to a lookup table for the given estimator id and feature parameters.
        """
        return to_lookup_table(self, feature_parameters, energy_target)

    def visualize_features(
        self,
        estimator_id: FeatureSetId,
        n_samples: Optional[int] = 100,
        output_path: Optional[str] = None,
        return_predictions: Optional[bool] = False,
    ) -> Optional[Dict[str, "Series"]]:
        """
        generates test links to independently test the model's features
        and creates plots of those predictions for the given estimator id

        Args:
            estimator_id: the estimator id for generating the plots
            n_samples: the number of samples used to generate the plots
            output_path: an optional path to save the plots as png files.
            return_predictions: if true, returns the dictionary containing the prediction values

        Returns: optionally returns a dictionary containing the predictions where the key is the feature tested
        """
        feature_set = self.metadata.config.get_feature_set(
            feature_id_to_names(estimator_id)
        )
        if feature_set is None:
            raise KeyError(
                f"Model does not have a feature set with the features: {feature_id_to_names(estimator_id)}"
            )
        feature_ranges = {}
        for f in feature_set.features:
            if isinf(f.constraints.upper) or isinf(f.constraints.lower):
                raise ValueError(
                    f"Feature: {f.name} has constraints with positive/negative infinity in the lower/upper bound. "
                    f"You can add constraints when training a model or set custom constraints during visualization using "
                    f"nrel.routee.powertrain.validation.feature_visualization.visualize_features"
                )
            feature_ranges[f.name] = {
                "upper": f.constraints.upper,
                "lower": f.constraints.lower,
                "n_samples": n_samples,
            }

        return visualize_features(
            model=self,
            feature_ranges=feature_ranges,
            output_path=output_path,
            return_predictions=return_predictions,
        )

    def contour(
        self,
        estimator_id: FeatureSetId,
        x_feature: str,
        y_feature: str,
        n_samples: Optional[int] = 100,
        output_path: Optional[str] = None,
    ):
        """
        generates a contour plot of the two test features: x_feature and y_feature.
        for the given estimator id

        Args:
            estimator_id: the estimator id for generating the plots
            x_feature: one of the features used to generate the energy matrix
                and will be the x-axis feature
            y_feature: one of the features used to generate the energy matrix
                and will be the y-axis feature
            n_samples: the number of samples used to generate the plots
            output_path: an optional path to save the plots as png files.
        """
        feature_set = self.metadata.config.get_feature_set(
            feature_id_to_names(estimator_id)
        )
        if feature_set is None:
            raise KeyError(
                f"Model does not have a feature set with the features: {feature_id_to_names(estimator_id)}"
            )
        feature_ranges = {}
        for f in feature_set.features:
            if isinf(f.constraints.upper) or isinf(f.constraints.lower):
                raise ValueError(
                    f"Feature: {f.name} has constraints with positive/negative infinity in the lower/upper bound. "
                    f"You can add constraints when training a model or set custom constraints during visualization using "
                    f"nrel.routee.powertrain.validation.feature_visualization.contour_plot"
                )
            feature_ranges[f.name] = {
                "upper": f.constraints.upper,
                "lower": f.constraints.lower,
                "n_samples": n_samples,
            }

        return contour_plot(
            model=self,
            x_feature=x_feature,
            y_feature=y_feature,
            feature_ranges=feature_ranges,
            output_path=output_path,
        )

    def predict(
        self,
        links_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        distance_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predict absolute energy consumption for each link

        Args:
            links_df: a dataframe containing the links to predict on
            feature_columns: the features to use for prediction
            distance_column: the column to use for distance

        Returns: a dataframe containing the predicted energy consumption for each link
        """
        config = self.metadata.config

        if distance_column is None:
            distance_column = config.distance.name
            if distance_column not in links_df.columns:
                raise ValueError(
                    f"links_df must contain a distance column named: '{distance_column}'"
                )
        else:
            links_df = links_df.rename(columns={distance_column: config.distance.name})

        # if we only have one estimator, just use that
        if len(self.estimators) == 1:
            feature_set_id = list(self.estimators.keys())[0]
            estimator = self.estimators.get(feature_set_id)
            if estimator is None:
                raise ValueError("Could not find estimator")

        # if no explicit feature names are supplied we assume that the
        # dataframe contains all the features needed for prediction;
        # if that isn't the case, we throw an error
        elif feature_columns is None:
            feature_columns = [c for c in links_df.columns if c != distance_column]
            feature_set_id = feature_names_to_id(feature_columns)
            estimator = self.estimators.get(feature_set_id)
            if estimator is None:
                raise ValueError(
                    "This model has multiple feature sets and no features were "
                    "explicitly provided. "
                    "We attempted to just use the columns in the incoming dataframe "
                    "but we couldn't find an estiamtor that matches the features: "
                    f"{feature_columns}. "
                    "Please provide an explicit list of feature names to the features "
                    "paramter of the predict method or provide a dataframe that only "
                    "contains the features you want to use. "
                    "Here are the feature sets that can be used: "
                    f"{self.feature_set_lists}"
                )
        else:
            feature_set_id = feature_names_to_id(feature_columns)
            estimator = self.estimators.get(feature_set_id)
            if estimator is None:
                raise ValueError(
                    "Could not find an estimator that matches the provided "
                    f"feature columns {feature_columns}. Here are the feature "
                    f"sets that can be used: {self.feature_set_lists}"
                )

        feature_set = self.metadata.config.feature_set_map.get(feature_set_id)
        if feature_set is None:
            raise ValueError(
                f"Could not find a feature set {feature_set_id} in model config"
            )

        pred_energy_df = estimator.predict(
            links_df,
            feature_set,
            self.metadata.config.distance,
            self.metadata.config.target,
            self.metadata.config.predict_method,
        )

        for energy in config.target.targets:
            if config.apply_real_world_adjustment:
                adjustment_factor = ADJUSTMENT_FACTORS.get(config.powertrain_type)
                if adjustment_factor is None:
                    raise ValueError(
                        f"Could not find an adjustment factor for powertrain type "
                        f"{config.powertrain_type}"
                    )
                pred_energy_df[energy.name] = (
                    pred_energy_df[energy.name] * adjustment_factor
                )

        return pred_energy_df

    def __repr__(self) -> str:
        """
        Shows a nice pretty printed summary of the model including:
         - Model average fuel consumption
         - Select set of errors
         - Expected features and their units
         - Powertrain specifications
        """
        config = self.metadata.config
        summary_lines = []
        summary_lines.append("=" * 40)
        summary_lines.append("Model Summary")
        summary_lines.append("-" * 20)
        summary_lines.append(f"Vehicle description: {config.vehicle_description}")
        summary_lines.append(f"Powertrain type: {config.powertrain_type.name}")
        summary_lines.append(f"Number of estimators: {len(self.estimators)}")
        summary_lines.append("=" * 40)
        for feature_set_id in self.estimators.keys():
            estimator_errors = self.errors.estimator_errors.get(feature_set_id)
            if estimator_errors is None:
                raise ValueError(
                    f"Could not find errors for estimator {feature_set_id}"
                )
            summary_lines.append("Estimator Summary")
            summary_lines.append("-" * 20)
            feature_set = config.feature_set_map.get(feature_set_id)
            if feature_set is None:
                raise ValueError(
                    f"Could not find a feature set {feature_set_id} in model config"
                )
            for feature in feature_set.features:
                summary_lines.append(f"Feature: {feature.name} ({feature.units})")
            summary_lines.append(
                f"Distance: {config.distance.name} ({config.distance.units})"
            )
            for target in config.target.targets:
                summary_lines.append(f"Target: {target.name} ({target.units})")
                target_errors = estimator_errors.error_by_target.get(target.name)
                if target_errors is None:
                    raise ValueError(f"Could not find errors for target {target.name}")

                summary_lines.append(
                    f"Raw Predicted Consumption: {target_errors.pred_dist_per_energy:.3f} "
                    f"({config.distance.units}/{target.units})"
                )
                summary_lines.append(
                    f"Real World Predicted Consumption: {target_errors.real_world_pred_dist_per_energy:.3f} "
                    f"({config.distance.units}/{target.units})"
                )
            summary_lines.append(
                f"Predict Method: {config.predict_method.value.upper()}"
            )
            summary_lines.append("=" * 40)
        return "\n".join(summary_lines)

    def _repr_html_(self) -> str:
        """
        Returns an html table of the model summary for display in a notebook
        """
        config = self.metadata.config

        # Start the HTML table
        html_lines = ['<table border="1" style="border-collapse: collapse;">']

        # Title: Model Summary
        html_lines.append(
            '<tr><th colspan="2" style="border-bottom: 2px solid black; text-align: center;">Model Summary</th></tr>'
        )
        html_lines.append(
            f"<tr><td>Vehicle description</td><td>{config.vehicle_description}</td></tr>"
        )
        html_lines.append(
            f"<tr><td>Powertrain type</td><td>{config.powertrain_type.name}</td></tr>"
        )
        html_lines.append(
            f"<tr><td>Number of estimators</td><td>{len(self.estimators)}</td></tr>"
        )

        for feature_set_id in self.estimators.keys():
            estimator_errors = self.errors.estimator_errors.get(feature_set_id)
            if estimator_errors is None:
                raise ValueError(
                    f"Could not find errors for estimator {feature_set_id}"
                )

            # Title: Estimator Summary
            html_lines.append(
                '<tr><th colspan="2" style="border-bottom: 2px solid black; text-align: center;">Estimator Summary</th></tr>'
            )

            feature_set = config.feature_set_map.get(feature_set_id)
            if feature_set is None:
                raise ValueError(
                    f"Could not find a feature set {feature_set_id} in model config"
                )

            for feature in feature_set.features:
                html_lines.append(
                    f"<tr><td>Feature</td><td>{feature.name} ({feature.units})</td></tr>"
                )

            html_lines.append(
                "<tr><td>Distance</td>"
                f"<td>{config.distance.name} ({config.distance.units})</td></tr>"
            )

            for target in config.target.targets:
                html_lines.append(
                    f"<tr><td>Target</td><td>{target.name} ({target.units})</td></tr>"
                )

                target_errors = estimator_errors.error_by_target.get(target.name)
                if target_errors is None:
                    raise ValueError(f"Could not find errors for target {target.name}")

                html_lines.append(
                    "<tr><td>Predicted Consumption</td>"
                    f"<td>{target_errors.pred_dist_per_energy:.3f} "
                    f"({config.distance.units}/{target.units})</td></tr>"
                )

                html_lines.append(
                    "<tr><td>Real World Predicted Consumption</td>"
                    f"<td>{target_errors.real_world_pred_dist_per_energy:.3f} "
                    f"({config.distance.units}/{target.units})</td></tr>"
                )
            html_lines.append(
                f"<tr><td>Predict Method</td>"
                f"<td>{config.predict_method.value.upper()}</td></tr>"
            )

        # End the HTML table
        html_lines.append("</table>")

        return "".join(html_lines)
