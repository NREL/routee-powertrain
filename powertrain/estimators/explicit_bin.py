from __future__ import annotations

import ast
from typing import Union

import numpy as np
import pandas as pd

from powertrain.core.features import FeaturePack, PredictType
from powertrain.estimators.estimator_interface import EstimatorInterface


class ExplicitBin(EstimatorInterface):
    """Energy consumption rates matrix with same dimensions as link features.

    The energy rates models are trained and used to predict energy consumption
    on link and route objects.

    The ExplicitBin estimator allows users to specify precisely
    which features to aggregate the data on and set the bin limits to discretize
    the data in each feature (dimension).

    Example application:
        > import routee
        > from routee.estimators import ExplicitBin
        >
        > attrb_dict = {'speed_mph_float':[1,10,20,30,40,50,60,70,80],
        >               'grade_percent_float':[-5,-4,-3,-2,-1,0,1,2,3,4,5],
        >               'num_lanes_int':[0,1,2,3,4,10]}
        >
        > model_eb = routee.Model(
        >                '2016 Ford Explorer',
        >                estimator = ExplicitBin(attrb_dict),
        >                )
        >
        > model_eb.train(fc_data, # fc_data = link attributes + fuel consumption
        >               energy='gallons',
        >               distance='miles',
        >               trip_ids='trip_ids')
        >
        > model_eb.predict(route1) # returns route1 with energy appended to each link
        
    Args:
        features (list):
            List of strings representing the input features used to predict energy.
        distance (string): 
            Name of column representing the distance feature.
        energy (string):
            Name of column representing the energy column (e.g. GGE or kWh).
        
    """

    def __init__(
            self,
            feature_pack: FeaturePack,
            predict_type: Union[str, int, PredictType] = PredictType.ENERGY_RAW,
            model: pd.DataFrame = pd.DataFrame(),
    ):
        if isinstance(predict_type, str):
            ptype = PredictType.from_string(predict_type)
        elif isinstance(predict_type, int):
            ptype = PredictType.from_int(predict_type)
        elif isinstance(predict_type, PredictType):
            ptype = predict_type
        else:
            raise TypeError(f"predict_type {predict_type} of python type {type(predict_type)} not supported")

        if ptype != PredictType.ENERGY_RAW:
            raise NotImplemented(f"{self.predict_type} not supported by ExplicitBin")

        self.predict_type = ptype
        self.bin_lims: dict = {}
        self.bin_labels: dict = {}
        self.model = model
        self.feature_pack: FeaturePack = feature_pack
        self.energy_rate_target: str = feature_pack.energy.name + '_per_100' + feature_pack.distance.name

    def train(self,
              data: pd.DataFrame,
              ):
        """
        train method for the base estimator (linear regression)
        Args:
            data:

        Returns:

        """
        if self.predict_type != PredictType.ENERGY_RAW:
            raise NotImplemented(f"{self.predict_type} not supported by ExplicitBin")

        x = data[self.feature_pack.feature_list + [self.feature_pack.distance.name]].astype(float)
        y = data[self.feature_pack.energy.name].astype(float)
        df = pd.concat([x, y], axis=1, ignore_index=True, sort=False)

        df.columns = self.feature_pack.feature_list + [self.feature_pack.distance.name] + [
            self.feature_pack.energy.name]

        # Set min and max bins using 95% interval (can also try 99%)
        # _mins = x.quantile(q=0.025)
        _mins = x.quantile(q=0)
        _maxs = x.quantile(q=.975)

        # TODO: Build a grid of bin limit permutations using 5,10,15,20 bins on each feature

        # Default bin limits and labels for grade and speed
        # format: {<keyword>: ([limits], [labels])}

        bin_defaults = {
            'grade': (
                [-15, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 15],
                [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
            ),
            'speed': (
                [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100],
                [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
            )
        }

        for f_i in self.feature_pack.feature_list:
            _unique_vals = len(df[f_i].unique())

            if _unique_vals <= 10:
                df.loc[:, f_i + '_bins'] = df.loc[:, f_i]

            elif list(bin_defaults.keys())[0] in f_i:
                self.bin_lims[f_i] = bin_defaults[list(bin_defaults.keys())[0]][0]
                self.bin_labels[f_i] = bin_defaults[list(bin_defaults.keys())[0]][1]
                df.loc[:, f_i + '_bins'] = pd.cut(df[f_i], self.bin_lims[f_i], labels=self.bin_labels[f_i])

            elif list(bin_defaults.keys())[1] in f_i:
                self.bin_lims[f_i] = bin_defaults[list(bin_defaults.keys())[1]][0]
                self.bin_labels[f_i] = bin_defaults[list(bin_defaults.keys())[1]][1]
                df.loc[:, f_i + '_bins'] = pd.cut(df[f_i], self.bin_lims[f_i], labels=self.bin_labels[f_i])

            else:
                _min_i = float(_mins[f_i])
                _max_i = float(_maxs[f_i])
                self.bin_lims[f_i] = np.linspace(_min_i, _max_i, num=10)
                self.bin_labels[f_i] = None
                df.loc[:, f_i + '_bins'] = pd.cut(df[f_i], self.bin_lims[f_i])

        # TODO: Test all bin limit permutations and select the one with the least errors

        # TODO: Need special checks for cumulative vs rates inputs on target variable

        # train rates table - groupby bin columns
        _bin_cols = [i + '_bins' for i in self.feature_pack.feature_list]
        _agg_funs = {self.feature_pack.distance.name: sum, self.feature_pack.energy.name: sum}

        self.model = df.dropna(subset=_bin_cols). \
            groupby(_bin_cols).agg(_agg_funs)

        energy_rate = 100.0 * self.model[self.feature_pack.energy.name] / self.model[self.feature_pack.distance.name]
        self.model.loc[:, self.energy_rate_target] = energy_rate

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Applies the estimator to to predict consumption.

        Args:
            data (pandas.DataFrame):
                Columns that match self.features and self.distance that describe
                vehicle passes over links in the road network.

        Returns:
            target_pred (float): 
                Predicted target for every row in links_df
        """
        links_df = data.astype(float)

        # Cut and label each attribute - manual
        for f_i in self.feature_pack.feature_list:

            _unique_vals = len(links_df[f_i].unique())
            if _unique_vals <= 10:
                links_df.loc[:, f_i + '_bins'] = links_df.loc[:, f_i]

            else:
                bin_lims = self.bin_lims[f_i]
                bin_labels = self.bin_labels[f_i]
                _min = bin_lims[0] + .000001
                _max = bin_lims[-1] - .000001
                # clip any values that exceed the lower or upper bin limits
                links_df.loc[:, f_i] = links_df[f_i].clip(lower=_min, upper=_max)
                links_df.loc[:, f_i + '_bins'] = pd.cut(links_df[f_i], bin_lims, labels=bin_labels)

        # merge energy rates from grouped table to link/route df
        bin_cols = [i + '_bins' for i in self.feature_pack.feature_list]
        links_df = pd.merge(links_df, self.model[[self.energy_rate_target]],
                            how='left', left_on=bin_cols, right_index=True)

        links_df.loc[:, self.feature_pack.energy.name] = (
                links_df[self.energy_rate_target] * links_df[self.feature_pack.distance.name] / 100.0)

        # TODO: more robust method to deal with missing bin values
        _nan_count = len(links_df) - len(links_df.dropna(how='any'))
        if _nan_count > 0:
            print(f'WARNING: prediction for {_nan_count}/{len(links_df)} '
                  'records set to zero because of nan values from table lookup process')

        return links_df[self.feature_pack.energy.name].fillna(0)

    def to_json(self) -> dict:
        out_json = {
            'model': self.model.to_json(orient="index"),
            'feature_pack': self.feature_pack.to_json(),
            'predict_type': self.predict_type.name
        }

        return out_json

    @classmethod
    def from_json(cls, json: dict) -> ExplicitBin:
        predict_type = PredictType.from_string(json['predict_type'])
        feature_pack = FeaturePack.from_json(json['feature_pack'])
        model_df = pd.read_json(json['model'], orient="index")
        model_df.index = pd.MultiIndex.from_tuples(
            [ast.literal_eval(i) for i in model_df.index],
            names=("speed_bins", "grade_bins")
        )

        return ExplicitBin(feature_pack=feature_pack, predict_type=predict_type, model=model_df)

    def dump_csv(self, fileout):
        """Dump CSV file of table ONLY. No associated metadata.

        Args:
            fileout (str):
                Path and filename of dumped CSV.
                
        """

        self.model = self.model.reset_index()
        self.model.to_csv(fileout, index=False)
