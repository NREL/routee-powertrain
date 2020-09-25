import pandas as pd
import numpy as np
import copy

from routee.estimators.base import BaseEstimator


class ExplicitBin(BaseEstimator):
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
    

    def __init__(self, features, distance, energy):
        self.metadata = {}
        self.metadata['features'] = [feat.name for feat in features]
        self.metadata['distance'] = distance.name
        self.metadata['energy'] = energy.name

    def train(self, x, y):
        """Method to train the estimator over a specific dataset. 
        Overrides BaseEstimator train method.

        Args:
            train (pandas.DataFrame):
                Data frame of the training data including features and target.
            test (pandas.DataFrame):
                Data frame of testing data for validation and error calculation.
            metadata (dict):
                Dictionary of metadata including features, target, distance column,
                energy column, and trip_ids column.

        Returns:
            errors (dict):
                Dictionary with error metrics.
                
        """

        # Combine x and y
        x = x.astype(float)

        df = pd.concat([x, y], axis=1, ignore_index=True, sort=False)
        df.columns = self.metadata['features'] + [self.metadata['distance']] + [self.metadata['energy']]

        # df = df.astype(float)

        # Set min and max bins using 95% interval (can also try 99%)
        # _mins = x.quantile(q=0.025)
        _mins = x.quantile(q=0)
        _maxs = x.quantile(q=.975)

        # TODO: Build a grid of bin limit permutations using 5,10,15,20 bins on each feature
        
        # Default bin limits and labels for grade and speed
        # format: {<keyword>: ([limits], [labels])}
        
        bin_defaults = {
            'grade':([-15,-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5,5.5,15], [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]),
            'speed':([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,100], [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80])
                        }

        self.bin_lims = {}
        self.bin_labels = {}

        # just one for testing
        for f_i in self.metadata['features']:
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
        _bin_cols = [i + '_bins' for i in self.metadata['features']]
        _agg_funs = {self.metadata['distance']: sum, self.metadata['energy']: sum}

        self.model = df.dropna(subset=_bin_cols). \
            groupby(_bin_cols).agg(_agg_funs)

        # rate is dependent on the energy and distance units provided (*100)
        self.metadata['target'] = self.metadata['energy'] + '_per_100' + self.metadata['distance']

        self.model.loc[:, self.metadata['target']] = 100.0 * self.model[self.metadata['energy']] / \
                                                     self.model[self.metadata['distance']]

    def predict(self, links_df):
        """Applies the estimator to to predict consumption.

        Args:
            links_df (pandas.DataFrame):
                Columns that match self.features and self.distance that describe
                vehicle passes over links in the road network.

        Returns:
            target_pred (float): 
                Predicted target for every row in links_df
        """
        links_df = links_df.astype(float)

        # Cut and label each attribute - manual
        for f_i in self.metadata['features']:

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
        bin_cols = [i + '_bins' for i in self.metadata['features']]
        links_df = pd.merge(links_df, self.model[[self.metadata['target']]], \
                            how='left', left_on=bin_cols, right_index=True)

        links_df.loc[:, self.metadata['energy']] = (
                    links_df[self.metadata['target']] * links_df[self.metadata['distance']] / 100.0)

        # print(links_df[links_df.isna().any(axis=1)])

        # links_df.dropna(how='any', inplace=True)
        # links_df.fillna(method='bfill', axis=1, inplace=True)

        # TODO: more robust method to deal with missing bin values
        _nan_count = len(links_df) - len(links_df.dropna(how='any'))
        if _nan_count>0:
            print('    WARNING: prediction for %d/%d records set to zero because of nan values from table lookup process' % (_nan_count, len(links_df)))

        return np.array(links_df[self.metadata['energy']].fillna(0))

    def dump_csv(self, fileout):
        """Dump CSV file of table ONLY. No associated metadata.

        Args:
            fileout (str):
                Path and filename of dumped CSV.
                
        """

        self.model = self.model.reset_index()
        self.model.to_csv(fileout, index=False)
