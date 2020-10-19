from typing import Union

from numpy import clip
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor

from powertrain.core.features import PredictType, FeaturePack
from powertrain.estimators.estimator_interface import EstimatorInterface


class RandomForest(EstimatorInterface):
    """This estimator uses a random forest to select an optimal decision tree,
    meant to serve as an automated construction of a lookup table.

    Example application:
        > import powertrain
        > from routee.estimators import RandomForest
        >
        >
        > model_rf = routee.Model(
        >                '2016 Ford Explorer',
        >                estimator = RandomForest(cores = 2),
        >                )
        >
        > model_rf.train(fc_data, # fc_data = link attributes + fuel consumption
        >               energy='gallons',
        >               distance='miles',
        >               trip_ids='trip_ids')
        >
        > model_rf.predict(route1) # returns route1 with energy appended to each link
        
    Args:
        cores (int):
            Number of cores to use during traing.
            
    """

    def __init__(
            self,
            feature_pack: FeaturePack,
            cores: int = 2,
            predict_type: Union[str, int, PredictType] = PredictType.ENERGY_RAW,
    ):
        if isinstance(predict_type, str):
            ptype = PredictType.from_string(predict_type)
        elif isinstance(predict_type, int):
            ptype = PredictType.from_int(predict_type)
        elif isinstance(predict_type, PredictType):
            ptype = predict_type
        else:
            raise TypeError(f"predict_type {predict_type} of python type {type(predict_type)} not supported")
        self.predict_type: PredictType = ptype

        mod = RandomForestRegressor(n_estimators=20,
                                    max_features='auto',
                                    max_depth=10,
                                    min_samples_split=10,
                                    n_jobs=cores,
                                    random_state=52)
        self.model: RandomForestRegressor = mod

        self.feature_pack: FeaturePack = feature_pack

    def train(self, data: DataFrame):
        """
        train method for the base estimator (linear regression)
        Args:
            data:

        Returns:

        """

        if self.predict_type == PredictType.ENERGY_RATE:  # convert absolute consumption to rate consumption
            energy_rate_name = self.feature_pack.energy_name + "_per_" + self.feature_pack.distance_name
            energy_rate = data[self.feature_pack.energy_name] / data[self.feature_pack.distance_name]
            data[energy_rate_name] = energy_rate

            x = data[self.feature_pack.feature_list]
            y = data[energy_rate_name]
        elif self.predict_type == PredictType.ENERGY_RAW:
            x = data[self.feature_pack.feature_list + [self.feature_pack.distance_name]]
            y = data[self.feature_pack.energy_name]
        else:
            raise NotImplemented(f"{self.predict_type} not supported by RandomForest")
        self.model = self.model.fit(x.values, y.values)

    def predict(self, data: DataFrame) -> Series:
        """Apply the estimator to to predict consumption.

        Args:
        data (pandas.DataFrame):
            Columns that match self.features and self.distance that
            describe vehicle passes over links in the road network.

        Returns:
            target_pred (float):
                Predicted target for every row in links_df.

        """
        if self.predict_type == PredictType.ENERGY_RATE:
            x = data[self.feature_pack.feature_list]
            _energy_pred_rates = self.model.predict(x.values)
            _energy_pred = _energy_pred_rates * data[self.feature_pack.distance_name]
        elif self.predict_type == PredictType.ENERGY_RAW:
            x = data[self.feature_pack.feature_list + [self.feature_pack.distance_name]]
            _energy_pred = self.model.predict(x.values)
        else:
            raise NotImplemented(f"{self.predict_type} not supported by RandomForest")

        energy_pred = Series(clip(_energy_pred, a_min=0), name=self.predict_type.name)

        return energy_pred

    def feature_importance(self):
        return self.model.feature_importances_
