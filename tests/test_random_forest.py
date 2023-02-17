from unittest import TestCase

from powertrain import Model
from powertrain.estimators.random_forest import RandomForest, RandomForestRegressor
from tests.mock_resources import *


class TestPredict(TestCase):
    def test_rf_model_custom(self):

        data, feature_pack = mock_data_single_feature()

        custom_model = RandomForestRegressor()

        est = RandomForest(feature_pack, model=custom_model)

        rf_model = Model(est)

        rf_model.train(data)
