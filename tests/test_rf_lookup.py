from pathlib import Path
from unittest import TestCase

import numpy as np

from powertrain.core.features import FeatureRange
from powertrain.estimators.rf_lookup import RandomForestLookup
from tests.mock_resources import *

data_path = Path("routee-powertrain-test-data") / "test_data_2016_TOYOTA_Corolla_4cyl_2WD.csv"


class TestRandomForestLookup(TestCase):
    def test_rf_lookup(self):
        data = pd.read_csv(data_path)

        features = (
            Feature('gpsspeed', units='mph', feature_range=FeatureRange(0, 100)),
            Feature('grade', units='decimal', feature_range=FeatureRange(-0.2, 0.2))
        )
        distance = Feature('miles', units='mi')
        energy = Feature('gge', units='gallons')
        feature_pack = FeaturePack(features, distance, energy)

        data["energy_rate"] = data[feature_pack.energy.name] / data[feature_pack.distance.name]
        data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

        rf_lookup = RandomForestLookup(feature_pack)

        rf_lookup.train(data, grid_shape=(100, 50))
        rf_lookup.predict(data)
