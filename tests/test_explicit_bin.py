import math
from pathlib import Path
from unittest import TestCase

from powertrain import Model
from powertrain.estimators.explicit_bin import ExplicitBin, BIN_DEFAULTS
from tests.mock_resources import *


class TestPredict(TestCase):
    def test_eb_model_predict(self):
        route = pd.DataFrame([{'miles': 1, 'gpsspeed': 65, 'grade': 0.1}])

        eb_model = mock_model()

        predictions = eb_model.predict(route)

        self.assertEqual(len(predictions), len(route), 'should produce same number of links')

        # TODO: check that predicted energy is in reasonable range for this test route.

    def test_eb_no_bins(self):
        outfile = Path(".tmpfile.json")

        train_data = mock_ice_data()

        feature_pack = FeaturePack(
            features=(Feature(name="gpsspeed", units=""),),
            distance=Feature(name="miles", units=""),
            energy=Feature(name="gge", units=""),
        )

        eb = ExplicitBin(feature_pack)

        m = Model(eb)

        m.train(train_data)

        # collect predictions from model
        route = mock_route()
        r1 = m.predict(route)
        energy1 = round(r1.sum(), 2)

        # test writing and reading file
        m.to_json(outfile)
        new_m = Model.from_json(outfile)

        # make sure the model predicts the same after being loaded from file
        r2 = new_m.predict(route)
        energy2 = round(r2.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2), "original model and json model should predict similar energy")

        # clean up
        outfile.unlink()

    def test_eb_single_index(self):
        outfile = Path(".tmpfile.json")

        train_data, feature_pack = mock_data_single_feature()

        bins = {
            'speed': BIN_DEFAULTS['speed_mph'],
        }

        eb = ExplicitBin(feature_pack, bins=bins)

        m = Model(eb)

        m.train(train_data)

        m.to_json(outfile)

        _ = Model.from_json(outfile)

        outfile.unlink()

