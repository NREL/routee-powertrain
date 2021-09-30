from pathlib import Path
from unittest import TestCase

from powertrain import Model
from powertrain.estimators.explicit_bin import ExplicitBin, BIN_DEFAULTS
from tests.mock_resources import *


class TestPredict(TestCase):
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

        new_m = Model.from_json(outfile)

        outfile.unlink()

