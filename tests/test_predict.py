from pathlib import Path
from unittest import TestCase

from powertrain import Model
from powertrain.estimators.explicit_bin import ExplicitBin
from tests.mock_resources import *


class TestPredict(TestCase):
    def test_eb_model_predict(self):
        route = mock_route()

        route = pd.DataFrame([{'distance': 1, 'speed': 65, 'grade': 0.1}])

        eb_model = Model.from_json("/Users/nreinick/Downloads/speed_grade/2016_Toyota_Camry_LookupTable.json")

        predictions = eb_model.predict(route)

        self.assertEqual(len(predictions), len(route), 'should produce same number of links')

        # TODO: check that predicted energy is in reasonable range for this test route.

    def test_eb_single_index(self):
        outfile = Path(".tmpfile.json")

        train_data, feature_pack = mock_data_single_feature()
        eb = ExplicitBin(feature_pack)
        m = Model(eb)

        m.train(train_data)

        m.to_json(outfile)

        new_m = Model.from_json(outfile)

        outfile.unlink()

