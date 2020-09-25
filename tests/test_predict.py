from unittest import TestCase

from tests.mock_resources import mock_route, mock_model


class TestPredict(TestCase):
    def test_eb_model_predict(self):
        route = mock_route()

        eb_model = mock_model("ExplicitBin")

        predictions = eb_model.predict(route)

        self.assertEqual(len(predictions), len(route), 'should produce same number of links')

        # TODO: check that predicted energy is in reasonable range for this test route.

    def test_lin_model_predict(self):
        route = mock_route()

        lin_model = mock_model("Linear")

        predictions = lin_model.predict(route)

        self.assertEqual(len(predictions), len(route), 'should produce same number of links')

        # TODO: check that predicted energy is in reasonable range for this test route.

    def test_rf_model_predict(self):
        route = mock_route()

        rf_model = mock_model("RandomForest")

        predictions = rf_model.predict(route)

        self.assertEqual(len(predictions), len(route), 'should produce same number of links')

        # TODO: check that predicted energy is in reasonable range for this test route.
