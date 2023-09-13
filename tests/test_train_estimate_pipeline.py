import logging as log
import math
from pathlib import Path
from unittest import TestCase

import pandas as pd

import nrel.routee.powertrain as pt
from nrel.routee.powertrain.estimators.onnx import ONNXEstimator
from nrel.routee.powertrain.estimators.smart_core import SmartCoreEstimator

from nrel.routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)
from nrel.routee.powertrain.trainers.smartcore_random_forest import (
    SmartcoreRandomForestTrainer,
)

this_dir = Path(__file__).parent

log.basicConfig(level=log.INFO)


class TestTrainEstimatePipeline(TestCase):
    def setUp(self) -> None:
        data_path = (
            this_dir
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )
        self.df = pd.read_csv(data_path)
        self.out_path = Path("tmp")
        self.out_path.mkdir(exist_ok=True)
        self.feature_pack = pt.FeaturePack(
            features=[
                pt.Feature(name="speed", units="mph"),
                pt.Feature(name="grade", units="decimal"),
            ],
            distance=pt.Feature(name="miles", units="miles"),
            energy=[
                pt.Feature(
                    name="gallons_fastsim",
                    units="gallons_gasoline",
                    feature_range=pt.FeatureRange(lower=0.0, upper=100.0),
                )
            ],
        )
        self.config = pt.ModelConfig(
            vehicle_description="Test Model",
            powertrain_type=pt.PowertrainType.ICE,
            feature_pack=self.feature_pack,
        )

    def tearDown(self) -> None:
        pass
        self.out_path.rmdir()

    def test_sklearn_random_forest(self):
        trainer = SklearnRandomForestTrainer()

        vehicle_model = trainer.train(self.df, self.config)

        r1 = vehicle_model.predict(self.df)
        energy1 = round(r1.gallons_fastsim.sum(), 2)

        # test out writing and reading to file
        outfile = self.out_path / "model.json"
        vehicle_model.to_file(outfile)
        new_vehicle_model = pt.read_model(outfile)
        outfile.unlink()

        # test writing inner estimator to file
        outfile = self.out_path / "estimator.onnx"
        vehicle_model.estimator.to_file(outfile)
        new_estimator = ONNXEstimator.from_file(outfile)
        outfile.unlink()

        r2 = new_vehicle_model.predict(self.df)
        energy2 = round(r2.gallons_fastsim.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2))

    def test_smartcore_random_forest(self):
        trainer = SmartcoreRandomForestTrainer()

        vehicle_model = trainer.train(self.df, self.config)

        r1 = vehicle_model.predict(self.df)
        energy1 = round(r1.gallons_fastsim.sum(), 2)

        # test out writing and reading to file
        outfile = self.out_path / "model.json"
        vehicle_model.to_file(outfile)
        new_vehicle_model = pt.read_model(outfile)
        outfile.unlink()

        # test writing inner estimator to file
        outfile = self.out_path / "estimator.json"
        vehicle_model.estimator.to_file(outfile)
        new_estimator = SmartCoreEstimator.from_file(outfile)
        outfile.unlink()

        outfile = self.out_path / "estimator.bin"
        vehicle_model.estimator.to_file(outfile)
        new_estimator = SmartCoreEstimator.from_file(outfile)
        outfile.unlink()

        r2 = new_vehicle_model.predict(self.df)
        energy2 = round(r2.gallons_fastsim.sum(), 2)

        self.assertTrue(math.isclose(energy1, energy2))
