import logging as log
import math
from pathlib import Path
from unittest import TestCase

import pandas as pd

import nrel.routee.powertrain as pt

from nrel.routee.powertrain.trainers.random_forest import RandomForestTrainer

this_dir = Path(__file__).parent

log.basicConfig(level=log.INFO)


class TestTrainPipeline(TestCase):
    def test_train_pipeline(self):
        data_path = (
            this_dir
            / Path("routee-powertrain-test-data")
            / Path("sample_train_data.csv")
        )

        out_path = Path("tmp")
        out_path.mkdir(exist_ok=True)
        df = pd.read_csv(data_path)

        feature_pack = pt.FeaturePack(
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

        config = pt.ModelConfig(
            vehicle_description="Test Model",
            powertrain_type=pt.PowertrainType.ICE,
            feature_pack=feature_pack,
        )

        trainer = RandomForestTrainer()

        vehicle_model = trainer.train(df, config)

        r1 = vehicle_model.predict(df)[0]
        energy1 = round(r1.sum(), 2)

        # test out writing and reading to file
        outfile = out_path / "model.onnx"
        vehicle_model.to_file(outfile)
        new_vehicle_model = pt.read_model(outfile)
        outfile.unlink()

        r2 = new_vehicle_model.predict(df)[0]
        energy2 = round(r2.sum(), 2)

        if math.isclose(energy1, energy2):
            log.info("\n\n ✅ Successfully saved and loaded model! \n\n")
        else:
            log.info(
                "\n\n ❌ The model loaded from file did not predict the same energy  \n\n"
            )

        out_path.rmdir()
