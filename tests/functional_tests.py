"""
NOTE: to run this functional test, you'll need to get the file: links_fastsim_2014mazda3.csv
from the location:

https://app.box.com/s/dm5w4mo56ej9jfmyo404kz98roz7jat7

"""
import logging as log
import math
from pathlib import Path

import pandas as pd

from powertrain.core.features import Feature, FeaturePack
from powertrain.core.model import Model
from powertrain.estimators.explicit_bin import ExplicitBin
from powertrain.estimators.linear_regression import LinearRegression
from powertrain.estimators.random_forest import RandomForest
from tests.mock_resources import mock_route

log.basicConfig(level=log.INFO)

data_path = Path("routee-powertrain-test-data") / Path("links_fastsim_2014mazda3.csv")

out_path = Path("tmp")
out_path.mkdir(exist_ok=True)

veh_name = "FUNC TEST - 2014 Mazda 3"

df = pd.read_csv(data_path, index_col=False)
df['grade'] = df.grade * 100.0

features = (
    Feature('gpsspeed', units='mph'),
    Feature('grade', units='decimal')
)
distance = Feature('miles', units='mi')
energy = Feature('gge', units='gallons')
feature_pack = FeaturePack(features, distance, energy)

train_df = df[['miles', 'gpsspeed', 'grade', energy.name]].dropna()
train_df = train_df[train_df.miles > 0]

ln_e = LinearRegression(feature_pack=feature_pack)
rf_e = RandomForest(feature_pack=feature_pack)
eb_e = ExplicitBin(feature_pack=feature_pack)

if __name__ == "__main__":

    for e in (ln_e, rf_e, eb_e):
        log.info(f"training estimator {e}..")
        m = Model(e, description=veh_name)
        m.train(train_df)
        log.info(f"errors: {m.metadata.errors}")

        # test out prediction
        log.info(f"predicting {m} over test route..")
        r1 = m.predict(mock_route())

        energy1 = round(r1.sum(), 2)
        log.info(f"predicted {energy1} gge over test route..")

        # test out writing and reading json
        log.info(f"writing {m} to json..")
        json_outfile = out_path.joinpath("model.json")
        m.to_json(json_outfile)

        log.info(f"reading {json_outfile}..")
        new_m = Model.from_json(json_outfile)

        log.info(f"predicting {new_m} over test route..")
        r2 = new_m.predict(mock_route())

        energy2 = round(r2.sum(), 2)
        log.info(f"predicted {energy2} gge over test route..")

        if math.isclose(energy1, energy2):
            log.info("\n\n ✅ Successfully saved and loaded model in json format! \n\n")
        else:
            log.info("\n\n ❌ The model loaded from json did not predict the same energy  \n\n")

        log.info("removing json file..")
        json_outfile.unlink()

        # test out writing and reading pickle
        log.info(f"writing {m} to pickle..")
        pickle_outfile = out_path.joinpath("model.pickle")
        m.to_pickle(pickle_outfile)

        log.info(f"reading {pickle_outfile}..")
        new_m = Model.from_pickle(pickle_outfile)

        log.info(f"predicting {new_m} over test route..")
        r3 = new_m.predict(mock_route())

        energy3 = round(r3.sum(), 2)
        log.info(f"predicted {energy3} gge over test route..")

        if math.isclose(energy1, energy3):
            log.info("\n\n ✅ Successfully saved and loaded model in pickle format! \n\n")
        else:
            log.info("\n\n ❌ The model loaded from pickle did not predict the same energy  \n\n")

        log.info("removing pickle file..")
        pickle_outfile.unlink()

    out_path.rmdir()
