import logging as log

import pandas as pd

from powertrain.core.features import Feature, FeaturePack
from powertrain.core.model import Model
from powertrain.estimators import RandomForest, ExplicitBin, LinearRegression
from powertrain.utils.fs import root
from tests.mock_resources import mock_route

log.basicConfig(level=log.INFO)

data_path = root() / "tests" / "routee-powertrain-test-data" / "links_fastsim_2014mazda3.csv"

out_path = root() / "tests" / "tmp"
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
        m = Model(e, veh_desc=veh_name)
        m.train(train_df)

        # test out prediction
        log.info(f"predicting {m} over test route..")
        r = m.predict(mock_route())

        energy = round(r.sum(), 2)
        log.info(f"predicted {energy} gge over test route..")

        # test out writing and reading json
        log.info(f"writing {m} to json..")
        json_outfile = out_path.joinpath("model.json")
        m.to_json(json_outfile)

        log.info(f"reading {json_outfile}..")
        new_m = Model.from_json(json_outfile)

        log.info(f"predicting {new_m} over test route..")
        r = new_m.predict(mock_route())

        energy = round(r.sum(), 2)
        log.info(f"predicted {energy} gge over test route..")

        log.info("removing json file..")
        json_outfile.unlink()

        # test out writing and reading pickle
        log.info(f"writing {m} to pickle..")
        pickle_outfile = out_path.joinpath("model.pickle")
        m.to_pickle(pickle_outfile)

        log.info(f"reading {pickle_outfile}..")
        new_m = Model.from_pickle(pickle_outfile)

        log.info(f"predicting {new_m} over test route..")
        r = new_m.predict(mock_route())

        energy = round(r.sum(), 2)
        log.info(f"predicted {energy} gge over test route..")

        log.info("removing pickle file..")
        pickle_outfile.unlink()

    out_path.rmdir()
