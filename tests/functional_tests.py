import pandas as pd
import logging as log

from powertrain.estimators import RandomForest, ExplicitBin, LinearRegression
from powertrain.core.model import Model
from powertrain.core.features import Feature, FeaturePack
from powertrain.utils.fs import root

FILEPATH = root() / "tests" / "routee-powertrain-test-data" / "links_fastsim_2014mazda3.csv"

VEH_NAME = "FUNC TEST - 2014 Mazda 3"

df = pd.read_csv(FILEPATH, index_col=False)
df['grade'] = df.grade * 100.0
df['speed_mph'] = df['gpsspeed']

features = (
    Feature('speed_mph', units='mph'),
    Feature('grade', units='decimal')
)
distance = Feature('miles', units='mi')
energy = Feature('gge', units='gallons')
feature_pack = FeaturePack(features, distance, energy)

train_df = df[['miles', 'speed_mph', 'grade', energy.name]].dropna()

train_df = train_df[train_df.miles > 0]

ln_e = LinearRegression(feature_pack=feature_pack)
rf_e = RandomForest(feature_pack=feature_pack)
eb_e = ExplicitBin(feature_pack=feature_pack)

for e in (ln_e, rf_e, eb_e):
    log.info(f"training estimator {e}..")
    m = Model(e, veh_desc=VEH_NAME)
    m.train(train_df)

