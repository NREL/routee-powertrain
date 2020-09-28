import pandas as pd
import sys
import os

import powertrain
from powertrain.estimators import RandomForest, ExplicitBin
from powertrain.core.model import Feature, Distance, Energy

FILEPATH = os.path.join("routee-powertrain-test-data", "links_fastsim_2014mazda3.csv")
VEH_NAME = "FUNC TEST - 2014 Mazda 3"
PICKLE_OUT_PATH = os.path.join("routee-powertrain-test-data")

df = pd.read_csv(FILEPATH, index_col=False)
df['grade'] = df.grade * 100.0
df['speed_mph'] = df['gpsspeed']

energy = Energy('gge', units='gallons')

features = [
    Feature('speed_mph', units='mph'),
    Feature('grade', units='decimal')
]

distance = Distance('miles', units='miles')

train_df = df[['miles', 'speed_mph', 'grade', energy.name]].dropna()

train_df = train_df[train_df.miles>0]

ln_model = powertrain.Model(VEH_NAME)
rf_model = powertrain.Model(VEH_NAME, estimator=RandomForest(cores=4))
eb_model = powertrain.Model(VEH_NAME, estimator=ExplicitBin(features=features, distance=distance, energy=energy))

print('training models')
ln_model.train(train_df, features=features, distance=distance, energy=energy)
rf_model.train(train_df, features=features, distance=distance, energy=energy)
eb_model.train(train_df, features=features, distance=distance, energy=energy)

# print(rf_model.errors)
