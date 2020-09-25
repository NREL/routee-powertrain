import pandas as pd
import sys

import routee
from routee.estimators import RandomForest, ExplicitBin
from routee.core.model import Feature, Distance, Energy

FILEPATH = "data/2016_FORD_Escape_4cyl_2WD.csv"
VEH_NAME = "FUNC TEST - 2016 Ford Escape"
PICKLE_OUT_PATH = "data/"

print('reading data..')
df = pd.read_csv(FILEPATH, index_col=False)
df['grade'] = df.grade * 100.0
df['speed_mph'] = (df['start_mph'] + df['end_mph']) / 2

energy = Energy('gallons', units='gallons')

features = [
    Feature('speed_mph', units='mph'),
    Feature('grade', units='decimal')
]

distance = Distance('miles', units='miles')

train_df = df[['miles', 'speed_mph', 'grade', energy.name]].dropna()

train_df = train_df[train_df.miles>0]

ln_model = routee.Model(VEH_NAME)
rf_model = routee.Model(VEH_NAME, estimator=RandomForest(cores=4))
eb_model = routee.Model(VEH_NAME, estimator=ExplicitBin(features=features, distance=distance, energy=energy))

print('training models..')
ln_model.train(train_df, features=features, distance=distance, energy=energy)
rf_model.train(train_df, features=features, distance=distance, energy=energy)
eb_model.train(train_df, features=features, distance=distance, energy=energy)

# print(rf_model.errors)
