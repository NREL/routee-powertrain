import sqlite3
import pandas as pd
import sys
import glob
import os
import pickle

sys.path.append('..')
import routee as rte

from routee.estimators import RandomForest, ExplicitBin
from routee.core.model import Feature, Distance, Energy

RAW_DATA_PATH = "/data/fastsim-results/2020_03_27_routee_library/routee_fastsim_veh_db/*.db"
PICKLE_OUT_PATH = "../routee/trained_models/"

raw_files = glob.glob(RAW_DATA_PATH)
print(f"Total of {len(raw_files)}")


def train_model(file):
    vehicle_name = os.path.splitext(os.path.basename(file))[0]

    print(f'Working on vehicle: {vehicle_name}')

    features = [
        Feature('gpsspeed', units='mph'),
        Feature('grade', units='percent_0_100'),
    ]
    distance = Distance('miles', units='mi')

    sql_con = sqlite3.connect(file)

    df = pd.read_sql_query('SELECT * FROM links', sql_con)
    df['grade'] = df.grade.apply(lambda x: x * 100)

    if df.gge.sum() > 0:
        energy = Energy('gge', units='gallons')
    elif df.esskwhoutach.sum() > 0:
        energy = Energy('esskwhoutach', units='kwh')
    else:
        raise RuntimeError('There is no energy reported in this data file..')

    train_df = df[['miles', 'gpsspeed', 'grade', energy.name]].dropna()

    ln_model = rte.Model(vehicle_name)
    rf_model = rte.Model(vehicle_name, estimator=RandomForest(cores=4))
    eb_model = rte.Model(vehicle_name, estimator=ExplicitBin(features=features, distance=distance, energy=energy))

    models = {
        'Linear': ln_model,
        'Random Forest': rf_model,
        'Explicit Bin': eb_model,
    }

    for name, model in models.items():
        model.train(train_df, feature_pack=features, distance=distance, energy=energy)
        model.dump_model(PICKLE_OUT_PATH + vehicle_name + '_' + name.replace(' ','_') + ".pickle")
        
#     with open(PICKLE_OUT_PATH + vehicle_name + "_errors.pickle", 'wb') as f:
#         pickle.dump(model_errors, f)


from multiprocessing import Pool

num_cores = 6 

with Pool(num_cores) as p:
    p.map(train_model, raw_files)

