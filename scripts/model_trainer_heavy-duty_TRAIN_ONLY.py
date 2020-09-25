import pandas as pd
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
import routee
from routee.estimators import ExplicitBin, RandomForest
from routee.core.model import Feature, Distance, Energy

"""
This script was originally run in a Jupyter Notebook on arnaud:
https://github.nrel.gov/MBAP/routee-notebooks/blob/master/notebooks/training/20200213-JH-train_hd_model.ipynb

The data is from FleetDNA, specifically for class 8 linehaul trucks

TODO: clean up script to run in routee repo instead of routee-notebooks
"""

# Train RouteE Models
# onroad_data_PATH = '/data/users/jholden/fdna_linehaul/processed/'
onroad_data_PATH = '/Users/jholden/Documents/local_data/fdna_linehaul/processed/'

## Link Aggregation
veh_csvs = os.listdir(onroad_data_PATH)
veh_csvs = [x for x in veh_csvs if x.endswith('.csv')]

# if 1:
if os.path.exists(onroad_data_PATH + 'aggregate.csv') == False:

    df_links = pd.DataFrame()
    print('aggregating point data')
    for file_i in veh_csvs:
        df = pd.read_csv(onroad_data_PATH + file_i)
        df['speed_mph'] = 0.621371 * df['WheelBasedVehicleSpeed']
        df['gallons'] = df['EngineFuelRate'] * (1 / df['speed_mph']) * (
                df['distance_ft'] / 5280) * 0.264172  # L/hr * hr/mile * mile * gal/L = gal

        df['elevation_ft_first'] = df['elevation_ft']
        df['elevation_ft_last'] = df['elevation_ft']

        agg_funcs = {'distance_ft': sum, 'elevation_ft_first': 'first', 'elevation_ft_last': 'last',
                     'speed_mph': 'mean', 'gallons': sum}

        df_grouped = df.groupby(['day_id', 'link_id']).agg(agg_funcs).reset_index()
        df_grouped['elevation_ft_delta'] = df_grouped['elevation_ft_last'] - df_grouped['elevation_ft_first']
        df_grouped['grade_dec'] = df_grouped['elevation_ft_delta']/df_grouped['distance_ft']

        df_grouped = df_grouped.replace([np.inf, -np.inf], np.nan).dropna()
        vid = file_i.split('_')[1].split('.')[0]
        df_grouped['veh_id'] = [vid] * len(df_grouped)
        df_links = df_links.append(df_grouped, ignore_index=True)

    print('aggregation complete')
    df_links.to_csv(onroad_data_PATH + 'aggregate.csv', index=False)

else:
    df_links = pd.read_csv(onroad_data_PATH + 'aggregate.csv', index_col=None)

# df_links = df_links[df_links.veh_id==250]

df_links['distance_mi'] = df_links['distance_ft'] / 5280
df_links['gal_100mi'] = 100 * df_links['gallons'] / df_links['distance_mi']  # for filtering
df_links['grade_perc'] = 100 * df_links['grade_dec']

## Filter Links
df_links_fltr = df_links[df_links.distance_mi < 3]
df_links_fltr = df_links_fltr[df_links_fltr.gal_100mi > 0]
df_links_fltr = df_links_fltr[df_links_fltr.gal_100mi < 100]

df_links_fltr = df_links_fltr[df_links_fltr.grade_dec.abs() < 0.04]
# df_links_fltr = df_links_fltr[df_links_fltr.grade_dec < df_links_fltr.grade_dec.quantile(q=.995)]
# df_links_fltr = df_links_fltr[df_links_fltr.grade_dec > df_links_fltr.grade_dec.quantile(q=.005)]

df_links_fltr = df_links_fltr[df_links_fltr.link_id != -99]

print("Filtered out %0.2f%% of links" % (100 - 100 * float(len(df_links_fltr)) / len(df_links)))
print("Total VMT = %0.0f miles" % (df_links_fltr.distance_mi.sum()))

df_links_fltr['gal_100mi'] = 100 * df_links_fltr['gallons'] / df_links_fltr['distance_mi']
# df_links_fltr.hist('grade_perc', bins=50, rwidth=0.95)
# plt.show()

## Train Models
distance = Distance('distance_mi', units='miles')
features = [
    Feature('speed_mph', units='mph'),
    Feature('grade_perc', units='percent')
]
energy = Energy('gallons', units='gallons')

eb_model = routee.Model('class_8_linehaul', estimator=ExplicitBin(features=features, distance=distance, energy=energy))
eb_model.train(df_links_fltr, features=features, distance=distance, energy=energy)
eb_model.dump_model('../routee/trained_models/class_8_linehaul_diesel_Explicit_Bin.pickle')

rf_model = routee.Model('class_8_linehaul', estimator=RandomForest(cores=3))
rf_model.train(df_links_fltr, features=features, distance=distance, energy=energy)
rf_model.dump_model('../routee/trained_models/class_8_linehaul_diesel_Random_Forest.pickle')

base_model = routee.Model('class_8_linehaul')
base_model.train(df_links_fltr, features=features, distance=distance, energy=energy)
base_model.dump_model('../routee/trained_models/class_8_linehaul_diesel_Linear.pickle')
