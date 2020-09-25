import pandas as pd
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

sys.path.append('../../../routee/')
from routee.validation import errors
import routee
from routee.estimators import ExplicitBin, RandomForest
from routee.core.model import Feature, Distance, Energy

sys.path.append('../../../mapit/')
from mapit.tomtom2014 import tomtom2014

sys.path.append('../../../GradeIT/')
from gradeit.gradeit import gradeit

"""
This script was originally run in a Jupyter Notebook on arnaud:
https://github.nrel.gov/MBAP/routee-notebooks/blob/master/notebooks/training/20200213-JH-train_hd_model.ipynb

The data is from FleetDNA, specifically for class 8 linehaul trucks

TODO: clean up script to run in routee repo instead of routee-notebooks
"""


vids = [24, 25, 26, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
        143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 249, 250, 251, 252, 253, 254, 9890, 9891, 9892, 9893, 9894, 9895, 9896, 9897, 9898, 9899, 9900,
        9901, 9902, 9903, 9906, 9907, 9908, 11637, 11638, 11639, 11640, 11641, 11642, 11660, 11661, 11803, 11804, 11805, 11806, 11807, 11808, 11809, 11810, 11811,
        11813, 11814, 11815, 11816, 11817, 11818, 11819, 11820, 11821, 11822, 11823, 11824, 11825, 11826, 11827, 11828, 11829, 11830, 11831, 11832, 12100, 12101,
        12102, 12103, 12104, 12105, 12106, 12107, 12108, 12109, 12110, 12111, 12112, 12113, 12114, 12115, 12116, 12117, 12118, 12119, 12120, 12121, 12122, 12123,
        12124, 12125, 12126, 12127, 12128]

data_path = '/Volumes/cfds/FleetDNA/Processed/'

# local_data_path = '/Users/jholden/Documents/local_data/fdna_linehaul/'
local_data_path = '/data/users/jholden/fdna_linehaul/'

vid_dirs = ['v_'+str(vid) for vid in vids]

# Read Data
ts=time.time()

colnames = ['cycle_sec',
            'EngineFuelRate',
            'speed',
            'WheelBasedVehicleSpeed',
            'Latitude',
            'Longitude']

df = pd.DataFrame()

# iterate through every vehicle
for vid_i in vid_dirs:
    days = os.listdir(data_path + vid_i)
    days = [day for day in days if day.startswith('.')==False]
    df_veh = pd.DataFrame()

    for day_i in days:
        df_day = pd.DataFrame()
        for col in colnames:
            try:
                data_col = pd.read_json(data_path + vid_i + '/' + day_i + '/' +col+'.json', orient='columns')
            except ValueError:
                print("column {0}, vehicle {1}, day {2}".format(col, vid_i, day_i))
            df_day = pd.concat([df_day, data_col], axis=1, ignore_index=True)
        df_day.columns = colnames
        df_day['vid'] = [int(vid_i.split('_')[1])]*len(df_day)
        df_day['day_id'] = [day_i]*len(df_day)

        df_veh = df_veh.append(df_day, ignore_index=True)
#         df = df.append(df_day, ignore_index=True)

    df_veh.to_csv(local_data_path+str(vid_i)+'.csv', index=False)

print('Runtime = %0.2f secs' % (time.time() - ts))
print('%d points' % (len(df)))

# Mapit & GradeIT
t0 = time.time()
df = pd.read_csv(local_data_path+vid_dirs[0]+'.csv')
t1 = time.time()
print('Read time = %0.2f secs' %(t1-t0))

link_ids = tomtom2014(df.Latitude, df.Longitude).links
t2 = time.time()
print('Map-match time = %0.2f secs' % (t2-t1))

# df_out = gradeit(df=df, lat_col='Latitude', lon_col='Longitude', filtering=True, source='usgs-local',
#                    usgs_db_path="/Volumes/ssh/backup/mbap_shared/NED_13/")
df_out = gradeit(df=df, lat_col='Latitude', lon_col='Longitude', filtering=True, source='usgs-local',
                   usgs_db_path="/backup/mbap_shared/NED_13/")
t3 = time.time()
print('Grade time = %0.2f secs' % (t3-t2))
print('')
print('Total runtime for {} points = {} mins'.format(len(df), (t3-t0)/60.0))

# Run all vehicles
t0 = time.time()

for vid_i in vid_dirs:

    df = pd.read_csv(local_data_path + vid_i + '.csv')

    link_ids = tomtom2014(df.Latitude, df.Longitude).links

    try:
        df_out = gradeit(df=df, lat_col='Latitude', lon_col='Longitude', filtering=True, source='usgs-local',
                         usgs_db_path="/backup/mbap_shared/NED_13/")
    except:
        continue

    df_out['link_id'] = link_ids

    outpath = local_data_path + 'processed/'
    df_out.to_csv(outpath + vid_i + '.csv', index=False)

print('Total runtime = {} mins'.format((time.time() - t0) / 60.0))


df_out['link_id'] = link_ids


# Train RouteE Models
onroad_data_PATH = '/data/users/jholden/fdna_linehaul/processed/'

## Link Aggregation
veh_csvs = os.listdir(onroad_data_PATH)
veh_csvs = [x for x in veh_csvs if x.endswith('.csv')]

df_links = pd.DataFrame()

for file_i in veh_csvs:
    df = pd.read_csv(onroad_data_PATH+file_i)
    df['speed_mph'] = 0.621371 * df['WheelBasedVehicleSpeed']
    df['gallons'] = df['EngineFuelRate'] * (1/df['speed_mph']) * (df['distance_ft']/5280) * 0.264172 # L/hr * hr/mile * mile * gal/L = gal

    agg_funcs = {'distance_ft':sum, 'grade_dec':'mean', 'speed_mph':'mean', 'gallons':sum}

    df_grouped = df.groupby(['day_id','link_id']).agg(agg_funcs).reset_index()
    df_grouped = df_grouped.replace([np.inf, -np.inf], np.nan).dropna()
    vid = file_i.split('_')[1].split('.')[0]
    df_grouped['veh_id'] = [vid]*len(df_grouped)
    df_links = df_links.append(df_grouped, ignore_index=True)

df_links['gal_100mi'] = 100*df_links['gallons']/(df_links['distance_ft']/5280)

## Filter Links
df_links_fltr = df_links[df_links.distance_ft < 4000]
df_links_fltr = df_links_fltr[df_links_fltr.gal_100mi > 0]
df_links_fltr = df_links_fltr[df_links_fltr.gal_100mi < 35]
df_links_fltr = df_links_fltr[df_links_fltr.grade_dec.abs() < 0.12]
df_links_fltr = df_links_fltr[df_links_fltr.link_id != -99]

print("Filtered out %0.2f%% of links" % (100 - 100*float(len(df_links_fltr))/len(df_links)))
print("Total VMT = %0.0f miles" % (df_links_fltr.distance_ft.sum()/5280))

df_links_fltr['gal_100mi'] = 100*df_links_fltr['gallons']/(df_links_fltr['distance_ft']/5280)
df_links_fltr.plot(x='speed_mph', y='gal_100mi', kind='scatter', alpha=.005)

## Train Models
distance = Distance('distance_ft', units='ft')
features = [
    Feature('speed_mph', units='mph'),
    Feature('grade_dec', units='decimal')
]
energy = Energy('gallons', units='gallons')

base_model = routee.Model('class_8_linehaul')
rf_model = routee.Model('class_8_linehaul', estimator = RandomForest(cores = 12))


base_model.train(df_links_fltr, features=features, distance=distance, energy=energy)
rf_model.train(df_links_fltr, features=features, distance=distance, energy=energy)


rf_model.dump_model('../../test_data/trained_models/class_8_linehaul_diesel.pickle')