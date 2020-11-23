import sys
sys.path.append('/home/jholden/GitHub/routee/')
import routee
from routee.estimators import ExplicitBin, RandomForest
from routee.core.model import Feature, Distance, Energy

import time
import pandas as pd
import numpy as np
import sqlalchemy as sql
import os
import yaml
import datetime

import multiprocessing as mp

import logging

logging.basicConfig(filename='batch_run.log', \
                    filemode='w', \
                    level=logging.DEBUG,\
                    format='%(asctime)s %(message)s')

logging.info('RouteE batch run START')


def load_config(config_file):
    """
    Load the user config file, config.yml
    This is where all configurations for the batch run are stored.
    """
    
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        

        
def run_routee(tuple_input):
    
    fs_path_file = tuple_input[0]
    routee_path = tuple_input[1]
    
    engine = sql.create_engine('sqlite:///'+fs_path_file)
    
    veh_desc = fs_path_file.split('/')[-1].split('.')[0]
    
    scenario_name = veh_desc.split('_CAV')[0].replace('_',' ')
    
    if 'Depleting' in veh_desc:
        ptType = 4
        phev_desc = 'charge_depleting'
        
    elif 'Sustaining' in veh_desc: 
        ptType = 2
        phev_desc = 'charge_sustaining'
        
#     elif 'Tesla' in veh_desc:
#         ptType = 4
#         phev_desc = ''
        
    elif '_ev_' in veh_desc or 'kWh' in veh_desc or 'TESLA' in veh_desc or 'MiEV' in veh_desc or 'Fuel_Cell' in veh_desc or 'Bolt' in veh_desc or 'Mirai' in veh_desc:
        ptType = 4
        phev_desc = ''
        
    else:
        ptType = 1
        phev_desc = ''

    # Identify whether or not vehicle model is EV or PHEV
    if ptType == 4:
        e_unit = 'esskwhoutach'
    else:
        e_unit = 'gallons'
    
    # Read FASTSim Results
    quer = """
            SELECT *
            FROM links
            """

    df_passes = pd.read_sql(quer, engine)
    
    df_passes['miles'] = df_passes['miles'].astype(float)
    df_passes['gpsspeed'] = df_passes['gpsspeed'].astype(float)
    df_passes['esskwhoutach'] = df_passes['esskwhoutach'].astype(float)
    df_passes['seconds'] = df_passes['seconds'].astype(float)
    df_passes['gallons'] = df_passes['gge'].astype(float)
    
    # Pre-process data   
    ### convert elevation change to gradient with units of    
    df_passes['grade_percent_float'] = 100.0*df_passes.grade 

    ### combine sampno, vehno, tripno to make a unique trip_id column for trip error calcs
    df_passes['trip_ids'] = (df_passes.sampno.astype(str)+
                             df_passes.vehno.astype(str)+
                             df_passes.tripno.astype(str)).astype(float)


    ### filter out VERY short link passes
    df_passes = df_passes[df_passes.miles>0.002]
    
    
    features = [
        Feature('gpsspeed', units='mph'),
        Feature('grade_percent_float', units='percent_0_100')]
    
    distance = Distance('miles', units='mi')
    
    energy = Energy(e_unit, units=e_unit)
    
    ln_model = routee.Model(scenario_name)
    rf_model = routee.Model(scenario_name, estimator=RandomForest(cores=4))
    eb_model = routee.Model(scenario_name, estimator=ExplicitBin(features=features, distance=distance, energy=energy))

    models = {
        'Linear': ln_model,
        'Random Forest': rf_model,
        'Explicit Bin': eb_model,
    }

    for name, model in models.items():
        model.train(df_passes, features=features, distance=distance, energy=energy)
        model.dump_model(routee_path + '/' + scenario_name.replace(' ','_') + '_' + name.replace(' ','_') + ".pickle")
    
#     eb_model.train(df_passes, features=features, distance=distance, energy=energy)
    
#     eb_model._estimator.model.reset_index().to_csv(routee_path_file, index=False)
        
# def run_routee_mp():
    
            
if __name__ == '__main__':
    
    config = load_config('config.yml')
    
    ts = time.time()
    
    # Initialize results location
    results_dir = config['vehicles_db'].replace('.csv','')
    results_dir = results_dir.replace(' ','_')
    
    logging.info('Inititalizing results directory: %s' % (config['routee_results_path']+results_dir))
    
    if not os.path.exists(config['routee_results_path']+results_dir):
        os.makedirs(config['routee_results_path']+results_dir)
        
    # Read FASTSim results DBs
    fs_results_dbs = os.listdir(config['fastsim_results_path'])
    fs_results_dbs = [fn for fn in fs_results_dbs if fn.endswith('.db')]
    
    # Build tuple input to sub
    tuple_input=[]
    for i in list(range(len(fs_results_dbs))):
        tuple_input.append((config['fastsim_results_path']+fs_results_dbs[i],\
                            config['routee_results_path']+results_dir))
#         tuple_input.append((config['fastsim_results_path']+fs_results_dbs[i],\
#                             config['routee_results_path']+results_dir+'/'+fs_results_dbs[i].replace('.db','.csv')))
#     run_routee(tuple_input[0])
    
    ## Multiprocessing
    pool = mp.Pool(processes = config['n_cores'])
    
    pool.map(run_routee, tuple_input)
    
    pool.close()
    pool.terminate()
    pool.join()

    runtime = (time.time() - ts)/60.0 #minutes
    logging.info('COMPLETE! Runtime = %0.2f mins' % runtime)