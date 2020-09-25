import pandas as pd
import sys
import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt

import routee

PATH_to_models = "../routee/trained_models/"


EV = [
    '2016_CHEVROLET_Spark_EV',
    '2016_MITSUBISHI_i-MiEV',
    '2016_Nissan_Leaf_24_kWh',
    '2016_Nissan_Leaf_30_kWh',
    '2016_TESLA_Model_S60_2WD',
    '2017_CHEVROLET_Bolt',
]

PHEV_CD = [
    '2016_BMW_i3_REx_PHEV_Charge_Depleting',
    '2016_CHEVROLET_Volt_Charge_Depleting',
    '2016_FORD_C-MAX_PHEV_Charge_Depleting',
    '2016_HYUNDAI_Sonata_PHEV_Charge_Depleting',
    '2017_Prius_Prime_Charge_Depleting',
]

HEV = [
    '2016_BMW_i3_REx_PHEV_Charge_Sustaining',
    '2016_CHEVROLET_Volt_Charge_Sustaining',
    '2016_FORD_C-MAX_PHEV_Charge_Sustaining',
    '2016_HYUNDAI_Sonata_PHEV_Charge_Sustaining',
    '2017_Prius_Prime_Charge_Sustaining',
    '2015_Honda_Accord_HEV',
    '2016_FORD_C-MAX_HEV',
    '2016_KIA_Optima_Hybrid',
    '2016_TOYOTA_Highlander_Hybrid',
]

CV = [
    '2016_AUDI_A3_4cyl_2WD',
    '2016_BMW_328d_4cyl_2WD',
    '2016_CHEVROLET_Malibu_4cyl_2WD',
    '2016_FORD_Escape_4cyl_2WD',
    '2016_FORD_Explorer_4cyl_2WD',
    '2016_HYUNDAI_Elantra_4cyl_2WD',
    '2016_TOYOTA_Camry_4cyl_2WD',
    '2016_TOYOTA_Corolla_4cyl_2WD'
]


eb_model_list = [x for x in os.listdir(PATH_to_models) if x.endswith('Explicit_Bin.pickle')]
rf_model_list = [x for x in os.listdir(PATH_to_models) if x.endswith('Random_Forest.pickle')]

root_list = [x.split('_Explicit_Bin')[0] for x in eb_model_list]

eb_model_error_dwrpe = []
rf_model_error_dwrpe = []

eb_model_error_net = []
rf_model_error_net = []

for basename_i in root_list:
    eb_model = routee.read_model(PATH_to_models+basename_i+'_Explicit_Bin.pickle')
    eb_model_error_dwrpe.append(eb_model.errors['distance_weighted_relative_percent_difference'])
    eb_model_error_net.append(eb_model.errors['net_error'])
    
    rf_model = routee.read_model(PATH_to_models+basename_i+'_Random_Forest.pickle')
    rf_model_error_dwrpe.append(rf_model.errors['distance_weighted_relative_percent_difference'])
    rf_model_error_net.append(rf_model.errors['net_error'])
    

fig, ax = plt.subplots()
plt.scatter(root_list, eb_model_error_dwrpe, s=125, alpha=0.6, label='explicit bin')
plt.scatter(root_list, rf_model_error_dwrpe, s=125, alpha=0.6, label='random forest')
plt.xticks(rotation='vertical')
plt.ylim([0,2])
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.legend()
plt.title('Distance Weighted Relative Percent Error (per link)')
plt.savefig('plots/dwrpe.png', bbox_inches='tight')


fig, ax = plt.subplots()
plt.scatter(root_list, eb_model_error_net, s=125, alpha=0.6, label='explicit bin')
plt.scatter(root_list, rf_model_error_net, s=125, alpha=0.6, label='random forest')
plt.xticks(rotation='vertical')
# plt.ylim([0,2])
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.legend()
plt.title('Net Error')
plt.savefig('plots/net.png', bbox_inches='tight')