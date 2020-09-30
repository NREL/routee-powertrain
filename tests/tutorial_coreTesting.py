'''
RouteE Core Testing
This notebook takes the complete TomTom dataset and runs routeE analysis (train + predict) on eagle. The following cases are investigated:

Analysis and preprocessing of the input data --> Zero data error for miles traveled, time traveled and fuel consumed
Feature importance characterization of the random forest regressor
Impact of feature inclusion/modification for the random forest regressor
Environment activation guide:

From putty-->create environment "routee" based on environment.yml
From putty--> conda activate routee
Install ipykernel--> conda install ipykernel python -m ipykernel install --user --name routee
Change Kernel (in the jupyter lab) to--> routee
Interactive Notebook/HPC:

From putty--> conda activate ~/envs/routee
sbatch --time=10 --account=aes4t notebook.sh (submit request)
squeue -u amahbub (check status of the request)
scancel -u amahbub (to cancel allocation)
vim notebook.out (read the .out file for plink)
from cmd, run plink link_from_above
from a browser, localhost: 8889
'''

# 1. Loading required modules
PICKLE_OUT_PATH = "data/"

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import powertrain

from powertrain.estimators import ExplicitBin, RandomForest, XGBoost
from powertrain.core.model import Feature, Distance, Energy
from powertrain.plot.feature_importance import plot_feature_importance

#################################################################
'''
2. Load FastSIM Data
RouteE models rely on high resolution (1 Hz) vehicle energy consumption data (either simulated or measured) in order to train the various estimators to predict energy consumption for much lower resolution data. Typically, this training data comes from FASTSim results for various powertrain models over TSDC drive cycles (1 Hz "points" data).

Based on the repo location, activate either of the following paths:
FASTSIM_RESULTS (local repo) = "C:/Users/amahbub/Documents/Data_RouteE/data_tomtom_new/2016_TOYOTA_Camry_4cyl_2WD_fs_results.csv"
FASTSIM_RESULTS (Eagle HPC repo) = "/projects/aes4t/jholden/data/fastsim_results/2016_TOYOTA_Camry_4cyl_2WD_fs_results.csv"
Provide the feature list --> Training + Target feature
Provide the # of samples to read. Provide "None" if all samples are to be read
'''


def get_df(nrows=None, feature_list=None):
    # FASTSIM_RESULTS = "C:/Users/amahbub/Documents/Data_RouteE/data_tomtom_new/2016_TOYOTA_Camry_4cyl_2WD_fs_results.csv"
    FASTSIM_RESULTS = "/projects/aes4t/jholden/data/fastsim_results/2016_TOYOTA_Camry_4cyl_2WD_fs_results.csv"
    df = pd.read_csv(FASTSIM_RESULTS, nrows=nrows, usecols=feature_list)
    return df


# user input: provide appropriate list of features
feature_list = ['gge', 'miles', 'meters', 'seconds', 'grade', 'frc', 'gpsspeed', 'minutes',
                'net2class', 'kph', 'speedcat', 'fow']
start_time = time.time()
df_passes = get_df(nrows=1000, feature_list=None)
end_time = time.time()
print(f"Function execution time: {end_time - start_time}.")

df_mod = df_passes[feature_list]


def remove_nan(df_passes, nan_threshold=1000000):
    columns = df_passes.columns
    nan_list = df_passes.isna().sum()  # get a df of nan values sums
    val = (
            nan_list > nan_threshold).values  # make a list of nan val greater than the threshold, i.e. inadmissible features --> 1M
    df_passes = df_passes.drop(columns=columns[val])  # column list defined above
    df_passes = df_passes.dropna()
    print(columns[val])
    return df_passes


df_mod = remove_nan(df_mod)

#################################################################
'''
3. Training Features
User input: Provide the feature names (str) and units (str) to be used to train the routeE model.

Note: The "Distance" named-tuple, which represents the length of each link, can be incorporated as a feature in two ways.

If option = 1 is selected, the 'distance' feature is not explicitely considered as a training feature. Rather, the target feature 'gge' is scaled as "energy per distance".
If option = 2 is selected, the 'distance' feature is explicitely considered as a training feature. In this case, the target feature 'gge' remains unaltered.
The parameter (1 or 2) of 'option' can be set before training individual models. See examples in later sections.
'''
features = [
    Feature('gpsspeed', units='mph'),
    Feature('grade', units='ratio'),
    Feature('seconds', units='seconds'),
    Feature('meters', units='meters'),
    Feature('minutes', units='minutes'),
    Feature('fow', units='fow'),
    Feature('speedcat', units='speed_cat'),
    Feature('frc', units='frc'),
    Feature('kph', units='kph'),
    Feature('net2class', units='net2class')]  # list of namedtouples

distance = Distance('miles', units='mi')
energy = Energy('gge', units='gallons')

##############################################################################
'''
4. Training Models
RouteE provides three different training capabilities.
'ExplicitBin' enables the binning of feature values to create a look-up table.
'RandomForest' enables the training of the model using scikit-learn's RandomForestRegressor.
'XGBoost' enables the training of the model using mlxtend's xgboost decision tree regressor.
The performance of the 'RandomForest' and 'XGBoost' decision tree models can be affected by setting 'option' values as stated previously. For example, the importance of features while training a decision tree regressor shows significant difference between the two 'option' settings.
'''

# Example 1: ExplicitBin
# creating estimator model
scenario_name = "2016_audi_A3"
option = 1  # or provide 2. Explicit bin module has no dependence on the 'option' setting

eb_model = powertrain.Model(scenario_name, option,
                        estimator=ExplicitBin(features=features, distance=distance, energy=energy))

# training model with data (also does validation with test data and calculates error metrics)
eb_model.train(df_mod, features=features, distance=distance, energy=energy)

print(eb_model.metadata)
print(eb_model.errors)

'''
Example 2: XGBoost module
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

User can view the stored metadata or error metrics by calling the class attributes.
To investigate the feature importance after the training, user can call the 'plot_feature_importance()' method to get a visual representation.
'''
# creating estimator model
scenario_name = "2016_audi_A3"
option = 2  # going for fc/dist option

xgb_model = powertrain.Model(scenario_name, option, estimator=XGBoost(cores=4))

# training model with data (also does validation with test data and calculates error metrics)
xgb_model.train(df_mod, features=features, distance=distance, energy=energy)

print(xgb_model.metadata)
print(xgb_model.errors)

plot_feature_importance(xgb_model)

'''
Example 3: RandomForest module
A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

User can view the stored metadata or error metrics by calling the class attributes.
To investigate the feature importance after the training, user can call the 'plot_feature_importance()' method to get a visual representation.
'''
# creating estimator model
scenario_name = "2016_audi_A3"
option = 2  # going for fc/dist option

rf_model = powertrain.Model(scenario_name, option, estimator=RandomForest(cores=4))
# training model with data (also does validation with test data and calculates error metrics)
rf_model.train(df_mod, features=features, distance=distance, energy=energy)
print(rf_model.errors)
plot_feature_importance(rf_model)

######################################################################################

'''
4. Prediction Comparison (TBD)
Here, we predict the energy consumption of a given test data and visualize the performance of different training modules used.

Note: The test data has to comply with all the features of the training model. Based on the complexity of the selected features, the test data can either be generated artificially, or a complete route containing the necessary features can be imported directly from an existing database. The details are as follows:

Imported test route: In the examples shown above, we have used a group of features that are complicated to be generated artifically. Therefore, we 'isolate' a test route from an existing FASTSIM database containing all the necessary features, and then use this test data to predict the performance. The methodology is described in Section 4.1.

Artificial test route: If the training features have relatively less complexity (for example grade, gpsspeed, frc etc), we can generate a test route artificially and test our training models. The methodology is described in Section 4.2
'''

'''
4.1: Isolated vehicle route
We define the functions to isolate any number of vehicle routes from an existing FASTSIM database.
'''


def isolate_route(df, routeNo=1):
    sample_list = df['sampno'].unique()
    df_route = []
    counter = 0
    for sample in sample_list:
        df_samp = df[df['sampno'] == sample]  # isolating sample
        veh_list = df_samp['vehno'].unique()
        for veh in veh_list:
            df_veh = df_samp[df_samp['vehno'] == veh]  # isolating vehicle
            trip_list = df_veh['tripno'].unique()
            for trip in trip_list:
                df_trip = df_veh[df_veh['tripno'] == trip]  # isolating trip
                df_trip = df_trip.sort_values(by=['time_local_start'])  # properly sorting the veh data by start time
                df_route.append(df_trip)
                counter += 1
                if counter == routeNo: return df_route


def get_isolated_df(df, routeNo=1):
    df_list = isolate_route(df, routeNo=routeNo)
    df_mod = pd.DataFrame()
    for i in df_list:
        # i.sort_values(by=['time_local_start']) #not properly sorting ****
        df_mod = pd.concat([df_mod, i])
    return df_mod


# We need the features 'tripno', 'vehno', 'sampno', 'time_local_start' to isolate the vehicle route. Note:We do not use these features to train the model itself.

route_features = ['tripno', 'vehno', 'sampno', 'time_local_start']
df_passes = df_passes[feature_list + route_features]

# Here, we isolate only 1 vehicle route by providing the parameter '1' in the isolate_route().
# selecting a random test sample
veh_data = isolate_route(df_passes, 1)
links_df = veh_data[0].dropna()
links_df['distance'] = links_df['miles'].cumsum()

# visualize the route/veh data
plt.plot(links_df['distance'], links_df['gpsspeed'])
plt.xlabel('distance [miles]')
plt.ylabel('speed [mph]')
plt.title('Test route')
plt.grid()

xgboost_output = xgb_model.predict(links_df)
rf_output = rf_model.predict(links_df)
baseline_output = eb_model.predict(links_df)

plt.plot(links_df['distance'],
         xgboost_output * 100 / links_df.miles,
         label='xgb'
         )

plt.plot(links_df['distance'],
         rf_output * 100 / links_df.miles,
         label='RF'
         )
plt.plot(links_df['distance'],
         links_df['gge'] * 100 / links_df.miles,
         label='FastSIM'
         )

plt.xlabel('Distance [miles]')
plt.ylabel('Gallons/100mi')
plt.grid()
plt.legend()


def plot_prediction(df, model_list, output_list, plot_parameters):
    for param in plot_parameters:
        for i in range(len(output_list)):
            plt.scatter(links_df[param],
                        output_list[i] * 100 / links_df.miles,
                        label='rf'  # model_list[i]
                        );
            plt.xlabel(param)
            plt.ylabel('Gallons/100mi')
        plt.grid()
        plt.show()


plot_features = ['gpsspeed', 'distance']
plot_prediction(links_df, ['xgb', 'rf', 'eb'], [xgboost_output, rf_output, baseline_output], plot_features)

'''
4.2: Artificial route:¶
Here, we artifically generate vehicle route information. Due to the limitation of information generation for different features (as listed previously), we take a reduced set of features. For example, in what follows, we generate a simple linearly increasing speed profile. To this end, we only generate the values for features 'gpsspeed' (avg. vehicle speed in the link), 'grade' (grade of each link) and 'miles' (length of each link).
'''
links_df = pd.DataFrame()
links_df['gpsspeed'] = np.linspace(2, 80, num=15)
links_df['grade'] = [0] * len(links_df)
links_df['miles'] = [.1] * len(links_df)
links_df['distance'] = links_df['miles'].cumsum()

# visualize data
plt.plot(links_df['distance'], links_df['gpsspeed'])
plt.xlabel('distance [miles]')
plt.ylabel('speed [mph]')
plt.grid()

# We provide the limited features according to the artificial route generated above.
features = [
    Feature('gpsspeed', units='mph'),
    Feature('grade', units='ratio')]  # list of namedtouples

distance = Distance('miles', units='mi')
energy = Energy('gge', units='gallons')

df_mod = df_passes[['gpsspeed', 'grade', 'gge', 'miles']]

# We train new models based on the reduced feature list.
option = 2
eb_model = powertrain.Model(scenario_name, option,
                        estimator=ExplicitBin(features=features, distance=distance, energy=energy))
rf_model = powertrain.Model(scenario_name, option, estimator=RandomForest(cores=4))
xgb_model = powertrain.Model(scenario_name, option, estimator=XGBoost(cores=4))

# training model with data (also does validation with test data and calculates error metrics)
eb_model.train(df_mod, features=features, distance=distance, energy=energy)
rf_model.train(df_mod, features=features, distance=distance, energy=energy)
xgb_model.train(df_mod, features=features, distance=distance, energy=energy)

# We predict the energy consumption by providing the artifically create route, and visualize the results.
xgboost_output = xgb_model.predict(links_df)
rf_output = rf_model.predict(links_df)
baseline_output = eb_model.predict(links_df)

plt.plot(links_df['distance'],
         xgboost_output * 100 / links_df.miles,
         label='xgb'
         )

plt.plot(links_df['distance'],
         rf_output * 100 / links_df.miles,
         label='RF'
         )

plt.xlabel('Distance [miles]')
plt.ylabel('Gallons/100mi')
plt.grid()
plt.legend()
