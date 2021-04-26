## training routee-powertrain models

Workflow for training a batch of routee-powertrain models

### setup

Setup for routee-powertrain batch training takes a few steps:

1. Create a new conda environment and install routee-powertrain:
```
conda create -n routee-powertrain python=3.8
conda activate routee-powertrain
git clone https://github.nrel.gov/MBAP/routee-powertrain.git
cd routee-powertrain
pip install -e .
```

3. Copy over the default config to your own file:
```
cd routee-powertrain/scripts/batch-train
cp default.config.yml config.yml
```
	
### input files

To train a set of routee-powertrain models, we'll need to modify the config.yml file to point to 
a set of training data and adjust the features and targets. 

To demonstrate, we'll use a database of fastsim models that have been driven over all of the 
trips in the TSDC on the eagle computer.

Let's take a look at the whole config file:

```yaml
# config for training a batch of models:

# where to find the training data;
# the script will look for all .db files that live in this folder
training_data_path: '/projects/mbap/data/fastsim_results/routee-train-data/2021-02-19_11-49-23/'

# where to write the resulting routee-powertrain models
output_path: '/projects/mbap/data/routee_results/'

# define the energy targets to use; the training database should indicate which energy type to use;
energy_targets:
- energy_type: electric
  name: esskwhoutach
  units: kwh
- energy_type: gasoline
  name: gge
  units: gallons

# define the distance feature to use
distance:
  name: miles
  units: miles

# define any other features to use for training the models
features:
- name: gpsspeed
  units: mph
- name: grade
  units: percent_0_100

# which estimators should be used: [explicit_bin, random_forest, linear_regression, xgboost]

estimators:
- explicit_bin
- random_forest

# how many cores to use
n_cores: 28 

# what format to write the routee-powertrain model: [json, pickle]
model_output_type: json
```


Note that we've set the names energy targets, distance features, and other features to match the names that the fastsim script outputs. 
If you're using custom data, these should match your own training data. 

### running

now that everything is setup, we can run the batch trainer script with either:

    > python batch-trainer.py config.yml

or, on eagle:

    > sbatch batch-trainer-slurm.sh

That's it! If (when) you encounter any problems please open an issue in this repository.