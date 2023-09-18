# RouteE BEAM integration docs

## Getting started
RouteE was designed to be used as a Python package that allows users to work 
with a set of pre-trained energy prediction models for a varity of vehicle 
types. Additionally, users can train their own models if "ground truth" energy
consumption and driving data are available.

Integration of the pre-trained energy prediciton models into the Java-based
BEAM platform can be done through directly working with the trained model files.
Trained models can be downloaded [here](https://nrel.box.com/s/itdni7vtoqflnehupnc9e7961365p5t7). Once downloaded, 
contents should be placed in ```../routee/models/trained_catalog/```.

## Model Types
There are multiple energy prediction model methods. In Python implementations,
the model objects can be read and in addition to the energy model itself, the 
objects also contain metadata about the validation performance of the 
model. In porting the model catalog over for Java compatibility (can be found in ```../routee/models/trained_catalog/java```), that metadata
is lost, so users must reference back to the Python objects in ```../routee/models/trained_catalog/python``` if desired.

### Explicit Bin
Currently, RouteE has two methods for training prediction models. The first is 
the "explicit bin" approach, where the driving data is partitioned by condition
and energy consumption rates are determined for each "bin" to generate a lookup
table.

Column headers that describe features in the lookup table are in the format 
"[feature]_[unit]_[datatype]_bins". The energy consumption rate header format is
"rate_[energy unit]_per_100[distance unit]". This table is applied at the link
level in the road network (see ```notebooks/demo_pretrained_calculation.ipynb``` for
a Pythonic demonstration).

### Random Forest
The second prediction method is a trained random forest regressor. This is a 
scikit learn object that has been pickled in the ```../routee/models/trained_catalog/java``` directory. __In order to complete 
the conversion to Java compatibility, users must run the [JPMML sklearn](https://github.com/jpmml/jpmml-sklearn) commandline tool to convert the models
to PMML docs.__ Given the BEAM team's expertise with Java, NREL has left this 
last conversion step to them due to hurdles in getting it working.

Required features for the random forest regressor are: ['speed_mph_float','grade_percent_float','num_lanes_int'].
