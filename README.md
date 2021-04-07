# RouteE Powertrain

## Setup
Clone (or download) the RouteE Powertrain repository and create a compatible python environment to ensure package compatibility.

`git clone https://github.nrel.gov/MBAP/routee-powertrain.git`


routee-powertrain depends on python 3.8. One way to satisfy this is to use [conda](https://conda.io/docs/):
```
conda create -n routee-powertrain python=3.8 
conda activate routee-powertrain
```
This will create a new conda environment that uses python 3.8.

Navigate to the the routee-powertrain root directory. Then: 

```
pip install .
```

You will now be able to import routee-powertrain in your code with:
```
import powertrain
```

## Pre-Trained Models
RouteE-Powertrain comes prepackaged with a few standard models and can access a large library of pretrained models.

To see which models are available you can use the function `list_available_models` and then you can load any model
with the function `load_pretrained_model`. 

Here's a sample workflow:

```python
import powertrain as pt

# determine which models are available
model_names = pt.list_available_models()
for name in model_names:
  print(name)

leaf = pt.load_pretrained_model("2016_Nissan_Leaf_30_kWh_ExplicitBin")
```

## Test Data
If you are developing on the routee-powertrain projects and plan to run any of the tests, you will need to also download the test data from [Box](https://app.box.com/s/dm5w4mo56ej9jfmyo404kz98roz7jat7). 

It is recommended that you move the downloaded and unzipped "routee-powertrain-test-data" directory into ```powertrain/tests/```.

## Getting Started

RouteE has two core functions: model training and energy prediction. The below image highlights RouteE's capabilities: 
<p align="center">
  <img src="docs/wiki_imgs/routee_workflow.jpg" width="50%" height="50%">
</p>


The typical user will utilize RouteE's catalog of pre-trained models. Currently, the 
catalog consists of light-duty vehicle models, including conventional gasoline, diesel, plug-in hybrid electric (PHEV), 
hybrid electric (HEV), and battery electric (BEV). These models can be applied to link-level driving data (in the form 
of [pandas](https://pandas.pydata.org/) dataframes) to output energy consumption predictions. Such predictions can be 
applied to single-vehicle routes or network-wide traffic data.

Users that wish to train new RouteE models can do so. The model training function of RouteE enables users to use their 
own drive-cycle data, powertrain modeling system, and road network data to train custom models. 

This [routee-notebooks](https://github.nrel.gov/MBAP/routee-notebooks) repo provides demonstrations notebooks for 
examples of RouteE usage. 

## Training Workflow
Training batches of RouteE models from FASTSim results can be performed with the [batch-train script](https://github.nrel.gov/MBAP/routee-powertrain/tree/master/scripts/batch-train) 

## License
RouteE Copyright ©2019   Alliance for Sustainable Energy, LLC All Rights Reserved
 
This computer software was produced by Alliance for Sustainable Energy, LLC under Contract No. DE-AC36-08GO28308 with the U.S. Department of Energy. For 5 years from the date permission to assert copyright was obtained, the Government is granted for itself and others acting on its behalf a non-exclusive, paid-up, irrevocable worldwide license in this software to reproduce, prepare derivative works, and perform publicly and display publicly, by or on behalf of the Government. There is provision for the possible extension of the term of this license.
Subsequent to that period or any extension granted, the Government is granted for itself and others acting on its behalf a non-exclusive, paid-up, irrevocable worldwide license in this software to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. The specific term of the license can be identified by inquiry made to Alliance for Sustainable Energy, LLC or DOE. NEITHER ALLIANCE FOR SUSTAINABLE ENERGY, LLC, THE UNITED STATES NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY DATA, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
