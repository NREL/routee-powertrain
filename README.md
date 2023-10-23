# <img src="docs/source/images/routeelogo.png" alt="drawing" width="100"/>

## Overview

The typical user will utilize RouteE's catalog of pre-trained models. Currently, the
catalog consists of light-duty vehicle models, including conventional gasoline, diesel, plug-in hybrid electric (PHEV),
hybrid electric (HEV), and battery electric (BEV). These models can be applied to link-level driving data (in the form
of [pandas](https://pandas.pydata.org/) dataframes) to output energy consumption predictions.

Users that wish to train new RouteE models can do so. The model training function of RouteE enables users to use their
own drive-cycle data, powertrain modeling system, and road network data to train custom models.

## Setup

Clone (or download) the RouteE Powertrain repository and create a compatible python environment to ensure package compatibility.

`git clone https://github.nrel.gov/MBAP/routee-powertrain.git`

routee-powertrain depends on python 3.8 and up. One way to satisfy this is to use [conda](https://conda.io/docs/):

```console
conda create -n routee-powertrain python=3.10
conda activate routee-powertrain
```

This will create a new conda environment that uses python 3.10. Navigate to the the routee-powertrain root directory. Then:

```console
pip install .
```

You will now be able to import routee-powertrain in your code with:

```console
import nrel.routee.powertrain as pt
```

## Quickstart

RouteE Powertrain is available on PyPI and can be installed with `pip`:

```bash
pip install nrel.routee.powertrain
```

Then, you can import the package and use a pre-trained model from the RouteE model catalog:

```python
import pandas as pd
import nrel.routee.powertrain as pt

# Load a pre-trained model
model = pt.load_pretrained_model("2016_TOYOTA_Camry_4cyl_2WD")

# Predict energy consumption for a set of road links
links_df = pd.DataFrame(
    {
        "distance": [0.1, 0.2, 0.3], # miles
        "speed": [30, 40, 50], # mph
        "grade": [-0.05, 0, 0.05], # decimal 
    }
)

energy_result = model.predict(links_df)
```

