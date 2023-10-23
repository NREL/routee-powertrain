# RouteE Powertrain

> TODO: Add a better description of the project here.

RouteE Powertrain is a Python package that allows users to work with a set of pre-trained energy prediction models for a varity of vehicle types. Additionally, users can train their own models if "ground truth" energy consumption and driving data are available.

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
