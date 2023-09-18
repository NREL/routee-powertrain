# Installation

To install the base package for model prediction, we recommend you use `pip`:

```bash
pip install nrel.routee.powertrain
```

To install the package for model training, you can still use pip and just specify which training dependencies you want to install. We currently have the following train pipelines available:

- `scikit`: scikit-learn based pipeline

```bash
pip install nrel.routee.powertrain[scikit]
```

## From Source

To install the package from source, you can clone the repository and install the package using `pip`:

```bash
git clone https://github.nrel.gov/MBAP/routee-powertrain.git
cd routee-powertrain
pip install .
```
