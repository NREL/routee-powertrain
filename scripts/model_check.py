import pandas as pd
import sys
import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt

from powertrain.core.model import Model
from pathlib import Path

PATH_to_models = "/projects/mbap/data/routee_results/2020_11_9/"
PATH_to_models = Path(PATH_to_models)

eb_model_list = [
    x for x in os.listdir(PATH_to_models) if x.endswith("ExplicitBin.json")
]
rf_model_list = [
    x for x in os.listdir(PATH_to_models) if x.endswith("RandomForest.json")
]

root_list = [x.split("_ExplicitBin")[0] for x in eb_model_list]

eb_model_error_dwrpe = []
rf_model_error_dwrpe = []

eb_model_error_net = []
rf_model_error_net = []

for basename_i in root_list:
    json_file = PATH_to_models.joinpath(basename_i + "_ExplicitBin.json")
    eb_model = Model.from_json(json_file)
    eb_model_error_dwrpe.append(
        eb_model.metadata["errors"]["distance_weighted_relative_percent_difference"]
    )
    eb_model_error_net.append(eb_model.metadata["errors"]["net_error"])

    json_file = PATH_to_models.joinpath(basename_i + "_RandomForest.json")
    rf_model = Model.from_json(json_file)
    rf_model_error_dwrpe.append(
        rf_model.metadata["errors"]["distance_weighted_relative_percent_difference"]
    )
    rf_model_error_net.append(rf_model.metadata["errors"]["net_error"])


fig, ax = plt.subplots()
plt.scatter(root_list, eb_model_error_dwrpe, s=125, alpha=0.6, label="explicit bin")
plt.scatter(root_list, rf_model_error_dwrpe, s=125, alpha=0.6, label="random forest")
plt.xticks(rotation="vertical")
plt.ylim([0, 2])
vals = ax.get_yticks()
ax.set_yticklabels(["{:,.1%}".format(x) for x in vals])
plt.legend()
plt.title("Distance Weighted Relative Percent Error (per link)")
plt.savefig("plots/dwrpe.png", bbox_inches="tight")


fig, ax = plt.subplots()
plt.scatter(root_list, eb_model_error_net, s=125, alpha=0.6, label="explicit bin")
plt.scatter(root_list, rf_model_error_net, s=125, alpha=0.6, label="random forest")
plt.xticks(rotation="vertical")
# plt.ylim([0,2])
vals = ax.get_yticks()
ax.set_yticklabels(["{:,.1%}".format(x) for x in vals])
plt.legend()
plt.title("Net Error")
plt.savefig("plots/net.png", bbox_inches="tight")
