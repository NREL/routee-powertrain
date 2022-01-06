from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from powertrain.estimators.random_forest import RandomForest
from powertrain.estimators.xgboost import XGBoost


def plot_feature_importance(estimator):
    features = [feat.name for feat in estimator.metadata["features"]]

    if estimator.option == 2:
        features.append(estimator.metadata["distance"].name)

    plot_model = deepcopy(estimator.model)
    plot_model.feature_names = features

    if isinstance(estimator, XGBoost):
        xgb.plot_importance(plot_model)
        plt.rcParams["figure.figsize"] = [5, 5]
        plt.show()

        pd.Series(plot_model.feature_importances_, index=features).nlargest(10).plot(
            kind="barh"
        )

    elif isinstance(estimator, RandomForest):
        plt.barh(plot_model.feature_names, plot_model.feature_importances_)
        plt.xlabel("Importance [0~1]")
        plt.show()
    else:
        raise TypeError(
            "Can only plot feature importance for XGBoost and RandomForest estimators"
        )
