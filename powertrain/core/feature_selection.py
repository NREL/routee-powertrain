from powertrain.feature_engineering import feature_selector


class Feature_Selection:
    """This is the preprocessing module for feature engineering.

    Args:
        df:
           Pandas dataframe containing the sample data.
        features:
            List of all relevant features --> Training + target features.

    """

    def __init__(self, df, features):
        self.df = df
        self.features = features

    def get_correlation(
        self,
        method=None,
        target_feature=None,
        multicolinearity=False,
        show_correlation=False,
        threshold=0.4,
    ):
        corr_obj = feature_selector.FeatureCorrelation(
            self.df,
            correlation_features=self.features,
            method=method,
            threshold=threshold,
        )
        if multicolinearity is not False:
            corr_obj.get_multicolinearity()
        if show_correlation is not False:
            corr_obj.show_corr()

    def get_n_features(
        self,
        target_feature=None,
        n_features=2,
        estimator="RFRegressor",
        search_method=None,
    ):
        """
        1. Args:
            - target_feature [str] --> The target feature name.
            - n_features [int] --> Number of features to be analysed. E.g., the user may want to find out the "best 3 features" to train the model.
            - estimator [str] --> Estimator name, e.g., "RFRegressor" for RandomForestRegressor() training estimator
            - search_method [str] --> The search method to be employed for the sequential feature search. Currently, there are six options.
                   1. sfs: Sequential forward selection
                   2. fsfs: Floating sequential forward selection
                   3. sbe: Sequential backward elimination
                   4. fsbe: Floating sequential backward elimination
                   5. rfe: Recursive feature elimination
                   6. all: Runs all of the above methods
        2. Returns:
            - prints a set of best "n" number of features.
            - prints the corresponding CV score.
            - shows a plot of the recursive search procedure (except 'rfe' method).
        """

        n_feature_obj = feature_selector.N_FeatureSearch(
            self.df,
            target_feature=target_feature,
            n_features=n_features,
            search_method=search_method,
            training_model=estimator,
        )
        if search_method == "all":
            n_feature_obj.all_sfs()
        elif search_method == "rfe":
            n_feature_obj.sfs_rfe()
        else:
            n_feature_obj.sequential_feature_search()

    def get_optimal_features(
        self, target_feature=None, estimator="RFRegressor", search_method=None
    ):
        """
        1. Args:
            - target_feature [str] --> The target feature name.
            - estimator [str] --> Estimator name, e.g., "RFRegressor" for RandomForestRegressor() training estimator
            - search_method [str] --> The search method to be employed for the sequential feature search. Currently, there are six options.
                   1. sfs: Sequential forward selection
                   2. fsfs: Floating sequential forward selection
                   3. sbe: Sequential backward elimination
                   4. fsbe: Floating sequential backward elimination
                   5. rfe: Recursive feature elimination
                   6. all: Runs all of the above methods
        2. Returns:
            - prints a set of optimal "n" number of features.
            - prints the corresponding CV score.
            - shows a plot of the recursive search procedure (except 'rfe' method).
        """
        opt_feature_obj = feature_selector.Optimal_FeatureSearch(
            self.df,
            target_feature=target_feature,
            search_method=search_method,
            training_model=estimator,
        )
        if search_method == "rfe":
            opt_feature_obj.optimal_feature_selector_RFE()
        else:
            opt_feature_obj.optimal_feature_selector_SFS()
