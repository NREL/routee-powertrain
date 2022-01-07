# utility functions for feature_selector module


def get_sfs_vals(search_method):
    selection_dict = {
        "sfs": [True, False],
        "sbe": [False, False],
        "fsfs": [True, True],
        "fsbe": [False, True],
    }
    # Sequential Forward Selection(sfs), Sequential Backward Elimination (sbe)
    forward_val = selection_dict[search_method][0]
    floating_val = selection_dict[search_method][1]
    return forward_val, floating_val
