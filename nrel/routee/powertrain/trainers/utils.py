import numpy as np
import pandas as pd

# we're pinning the onnx opset to 13 since the rust onnxruntime crate
# is built from onnx runtime version 1.8 which only supports opset 13
# see here: https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
ONNX_OPSET_VERSION = 13

def test_train_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 123):
    """
    Split a dataframe into training and testing sets
    """
    np.random.seed(seed)
    mask = np.random.rand(len(df)) < (1 - test_size)
    train_df = df[mask]
    test_df = df[~mask]
    return train_df, test_df
