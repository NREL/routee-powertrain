import numpy as np

def test_train_split(df, test_perc):
    np.random.seed = 123
    msk = np.random.rand(len(df)) < (1-test_perc)
    train = df[msk]
    test = df[~msk]
    return train, test
