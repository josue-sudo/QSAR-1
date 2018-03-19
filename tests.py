import pandas as pd

from helpers import descriptor_activation_split


def test_descriptor_activation_split():
    data = {'Act': [1, 2, 3], 'D_1781': [1, 1, 2], 'D_2075': [5, 6, 7]}
    df = pd.DataFrame(data=data)
    split = descriptor_activation_split(df)
    print(df.loc[:, 'Act'].values)
    assert split.act.tolist() == [1, 2, 3]
    assert split.descriptors.tolist() == [[1, 5], [1, 6], [2, 7]]
    assert split.shape == (2, )
