from collections import namedtuple
import logging
import os
import sys

import pandas as pd

DATASETS = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1',
            'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']
INDEX_LABEL = "MOLECULE"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Split = namedtuple('Split', ['act', 'descriptors', 'shape'])


def read(file_path):
    try:
        df = pd.read_csv(file_path)
        df.set_index(INDEX_LABEL, inplace=True)
        return df
    except FileNotFoundError:
        sys.exit("Unable to find file", exc_info=True)


def write(file_dir, file_name, df):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    df.to_csv("{file_dir}{file_name}".format(
            file_dir=file_dir,
            file_name=file_name
        )
    )


def descriptor_activation_split(df):
    act = df.loc[:, 'Act'].values
    descriptors = df.loc[:, df.columns != 'Act']
    shape = (descriptors.shape[1],)
    descriptors = descriptors.as_matrix()
    return Split(act=act, descriptors=descriptors, shape=shape)


def is_valid_dataset(dataset):
    if dataset not in DATASETS:
        raise ValueError("The specified dataset is not valid. The following "
                         "are valid values: {0}".format(
                            ", ".join(DATASETS)))
        return True
