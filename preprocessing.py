import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1',
            'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']
DATADIR = "/data/"
INDEX_LABEL = "MOLECULE"
TRAIN_FILE_SUFFIX = "_test_disguised.csv"
TEST_FILE_SUFFIX = "_training_disguised.csv"


class Preprocessor(object):
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

    def transform(self):
        train, test = self._remove_noncommon_descriptors(self.train_df,
                                                         self.test_df)
        train, test = self._normalize_activations(train, test)
        self.train_df, self.test_df = self._rescale_descriptors(train, test)
        return self.train_df, self.test_df

    @staticmethod
    def _remove_noncommon_descriptors(train, test):
        train_descriptors = set(train.columns.values)
        test_descriptors = set(test.columns.values)
        common_descriptors = set.intersection(train_descriptors,
                                              test_descriptors)
        train_descriptors = [td for td in train_descriptors if td in
                             common_descriptors]
        test_descriptors = [td for td in test_descriptors if td in
                            common_descriptors]
        return train[train_descriptors], test[test_descriptors]

    @staticmethod
    def _normalize_activations(train, test):
        mean = train['Act'].mean()
        std = train['Act'].std()
        train['Act'].apply(lambda x: (x - mean) / std)
        test['Act'].apply(lambda x: (x - mean) / std)
        return train, test

    @staticmethod
    def _rescale_descriptors(train, test):
        train.loc[:, train.columns != 'Act'].apply(lambda x: np.log(x + 1))
        test.loc[:, test.columns != 'Act'].apply(lambda x: np.log(x + 1))
        return train, test


class FileHelper(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.test_file = "{0}{1}{2}".format(DATADIR, dataset,
                                            TEST_FILE_SUFFIX)
        self.train_file = "{0}{1}{2}".format(DATADIR, dataset,
                                             TRAIN_FILE_SUFFIX)
        self.write_destination = "{0}preprocessed/".format(DATADIR)

    def read(self):
        try:
            train_df = pd.read_csv(self.train_file)
            train_df.set_index(INDEX_LABEL, inplace=True)
            test_df = pd.read_csv(self.test_file)
            test_df.set_index(INDEX_LABEL, inplace=True)
            return train_df, test_df
        except FileNotFoundError:
            logger.error("Unable to find file", exc_info=True)
            sys.exit()

    def write(self, df, suffix):
        if not os.path.exists(self.write_destination):
            os.makedirs(self.write_destination)
        df.to_csv("{0}{1}{2}".format(
                self.write_destination, self.dataset, suffix
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess QSAR data')
    parser.add_argument('--dataset', help='Enter one of available datasets: {}'
                        .format(", ".join(DATASETS)), required=True)

    args = parser.parse_args()
    dataset = args.dataset
    if dataset not in DATASETS:
        raise ValueError("The specified dataset is not valid. The following "
                         "are valid values: {0}".format(", ".join(DATASETS)))

    # Read data
    logger.info("Reading in {0} dataset".format(dataset))
    file_helper = FileHelper(dataset)
    train_df, test_df = file_helper.read()
    logger.info("Finished reading in {0} dataset".format(dataset))

    # Transform
    logger.info("Transforming {0} dataset".format(dataset))
    start = time.time()
    train_df, test_df = Preprocessor(train_df, test_df).transform()
    logger.info("Transformation took {0} seconds".format(
        time.time() - start))

    # Write operations
    logger.info("Writing preprocessed {0} dataset to disk".format(dataset))
    file_helper.write(train_df, TRAIN_FILE_SUFFIX)
    file_helper.write(test_df, TEST_FILE_SUFFIX)
    logger.info("Finished writing preprocessed {0} dataset ".format(dataset))
