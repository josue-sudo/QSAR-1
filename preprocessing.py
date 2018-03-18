import argparse
import time

import numpy as np

from helpers import DATASETS, is_valid_dataset, logger, read, write

DATADIR = "/data/"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess QSAR data')
    parser.add_argument('--dataset', help='Enter one of available datasets: {}'
                        .format(", ".join(DATASETS)), required=True)

    args = parser.parse_args()
    dataset = args.dataset

    is_valid_dataset(dataset)

    logger.info("Reading in {0} dataset".format(dataset))
    train_file_path = "{dir}{dataset}{suffix}".format(dir=DATADIR,
                                                      dataset=dataset,
                                                      suffix=TRAIN_FILE_SUFFIX)
    test_file_path = "{dir}{dataset}{suffix}".format(dir=DATADIR,
                                                     dataset=dataset,
                                                     suffix=TEST_FILE_SUFFIX)
    train_df = read(train_file_path)
    test_df = read(test_file_path)
    logger.info("Finished reading in {0} dataset".format(dataset))

    logger.info("Transforming {0} dataset".format(dataset))
    start = time.time()
    train_df, test_df = Preprocessor(train_df, test_df).transform()
    logger.info("Transformation took {0} seconds".format(
        time.time() - start))

    logger.info("Writing preprocessed {0} dataset to disk".format(dataset))
    write(DATADIR, dataset, train_df)
    write(DATADIR, dataset, test_df)
    logger.info("Finished writing preprocessed {0} dataset ".format(dataset))
