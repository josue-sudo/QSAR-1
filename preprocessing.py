import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATASETS = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1',
            'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']


class Preprocessor(object):
    def __init__(self, dataset, root):
        self.dataset = dataset
        self.root = root
        self.test_file = "{}{}_test_disguised.csv".format(root, dataset)
        self.train_file = "{}{}_training_disguised.csv".format(root, dataset)

        def transform(self):
            try:
                train = pd.read_csv(self.train_file)
                test = pd.read_csv(self.test_file)
            except FileNotFoundError:
                logger.error("Unable to find file", exc_info=True)
                sys.exit()

            train, test = self._remove_noncommon_descriptors(train, test)
            train, test = self._normalize_activations(train, test)
            train = self._rescale_descriptors(train)
            test = self._rescale_descriptors(test)
            return train, test

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
            train['Act'] = (train['Act'] - mean) / std
            test['Act'] = (test['Act'] - mean) / std
            return train, test

        @staticmethod
        def _rescale_descriptors(df):
            df.ix[:, 2:] = np.log(df.ix[:, 2:] + 1)
            return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess QSAR data')
    parser.add_argument('--dataset', help='Enter one of available datasets: {}'
                        .format(", ".join(DATASETS)), required=True)
    parser.add_argument('--file_path', help='Specify path that contains '
                        'dataset files', required=True)

    args = parser.parse_args()
    dataset = args.dataset
    file_path = "{}/".format(args.file_path) if not \
        args.file_path.endswith("/") else args.file_path
    if dataset not in DATASETS:
        raise ValueError("The specified dataset is not valid. The following "
                         "are valid values: {}".format(", ".join(DATASETS)))

    logger.info("Transforming {} dataset".format(dataset))
    start = time.time()
    train, test = Preprocessor(dataset, file_path).transform()
    logger.info("Transformation took {} seconds".format(time.time() - start))

    train.to_csv("{}preprocessed/{}_training_disguised.csv").format(
        file_path, dataset
    )
    test.to_csv("{}preprocessed/{}_test_disguised.csv").format(
        file_path, dataset
    )
    logger.info("Write operations complete")
