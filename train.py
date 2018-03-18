import argparse
import logging
import sys

import pandas as pd

from neural_net import load_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1',
            'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']
DATADIR = "/data/preprocessed/"
INDEX_LABEL = "MOLECULE"
TRAIN_FILE_SUFFIX = "_test_disguised.csv"
TEST_FILE_SUFFIX = "_training_disguised.csv"
WEIGHTS_DIR = "/data/weights/"


class FileHelper(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.test_file = "{0}{1}{2}".format(DATADIR, dataset,
                                            TEST_FILE_SUFFIX)
        self.train_file = "{0}{1}{2}".format(DATADIR, dataset,
                                             TRAIN_FILE_SUFFIX)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess QSAR data')
    parser.add_argument('--dataset', help='Enter one of available datasets: {}'
                        .format(", ".join(DATASETS)), required=True)
    parser.add_argument('--epochs', help='Enter Number of training cycles',
                        required=True)
    parser.add_argument('--batch_size', help='the number of training examples \
                        in one forward/backward pass',
                        required=True)

    args = parser.parse_args()
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size

    if dataset not in DATASETS:
        raise ValueError("The specified dataset is not valid. The following "
                         "are valid values: {0}".format(
                            ", ".join(DATASETS)))

    # Read data
    logger.info("Reading in preprocessed {0} dataset".format(dataset))
    file_helper = FileHelper(dataset)
    train_df, _ = file_helper.read()
    logger.info("Finished reading in preprocessed {0} dataset".format(dataset))

    # Prepare training data
    logger.info("preparing training data")
    train_act = train_df.loc[:, 'Act'].values
    train_descriptors = train_df.loc[:, train_df.columns != 'Act']
    input_shape = (train_descriptors.shape[1],)
    train_descriptors = train_descriptors.as_matrix()

    logger.info("loading model")
    model = load_model(input_shape)

    logger.info("fitting model")
    model.fit(train_descriptors, train_act,
              epochs=epochs,
              batch_size=batch_size)

    logger.info("saving model")
    model.save('{0}{1}.h5'.format(WEIGHTS_DIR, dataset))
