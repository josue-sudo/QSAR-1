import argparse

from keras.models import load_model

from helpers import DATASETS, descriptor_activation_split, is_valid_dataset, \
    logger, read
from neural_net import r2


DATADIR = "/data/preprocessed/"
TEST_FILE_SUFFIX = "_training_disguised.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate QSAR data')
    parser.add_argument('--dataset', help='Enter one of available datasets: {}'
                        .format(", ".join(DATASETS)), required=True)

    args = parser.parse_args()
    dataset = args.dataset

    is_valid_dataset(dataset)

    file_path = "{dir}{dataset}{suffix}".format(dir=DATADIR, dataset=dataset,
                                                suffix=TEST_FILE_SUFFIX)
    logger.info("Reading in preprocessed testing {0} dataset".format(dataset))
    test_df = read(file_path)
    logger.info("Finished reading in preprocessed {0} dataset".format(dataset))

    logger.info("Preparing testing data")
    split = descriptor_activation_split(test_df)

    logger.info("Loading {0} model".format(dataset))
    model = load_model("./weights/{0}.h5".format(dataset),
                       custom_objects={'r2': r2})

    logger.info("Evaluating {0} model".format(dataset))
    score = model.evaluate(split.descriptors, split.act)

    logger.info("R2 score {0}".format(score[1]))
