import argparse

from helpers import DATASETS, descriptor_activation_split, is_valid_dataset, \
    logger, read
from neural_net import generate_model

DATADIR = "/data/preprocessed/"
TRAIN_FILE_SUFFIX = "_test_disguised.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train QSAR data')
    parser.add_argument('--dataset', help='Enter one of available datasets: {}'
                        .format(", ".join(DATASETS)), required=True)
    parser.add_argument('--epochs', help='Enter Number of training cycles',
                        required=True,
                        type=int)
    parser.add_argument('--batch_size', help='the number of training examples \
                        in one forward/backward pass',
                        required=True,
                        type=int)

    args = parser.parse_args()
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size

    is_valid_dataset(dataset)

    logger.info("Reading in preprocessed training {0} dataset".format(dataset))
    file_path = "{dir}{dataset}{suffix}".format(dir=DATADIR, dataset=dataset,
                                                suffix=TRAIN_FILE_SUFFIX)
    train_df = read(file_path)
    logger.info("Finished reading in preprocessed {0} dataset".format(dataset))

    logger.info("preparing training data")
    split = descriptor_activation_split(train_df)

    logger.info("Generating model")
    model = generate_model(split.shape)

    logger.info("fitting model")
    model.fit(split.descriptors, split.act,
              epochs=epochs,
              batch_size=batch_size)

    logger.info("saving model")
    model.save('{0}{1}.h5'.format('/data/', dataset))

    logger.info("Success !")
