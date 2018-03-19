# QSAR

The pharmaceutical industry relies on Quantitative structure−activity relationships (QSAR) models to predict a quantified biological response of a molecule based on its descriptors, which are essentially studied properties of the molecule. These descriptors vary in complexity and can range from simple molecular weight measures to complex geometric features. Drug discovery is a time-consuming and expensive process for pharma. A major purpose of these QSAR models is to help accelerate discovery of molecular drug candidates through reduced experimental work, and eventually bring a drug to market faster. Due to recent advances in Machine Learning and hardware capabilities, Deep Neural Networks (DNNs) serve as a promising tool to predict biological activity, such as receptor binding or enzyme inhibition, based on molecular descriptors.

This project implements a DNN based on the architecture and parameters described in the following paper:

`Ma, J., Sheridan, R.P., Liaw, A., Dahl, G.E. and Svetnik, V., 2015. Deep neural nets as a method for quantitative structure–activity relationships. Journal of chemical information and modeling, 55(2), pp.263-274.`

## Getting Started

The data used for training and evaluation of the model is can be downloaded from the paper's supplementary section.

[Paper's supplementary page](https://pubs.acs.org/doi/suppl/10.1021/ci500747n)

Both the training and test data are structured in such a way that each row represents a molecule. There is a single column called "Act" that represents the biological activity that is to be predicted. The rest of the columns are molecular descriptors.

### Prerequisites

* Docker
* Pipenv


[Docker Installation Documentation](https://docs.docker.com/install/#desktop)

[Pipenv Installation Documentation](https://docs.pipenv.org/)

### Installing
* Install dependencies via Pipenv
* Build Docker image based on Dockerfile

```
make build
```

### Preparing the Data for Training
Specify the dataset of interest and its location. For example:

```
make preprocess DATASET=NK1 DATA=~/Documents/qsar/
```


### Training the Model

Specify the dataset of interest and its location and override the batch size and number of epochs specified in the Makefile.

```
make train DATASET=NK1 DATA=~/Documents/qsar/ BATCH_SIZE=64 EPOCHS=128
```

### Evaluating the Model

Specify the dataset of interest and its location. For example:

```
make evaluate DATASET=NK1 DATA=~/Documents/qsar/
```

### Testing

Pytest

```
make test
```
### Linting

Flake8 is the chosen linter

```
make lint
```

## Acknowledgments

Thank you to Ma, J et al. for clear description of DNN architecture and supplementary data

### Future Work
* NVIDIA Docker image for GPU based Training
* Error handling if weights aren't available for a dataset
* Tests around the Preprocessor
