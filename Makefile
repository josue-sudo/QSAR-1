DATA?="${HOME}/Data"
DATASET=""
SRC=$(shell pwd)

BATCH_SIZE=64
EPOCHS=2

build:
				pipenv install
				docker build -t qras -f Dockerfile .

preprocess:
				docker run -it --rm -v $(SRC):/src -v $(DATA):/data qras python3 preprocessing.py --dataset=$(DATASET)

train:
				docker run -it --rm -v $(SRC):/src -v $(DATA):/data qras python3 train.py --dataset=$(DATASET) --batch_size=$(BATCH_SIZE) --epochs=$(EPOCHS)

evaluate:
				docker run -it --rm -v $(SRC):/src -v $(DATA):/data qras python3 evaluate.py --dataset=$(DATASET)

lint:
				flake8

test:
				pytest tests.py
