# txtcat-nb

## Description:
txtcat-nb is a simple text file classifier using naive bayes.

## Data
Training data is assumed to be in a single directory; subdirectories within the data directory used for labels.

For example,

> >ls data/
> data/a data/b
> >ls data/*
> data/a/1.txt data/a/2.txt data/b/3.txt data/b/4.txt

would correspond to a two class dataset with labels `a` and `b`, each with two data files.

## Requirements:
python >= 2.7 (for collections.Counter and json).  Also works with python3 (tested with 3.1.2).

## Usage:
### Training usage:
For training data as above, storing trained classifier in `model.json`, using 30% of training files for each label for statistics:
> train.py -d data/ -o model.json -p 0.3

or
> train.py --datadir=data/ --output=model.json --pct=0.3

### Classification usage:
> classify.py -m model.json -d unlabled-data/

or
> classify.py --model=model.json --datadir=unlabled-data/

## Known Issues:
 1. Doesn't currently support full cross-validation in training.
 2. No pre-processing of words yet - accuracy for my current test cases is sufficient with no preprocessing.

