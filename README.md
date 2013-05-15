# txtcat-nb

## Description:
txtcat-nb is a simple text file classifier using naive bayes.

## Data
Training data is assumed to be in a single directory; subdirectories within the data directory used for labels.

For example,

> >ls data/
> data/a data/b
> >ls data/*
> data/a/1.txt data/a/2.txt data/b/3.txt data/4.txt

would correspond to a two class dataset with labels `a` and `b`, each with two data files.

## Requirements:
python >= 2.7 (for collections.Counter and json).

## Usage:
TODO

## Known Issues:
 1. Training statics are rudimentary.
 2. Doesn't currently support full cross-validation in training.
 3. No pre-processing of words yet - accuracy for my current test cases is sufficient with no preprocessing.

