#!/usr/bin/env python

import os.path
import os
from NaiveBayes import NaiveBayesClassifier
from nbio import load_words, store_classifier
import random

def train( data_dir, outfile , test_pct = 0.3, verbose = True):
    '''
    train naive bayes classifier using ML estimates.
    if test percentage (test_pct) > 0, hold out that percentage of training 
    data files from each class and use for statistics.
    '''
    labels = os.listdir( data_dir )
    if verbose:
        print(labels)
    nbc = NaiveBayesClassifier( labels )

    def load_label_dir( ddir ):
        '''
        load all datafiles within single directory
        '''
        fnames = os.listdir( os.path.join( data_dir, ddir ) )
        def w0(f):
            ''' worker function (full path) '''
            return load_words( os.path.join( data_dir, ddir, f ) )
        return [ (fname, w0(fname) ) for fname in fnames ]

    te_dl = []
    n_tr = 0
    for label in labels:
        for (f, ws) in load_label_dir( label ):
            if random.random() < test_pct:
                te_dl.append( (f, ws, label) )
            else:
                nbc.add_example( label, ws )
                n_tr += 1
    
    #if we've picked a test set, use it
    if test_pct > 0.0:
        #TODO - stats for each class
        tp = 0
        fn = 0
        n_t = 0
        for (f, ws, l) in te_dl:
            n_t += 1
            c = nbc.classify( ws )
            if l == c:
                tp += 1
            else:
                fn += 1
        if n_t == 0 :
            print('empty test set - no statistics')
        else:
            p = 100.0 * float(tp) / n_t
            #TODO - precision/recall for multi-class classification
            print('%d of %d test files classified correctly : %4.2f %% (%d training files)' % ( tp, n_t, p, n_tr ) )

    #store trained classifier
    store_classifier( outfile, nbc )

if __name__ == '__main__':
    #TODO - argument parsing
    #train('data0','d0.json')
    train('data','d1.json')
