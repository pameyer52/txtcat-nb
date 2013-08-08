#!/usr/bin/env python

import os.path
import os
from NaiveBayes import NaiveBayesClassifier
from nbio import load_words, store_classifier
import random
from stats import calc_F1

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
    def show_stats(s, lbl = None):
        ''' display F1 stats '''
        f1 = s['F1'] * 100.0
        pr = s['precision'] * 100.0
        rc = s['recall'] * 100.0
        if None != lbl:
            print('%s : F1=%f precision=%f recall=%f' % ( lbl, f1, pr, rc) )
        else:
            print('F1=%f precision=%f recall=%f' % ( f1, pr, rc) )
    if test_pct > 0.0:
        preds = []
        obs = []
        for (f, ws, l) in te_dl:
            obs.append(l)
            c = nbc.classify( ws )
            preds.append(c)
        sts = calc_F1( preds, obs )
        show_stats( sts['overall'] )
        for l in labels:
            show_stats( sts[l], l )
    #store trained classifier
    store_classifier( outfile, nbc )

if __name__ == '__main__':
    def usage():
        ''' show usage message and exit '''
        print('usage message goes here') #TODO - useful message
        sys.exit(1)

    #data_dir, outfile, test_pct, verbose
    # -d --datadir %a, -o --output %a, -p --pct %f , -v --verbose
    import getopt
    import sys
    try:
        opts, args = getopt.getopt( sys.argv[1:], 'hvd:o:p:',['help','verbose','datadir=','output=','pct='])
    except getopt.GetoptError as err:
        print(err)
        usage()
    if 0 != len(args):
        usage()
   
    verbose = False
    tpct = 0.3
    ddir = None
    ofile = None
    for o,a in opts:
        if o in ('-h','--help'):
            usage()
        elif o in ('-v', '--verbose'):
            verbose = True
        elif o in ('-d','--datadir'):
            ddir = a
        elif o in ('-o','--output'):
            ofile = a
        elif o in ('-p','--pct'):
            tpct = float( a )
        else:
            print('unrecognized option %s (argument %s)' % (o,a) )
            usage()
    assert None != ddir, 'no input data directory specified'
    assert None != ofile, 'no output file specified'
    assert ( 0.0 <= tpct ) and ( tpct <= 1.0 ), 'test percentage out of range'

    if not os.path.exists( ddir ):
        usage()

    #finally time to do work
    train( ddir, ofile, tpct, verbose )

