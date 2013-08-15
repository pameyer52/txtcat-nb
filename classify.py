#!/usr/bin/env python

import os.path
import os
from NaiveBayes import NaiveBayesClassifier
from nbio import load_words, load_classifier

def classify( infile, data_dir, k_smooth = 1 ):
    '''
    assign classification labels to documents using trained classifier.
    '''
    #initialize classifier
    nbc = load_classifier( infile )
    print('classifier loaded')

    #load inputs
    fws = []
    for fn in os.listdir( data_dir ):
        ws = load_words( os.path.join( data_dir, fn ) )
        fws.append( (fn, ws) )
    print('loaded %s files for classification' % len(fws) )
    
    #classify
    for (f, ws) in fws:
        (l,u) = nbc.classify( ws, k_smooth )
        print('%s classified as %s LPG=%f' % (f, l,u) )

if __name__ == '__main__':
    import getopt
    import sys
    def usage():
        ''' show usage message and exit '''
        print('classify (txtcat-nb)')
        print('options:')
        print('\t-h --help\t\t\tshow this usage information')
        print('\t-m --model=\t\t\tclassification model file')
        print('\t-d --datadir=\t\t\tdata directory')
        print('\t-k --smooth=\t\t\tsmoothing constant (default 1)')
        sys.exit(1)

    try:
        opts, args = getopt.getopt( sys.argv[1:], 'hm:d:k:',['help','model=','datadir=','smooth='])
    except getopt.GetoptError as err:
        print(err)
        usage()
    if 0 != len(args):
        usage()
    mfile = None
    ddir = None
    k = 1
    for o,a in opts:
        if o in ('-h',' --help'):
            usage()
        elif o in ('-m', '--model'):
            mfile = a
        elif o in ('-d', '--datadir'):
            ddir = a
        elif o in ('-k', '--smooth'):
            k = int( a )
            assert k >= 0, 'smoothing constant must be non-negative'
        else:
            print('unrecognized option %s (argument %s)' % (o,a) )
            usage()
    if None == mfile:
        usage()
    if None == ddir:
        usage()
    classify( mfile, ddir, k )

