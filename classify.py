#!/usr/bin/env python

import os.path
import os
from NaiveBayes import NaiveBayesClassifier
from nbio import load_words, load_classifier

def cp(src, dst):
    inp = open(src, 'r')
    opf = open(dst, 'w')
    opf.write( inp.read() )
    inp.close()
    opf.close()

def classify( infile, data_dir, out_dir, f_fnc = None ):
    '''
    assign classification labels to documents using trained classifier.
    Optionally apply f_fnc to results (to move/copy documents into 
    subdirectories of out_dir according to assigned label).
    '''
    #TODO - verbose as option
    #TODO - k as option
    #TODO - confusion stats 

    #initialize classifier
    nbc = load_classifier( infile )
    print('classifier loaded')

    #load inputs
    fws = []
    for fn in os.listdir( data_dir ):
        ws = load_words( os.path.join( data_dir, fn ) )
        fws.append( (fn, ws) )
    print('loaded %s files for classification' % len(fws) )
    
    #setup outputs
    if None != f_fnc:
        if os.path.exists( out_dir ):
            print('output directory %s already exists, aborting' % out_dir)
            return
        os.mkdir( out_dir )
        labels = nbc.get_labels()
        for l in labels:
            p = os.path.join( out_dir, l )
            os.mkdir( p )

    #classify
    for (f, ws) in fws:
        l = nbc.classify( ws )
        print('%s classified as %s' % (f, l) )
        if None != f_fnc:
            s = os.path.join( data_dir, f )
            d = os.path.join( out_dir, l, f )
            f_fnc( s, d )

if __name__ == '__main__':
    def usage():
        ''' show usage message and exit '''
        print('usage message goes here')
        sys.exit(1)
    import getopt
    import sys
    #model_file, data_directory, output_directory
    # -m --model %a , -d --datadir %a 
    try:
        opts, args = getopt.getopt( sys.argv[1:], 'hm:d:',['help','model','datadir'])
    except getopt.GetoptError as err:
        print(err)
        usage()
    if 0 != len(args):
        usage()
    mfile = None
    ddir = None
    odir = None
    for o,a in opts:
        if o in ('-h',' --help'):
            usage()
        elif o in ('-m', '--model'):
            mfile = a
        elif o in ('-d', '--datadir'):
            ddir = a
        else:
            print('unrecognized option %s (argument %s)' % (o,a) )
            usage()
    if None == mfile:
        usage()
    if None == ddir:
        usage()
    classify( mfile, ddir, odir )

