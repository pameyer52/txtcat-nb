#!/usr/bin/env python

import json
from NaiveBayes import nbHelper, NaiveBayesClassifier

def load_words( infile ):
    inp = open(infile, 'r')
    txt = inp.read()
    inp.close()
    w0 = txt.split()
    #TODO - preprocess here as necessary
    return w0 

class NaiveBayesJSONEncoder( json.JSONEncoder ):

    def default( self, o ):
        if isinstance( o, nbHelper ):
            return {'nbHelper': o.__dict__ }
        elif isinstance( o, NaiveBayesClassifier ):
            return {'NaiveBayesClassifier': o.__dict__ }
        return json.JSONEncoder.default( self, o )

def pack_classifier( nbc ):
    enc = NaiveBayesJSONEncoder()
    jtxt = enc.encode( nbc )
    return jtxt

def load_classifier( infile ):
    inp = open(infile, 'r')
    jtxt = inp.read()
    inp.close()
    jd = json.loads( jtxt )
    return NaiveBayesClassifier.unfold_classifier( jd )

def store_classifier( outfile, nbc ):
    opf = open(outfile, 'w' )
    jtxt = pack_classifier( nbc )
    opf.write( jtxt )
    opf.close()
