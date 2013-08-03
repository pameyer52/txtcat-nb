#!/usr/bin/env python

''' multi-class naive bayes classifier '''
from collections import Counter
from math import log

class nbHelper:
    ''' single-class helper for naive bayes '''
    def __init__(self):
        self.words = Counter()
        self.n_examples = 0 #number of training examples seen

    def add_example( self, words ):
        '''
        store counts for a training example
        '''
        self.n_examples += 1
        for word in words:
            self.words[ word ] += 1

class NaiveBayesClassifier:
    ''' multi-class naive bayes classifier '''
    def __init__(self, kclassnames):
        self.vocabulary = []
        self.all_words = []
        self.vocab_set = False
        self.classify_prep = False
        self.helpers = {}
        for kclassname in kclassnames:
            self.helpers[ kclassname ] = nbHelper()
    
    def get_labels(self):
        ''' return set of labels for this classifier '''
        return self.helpers.keys() 

    @classmethod
    def unfold_classifier( cls, d ):
        '''
        initialize classifier from unpacked json data dictionary
        '''
        d0 = d['NaiveBayesClassifier']
        hd0 = d0['helpers']
        kclassnames = hd0.keys()
        c = cls( kclassnames )
        c.helpers = {}
        c.all_words = d0['all_words']
        for kc in kclassnames:
            h = nbHelper()
            h.n_examples = hd0[kc]['nbHelper']['n_examples']
            h.words = Counter( hd0[kc]['nbHelper']['words'] )
            c.helpers[ kc ] = h
        return c

    def add_example(self, kclass, words ):
        '''
        store counts for training example
        '''
        try:
            helper = self.helpers[ kclass ]
        except KeyError:
            raise TypeError('unknown kclass "%s"' % kclass )
        helper.add_example( words )
        self.vocab_set = False
        self.classify_prep = False
        for w in words:
            self.all_words.append( w )

    def set_vocab(self):
        '''
        counts for classification
        '''
        seen = {}
        self.vocabulary = []
        for item in self.all_words:
            if item in seen:
                continue
            seen[item] = 1
            self.vocabulary.append( item )
        self.vocab_set = True

    def prep_classify(self):
        '''
        setup for classification
        '''
        nc = len( self.helpers )
        self.lp = {} 
        self.n_c = {}
        n_examples_t = 0
        for kclass in self.helpers.keys():
            n_examples_t += self.helpers[ kclass ].n_examples
        for kclass in self.helpers.keys():
            self.lp[kclass] = log( self.helpers[kclass].n_examples, 10 ) - log( n_examples_t, 10)
            self.n_c[kclass] = sum( self.helpers[kclass].words.values() )
        self.nv = len( self.vocabulary )

    def classify(self, words, k = 1, verbose = False):
        '''
        classify list of words
        '''
        #prep
        if not self.vocab_set:
            self.set_vocab()
        if not self.classify_prep:
            self.prep_classify()

        l_p = dict( self.lp )
        #classify
        for word in words:
            for kclass in self.helpers.keys():
                c_w = self.helpers[kclass].words[word]
                lpc = log( c_w + k, 10) - log(self.n_c[kclass] + (k*self.nv),10)
                l_p[kclass] += lpc
        #pick class label with highest log prob
        if verbose:
            print( l_p )
        return max( l_p, key = l_p.get )

