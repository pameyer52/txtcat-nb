#!/usr/bin/env python

'''
classification statistics
'''

def calc_F1(preds, obs):
    '''
    multi-class F1 score
    '''
    n = len(preds)
    assert n == len(obs), 'mismatch in number of predictions and observations'
    vocab = set(preds).union( set(obs) )
    rs = {}
    for sy in vocab:
        p_sy = [ 1 if sy == x else 0 for x in preds]
        o_sy = [ 1 if sy == x else 0 for x in obs ]
        po = zip( p_sy, o_sy )
        tp = float(sum( [ 1 if (1 == p and 1 == o) else 0 for (p,o) in po ] ))
        fp = float(sum( [ 1 if (1 == p and 0 == o) else 0 for (p,o) in po ] ))
        fn = float(sum( [ 1 if (0 == p and 1 == o) else 0 for (p,o) in po ] ))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = ( 2 * prec * rec )  / ( prec + rec )
        r = { 'F1':F1, 'precision':prec, 'recall':rec }
        rs[ sy ] = r
    return rs

