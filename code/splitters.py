"""
Collect the various splitting strategies in one place
"""

import numpy as np
from scipy.ndimage import generic_filter
from scipy.spatial.distance import cdist
from numpy.random import rand
import tools


####################
# C99
####################

def rankkern(x):
    """ The kernel for the rank transformation, measures the fraction of the neighbors that
    take on a value less than the middle value """
    n = x.size
    mid = n//2
    better = ( (x >= 0) & (x<x[mid]) ).sum()
    return better / ( (x>=0).sum() - 1.0)

def rankify(mat, size=11):
    """ Apply the ranking transformation of a given size """
    return generic_filter(mat, rankkern, size=(size,size), mode='constant', cval=-1)

def c99score(distsmat, hyp, minlength=1, maxlength=None):
    """ Do the choi c99 scoring for a hypothesis splitting """
    N = distsmat.shape[0]
    beta = 0.0
    alpha = 0.0
    for (a,b) in tools.seg_iter(hyp):
        beta += distsmat[a:b,a:b].sum()
        alpha += (b-a)**2
        if minlength:
            if (b-a) < minlength: beta += -np.inf
        if maxlength:
            if (b-a) > maxlength: beta += -np.inf
    return -beta/(alpha+0.)

def c99split(distsmat, k, rank=0, *args, **kwargs):
    """ Do the Choi style c99 splitting, given a matrix of distances D,
    and k splits to perform.  The rank keyword denotes whether we want to 
    do the ranking transformation if positive and if so denotes the size of the
    ranking filter """

    # perform ranking if desired
    if rank:
        distsmat = rankify(distsmat, rank)

    N = distsmat.shape[0]
    score = np.inf
    splits = [N]
    n = 0
    while n < k:
        newans = min( 
                ( c99score( distsmat, sorted(splits+[i]), *args, **kwargs ), splits+[i] ) 
                    for i in xrange(1,N-1) if i not in set(splits) )
        n += 1
        splits = newans[1]
        score = newans[0]
    return sorted(splits), score

    
####################
# DP
####################

# The dynamic programming splitter

def gensig_euclidean(X,minlength=1,maxlength=None):
    """ Generate the sigma for the squared difference from the mean """
    cs = X.cumsum(0)
    css = (X**2).sum(1).cumsum(0)   
    def sigma(i,j): 
        length = j-i
        if minlength:
            if length < minlength: return np.inf
        if maxlength:
            if length > maxlength: return np.inf
        if i == 0:
            return css[j-1] - 1./j * ((cs[j-1])**2).sum() 
        else: 
            return ( css[j-1]-css[i-1] ) - 1./(j-i) * ((cs[j-1] - cs[i-1])**2).sum() 
    return sigma


def gensig_cosine(X, minlength=1, maxlength=None):
    """ Generate the sigma for the cosine similarity """
    def sigma(a,b):
        length = (b-a)
        if minlength:
            if length < minlength: return np.inf
        if maxlength:
            if length > maxlength: return np.inf
        rep = X[a:b].mean(0)
        if length < 2:
            return np.inf
        return (cdist( X[a:b], [ rep ], 'cosine')**2).sum()
    return sigma


def gensig_model_old(X, minlength=1, maxlength=None, lam=0.0):
    N,D = X.shape
    over_sqrtD = 1./np.sqrt(D)
    def sigma(a,b):
        length = (b-a)
        if minlength:
            if length < minlength: return np.inf
        if maxlength:
            if length > maxlength: return np.inf
        rep = (2*(X[a:b].sum(0)>0)-1)*over_sqrtD
        return -X[a:b].dot(rep).sum() 
        # return -X[a:b].dot(rep).sum() + lam*np.sqrt(length)/np.log(N)
    return sigma

def gensig_model(X, minlength=1, maxlength=None, lam=0.0):
    N,D = X.shape
    over_sqrtD = 1./np.sqrt(D)
    cs = np.cumsum(X,0)

    def sigma(a,b):
        length = (b-a)
        if minlength:
            if length < minlength: return np.inf
        if maxlength:
            if length > maxlength: return np.inf

        tot = cs[b-1].copy()
        if a > 0:
            tot -= cs[a-1]
        signs = np.sign(tot)
        return -over_sqrtD*(signs*tot).sum()
    return sigma


def tiebreak():
    return 1e-10*rand()

def gensig_choi(distsmat, minlength=1, maxlength=None, rank=0):
    """ The two dimensional sigma function for the c99 splitting """
    if rank:
        distsmat = rankify(distsmat, rank)
    def sigma(a,b):
        length = (b-a)
        beta = distsmat[a:b,a:b].sum()
        alpha = (b-a)**2
        if minlength:
            if (b-a) < minlength: beta += np.inf
        if maxlength:
            if (b-a) > maxlength: beta += np.inf
        return (-beta, alpha)
    return sigma

def dpsplit(n,k, sig):
    """ Perform the dynamic programming optimal segmentation, using the sig function
    to determine the cost of a segment sig(i,j) is the cost of the i,j segment.  These
    are then added together
    """

    # Set up the tracking tables
    K = k + 1
    N = n
    segtable = np.zeros((n,K)) + np.nan
    segtable[:,0] = [ sig(0,j+1) for j in xrange(N) ]
    segindtable = np.zeros((N,K), dtype='int') - 1

    # fill up the table in a clever order
    for k in xrange(1,K):
        for j in xrange(k,N):
            #fill the j,k element
            ans = min( ( (segtable[l,k-1] + sig(l+1,j+1), l+1 )
                for l in xrange(k-1,j) ) )
            segtable[j,k] = ans[0]
            segindtable[j,k] = ans[1]

    # read out the path
    current_pointer = segindtable[-1,K-1]
    path = [current_pointer]
    for k in xrange(K-2, 0, -1):
        current_pointer = segindtable[current_pointer-1, k]
        path.append(current_pointer)

    return sorted(path + [N]), segtable[-1,K-1]

def dpsplit_general(n,k, sig, combine=lambda a,b: a+b, key=lambda a: a, d=1):
    """ Perform the dynamic programming optimal segmentation, using the sig function
    to determine the cost of a segment sig(i,j) is the cost of the i,j segment.  These
    are then added together using the combine function and reduced to a scalar cost with the 
    key function.  d sets the dimensionality of the intermediary representation
    """

    # Set up the tracking tables
    K = k + 1
    N = n
    if d > 1:
        segtable = np.zeros((n,K,d)) + np.nan
    else:
        segtable = np.zeros((n,K)) + np.nan
    segtable[:,0] = [ sig(0,j+1) for j in xrange(N) ]
    segindtable = np.zeros((N,K), dtype='int') - 1

    # fill up the table in a clever order
    for k in xrange(1,K):
        for j in xrange(k,N):
            #fill the j,k element
            ans = min( ( ( combine(segtable[l,k-1],sig(l+1,j+1)), l+1 )
                for l in xrange(k-1,j) ), key=lambda x: key(x[0]) )
            segtable[j,k] = ans[0]
            segindtable[j,k] = ans[1]

    # read out the path
    current_pointer = segindtable[-1,K-1]
    path = [current_pointer]
    for k in xrange(K-2, 0, -1):
        current_pointer = segindtable[current_pointer-1, k]
        path.append(current_pointer)

    return sorted(path + [N]), key(segtable[-1,K-1])


####################
# Greedy
####################


def greedysplit(n, k, sigma):
    """ Do a greedy split """
    splits = [n]
    s = sigma(0,n)

    def score(splits, sigma):
        splits = sorted(splits)
        return sum( sigma(a,b) for (a,b) in tools.seg_iter(splits) )

    while k > 0:
        usedinds = set(splits)
        new = min( ( score( splits + [i], sigma), splits + [i] )
                for i in xrange(1,n) if i not in usedinds )
        splits = new[1]
        s = new[0]
        k -= 1
    return sorted(splits), s

def greedysplit_general(n, k, sigma, combine=lambda a,b: a+b, key=lambda a: a):
    """ Do a greedy split """
    splits = [n]
    s = sigma(0,n)

    def score(splits, sigma):
        splits = sorted(splits)
        return key( reduce( combine, (sigma(a,b) for (a,b) in tools.seg_iter(splits) ) ))

    while k > 0:
        usedinds = set(splits)
        new = min( ( score( splits + [i], sigma), splits + [i] )
                for i in xrange(1,n) if i not in usedinds )
        splits = new[1]
        s = new[0]
        k -= 1
    return sorted(splits), s

def bestsplit(low, high, sigma, minlength=1, maxlength=None):
    """ Find the best split inside of a region """
    length = high-low
    if length < 2*minlength:
        return (np.inf, np.inf, low)
    best = min( ((sigma(low,j), sigma(j, high), j) for j in xrange(low+1,high)), key=lambda x: x[0]+x[1] )
    return best


def greedysplit_old(n, k, sigma):
    """ Do a greedy split """
    k = k + 1
    splits = [0,n]
    costs = [sigma(0,n)]
    cost = costs[0]
    # path = []

    while k > 0:
        bestcosts = []
        bsp = []
        bestcost = np.inf
        for j in xrange(len(splits)-1):
            left, right, sp = bestsplit(splits[j], splits[j+1], sigma)
            newcost = left+right + sum(costs[:j]) + sum(costs[j+1:])
            if newcost < bestcost:
                bestcost = newcost
                bsp = splits[:j+1] + [sp] + splits[j+1:]
                bestcosts = costs[:j] + [left,right] + costs[j:]
        costs = bestcosts
        cost = bestcost
        splits = bsp
        # path.append( (splits, cost, k*(d+1)*np.log(d*top) ) )
        k -= 1

    return splits[1:], cost


def refine(splits, sigma, n=1):
    """ Given some splits, refine them a step """
    oldsplits = splits[:]
    counter = 0
    n = n or np.inf

    while counter < n:
        splits = [0]+splits
        n = len(splits) - 2
        new = [splits[0]]
        for i in xrange(n):
            out = bestsplit(splits[i], splits[i+2], sigma)
            new.append(out[2])
        new.append(splits[-1])
        splits = new[1:]

        if splits == oldsplits:
            break
        oldsplits = splits[:]
        counter += 1

    return splits

def bestsplit_general(splits, pk, sigma, combine=lambda a,b: a+b, key=lambda a: a):
    """ Move the pk-th split to its best location """
    def score(splits, sigma):
        splits = sorted(splits)
        return key( reduce( combine, (sigma(a,b) for (a,b) in tools.seg_iter(splits) ) ))

    if pk == 0:
        left = 0
    else:
        left = splits[pk-1]
    right = splits[pk+1]

    best = min( (score( splits[:pk] + [j] + splits[pk+1:], sigma),j) for j in xrange(left+1,right) )
    return best[1]

def refine_general(splits, sigma, n=1, combine=lambda a,b: a+b, key=lambda a: a):
    """ Do a general refinement of up to n steps """
    oldsplits = splits[:]
    N = splits[-1]
    counter = 0
    k = len(splits)
    n = n or np.inf

    while counter < n:
        splits = [ bestsplit_general(splits, i, sigma, combine, key) for i in xrange(k-1) ] + [N]

        if splits == oldsplits:
            break
        oldsplits = splits[:]
        counter += 1

    return splits

