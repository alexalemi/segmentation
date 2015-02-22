""" 
Code for figuring out various vector representions of documents
"""

import numpy as np
from collections import defaultdict
import tools

def tf_sents(doc):
    """ Create a sentence level tf representation of the document """
    words = set( word for word in tools.word_iter(doc) )
    word_pk = { word:pk for pk,word in enumerate(words) }

    vecs = []
    for part in doc:
        for sent in part:
            wordcounter = defaultdict(int)
            for word in sent:
                wordcounter[word] += 1

            vec = np.zeros(len(words))
            for word,count in wordcounter.iteritems():
                if word in words:
                    vec[word_pk[word]] += count
            vecs.append(vec)

    return np.array(vecs)

def tf_words(doc):
    """ Create a sentence level tf representation of the document """
    words = set( word for word in tools.word_iter(doc) )
    word_pk = { word:pk for pk,word in enumerate(words) }

    vecs = []
    for part in doc:
        for sent in part:
            for word in sent:
                vec = np.zeros(len(words))
                if word in words:
                    vec[word_pk[word]] += 1
                vecs.append(vec)

    return np.array(vecs)


def vec_sents(doc, word_lookup, wordreps):
    """ Create a vector representation of the document """
    vecs = []
    for part in doc:
        for sent in part:
            wordvecs = [np.zeros(wordreps.shape[1])]
            for word in sent:
                pk = word_lookup.get(word,-1)
                if pk >= 0:
                    wordvecs.append( wordreps[pk] )
            vecs.append( np.mean(wordvecs,0) )

    return np.array(vecs)

def vec_words(doc, word_lookup, wordreps):
    """ Create a vector representation of the document """
    vecs = []
    for part in doc:
        for sent in part:
            for word in sent:
                pk = word_lookup.get(word,-1)
                if pk >= 0:
                    vecs.append( wordreps[pk] )
                else:
                    vecs.append( np.zeros(wordreps.shape[1]) )

    return np.array(vecs)

def vectop_sents(doc, word_lookup, wordreps):
    """ Create a vector representation of the document """
    vecs = []
    N = wordreps.max()+1
    for part in doc:
        for sent in part:
            sentvec = np.zeros(N)
            for word in sent:
                pk = word_lookup.get(word,-1)
                if pk >= 0:
                    sentvec[wordreps[word_lookup[word]]] += 1
            vecs.append( sentvec )

    return np.array(vecs)
            

def vecdf_sents(doc, word_lookup, wordreps, dfcounter):
    """ Create a vector representation of the document """
    vecs = []
    for part in doc:
        for sent in part:
            wordvecs = [np.zeros(wordreps.shape[1])]
            for word in sent:
                pk = word_lookup.get(word,-1)
                if pk >= 0:
                    wordvecs.append( np.log(500./(dfcounter.get(word,1.0)+0.0))*wordreps[pk] )
            vecs.append( np.mean(wordvecs,0) )

    return np.array(vecs)


def vecdf_words(doc, word_lookup, wordreps, dfcounter):
    """ Create a vector representation of the document """
    vecs = []
    for part in doc:
        for sent in part:
            for word in sent:
                pk = word_lookup.get(word,-1)
                if pk >= 0:
                    vecs.append( np.log(500./(dfcounter.get(word,1.0)+0.0))*wordreps[pk] )
                else:
                    vecs.append( np.zeros(wordreps.shape[1]) )
    return np.array(vecs)
