"""
A set of useful tools for handing the text segmentation tasks.
"""
import numpy as np
from itertools import chain, tee, izip
# from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stopword_set = set()

with open("/home/aaa244/storage/segmentation/data/STOPWORD.list") as f:
    for line in f:
        stopword_set.add(line.strip())
# add punctuation
stopword_set.update(["''",",",".","``","'","!",'"',"#","$","%","&","(",")","*","+","-","/",
    ":",";","<","=",">","?","@","[","\\","]","^","_","`","{","|","}","~"])

stemmer = PorterStemmer()

CHOI_TEMPLATE = "/home/aaa244/storage/segmentation/data/choi/{}/{}/{}.ref"
def choi_loader(doc, tp, ref, word_cut=0, remove_stop=False, stem=False):
    """ Load a choi document from the dataset,
        returns a list of parts
        each part is a list of sentences
        each sentence is a list of words,
        the only preprocessing is to lowercase everything """

    # Open the corresponding file
    with open(CHOI_TEMPLATE.format(doc,tp,ref)) as f:
        doc = f.read()

    def is_valid(word):
        if stem:
            word = stemmer.stem(word)
        if remove_stop:
            if word in stopword_set: return False
        return True
    
    # Split into parts 
    parts = [ x.splitlines() for x in doc.split("==========\n") if x ]

    doc = [ [ [ x.lower() for x in sent.split() if is_valid(x) ] for sent in doc ] for doc in parts ]

    filtered = [ [sent for sent in part if len(sent) > word_cut ] for part in doc ]

    return filtered

ARX_TEMPLATE = "/home/aaa244/storage/segmentation/data/arxiv/{:03d}.ref"
def arx_loader(num):
    """ Load an arxiv document from the dataset,
        returns a list of parts
        each part is a list of words"""

    # Open the corresponding file
    with open(ARX_TEMPLATE.format(num)) as f:
        doc = f.read()
    
    # Split into parts 
    return [ [ [x] for x in x.split() ] for x in doc.split("BR") ]


def allchoi(set, *args, **kwargs):
    if set=="3-5":
        for a in [1,2]:
            for i in xrange(50):
                yield choi_loader(a,"3-5",i, *args, **kwargs)
    if set=="6-8":
        for a in [1,2]:
            for i in xrange(50):
                yield choi_loader(a,"6-8",i, *args, **kwargs)
    if set=="9-11":
        for a in [1,2]:
            for i in xrange(50):
                yield choi_loader(a,"9-11",i, *args, **kwargs)
    if set=="3-11":
        for a in [1,2]:
            for i in xrange(50):
                yield choi_loader(a,"3-11",i, *args, **kwargs)
        for i in xrange(300):
            yield choi_loader(3,"3-11",i, *args, **kwargs)

def collapse(doc):
    """ Turn a document into a single string """
    return " ".join( " ".join(" ".join(sent) for sent in part) for part in doc )

def collapse_sents(doc):
    """ Collapse a doc to a list of sentences """
    return [ sent for part in doc for sent in part ]

def collapse_words(doc):
    """ Collapse a doc to a list of words """
    return [ word for part in doc for sent in part for word in sent ]

def word_iter(doc):
    """ Iterate over the words in a document """
    words = ( word for part in doc for sent in part for word in sent )
    for word in words: 
        yield word

def sent_iter(doc):
    """ Iterate over the sentences in a document """
    sents = ( sent for part in doc for sent in part )
    for sent in sents:
        yield sent

def refsplit(doc):
    """ Get the reference splitting for the document """
    middle =  np.cumsum([1] +[ sum(1 for sent in part for word in sent) for part in doc ])
    return (middle[1:-1]-1).tolist() + [middle[-1]-1]

def refsplit_sent(doc):
    """ Get the reference splitting for the sentence representation """
    middle =  np.cumsum([1] +[ sum(1 for sent in part) for part in doc ])
    return (middle[1:-1]-1).tolist() + [middle[-1]-1]


# A testing document in the same structure as a choi doc
testdoc = [ ["this is the first sentence".split()]*5 , 
       ["second sentence same as the first".split()]*3 ,
       ["the blue fish went to the market".split()]*4,
       ["once upon a midnight dreary with my pack".split()]*5,
       ["while i pondered weak and weary".split()]*3,
       ["one fish two fish three fish blue fish".split()]*5,
       ["pack it up pack it in let me begin".split()]*3,
       ["i came to win battle me that is a sin to begin".split()]*5,
       ["and think about how ravens and writing desks".split()]*4, 
       ["other people are people too not ravens or fish".split()]*3 ]

# Utility

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def seg_iter(splits):
    return pairwise([0] + splits)

def length_iter(splits):
    return ( (b-a) for (a,b) in seg_iter(splits) )

# Scoring functions

xnor = lambda a,b: (a and b) or (not a and not b)

def score(hyp, ref, k=None):
    """ The Pk metric from Beeferman """
    # if k is undefined, use half the mean segment size
    k = k or int(round(0.5*ref[-1]/len(ref)))-1

    length = ref[-1]
    probeinds = np.arange(length - k)
    dref = np.digitize(probeinds, ref) == np.digitize(probeinds+k, ref)
    dhyp = np.digitize(probeinds, hyp) == np.digitize(probeinds+k, hyp)
    
    return (dref ^ dhyp).mean()

def score_wd(hyp, ref, k=None):
    """ The window diff metric of Pevzner """
    k = k or int(round(0.5*ref[-1]/len(ref)))-1

    length = ref[-1]
    hyp = np.asarray(hyp)
    ref = np.asarray(ref)

    score = 0.0
    tot = 0.0
    for i in xrange(length - k):
        bref = ((ref > i) & (ref <= i+k)).sum()
        bhyp = ((hyp > i) & (hyp <= i+k)).sum()
        score += 1.0*(np.abs(bref-bhyp) > 0)
        tot += 1.0
    return score/tot
    


