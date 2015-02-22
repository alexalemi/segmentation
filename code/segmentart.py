import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import tools
import splitters
import representations
import re
import sys

K = int(sys.argv[1])
infile = sys.argv[2]

with open(infile,"r") as f:
    txt = f.read()

punctuation_pat = re.compile(r"""([!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~])""")
hyphenline_pat = re.compile(r"-\s*\n\s*")
multiwhite_pat = re.compile(r"\s+")
cid_pat = re.compile(r"\(cid:\d+\)")
nonlet = re.compile(r"([^A-Za-z0-9 ])")
def clean_text(txt):
    txt = txt.decode("utf-8")

    txt = txt.lower()
    txt = cid_pat.sub(" UNK ", txt)
    txt = hyphenline_pat.sub("", txt)
    # print punctuation_pat.findall(txt)
    txt = punctuation_pat.sub(r" \1 ", txt)
    txt = re.sub("\n"," NL ", txt)
    txt = nonlet.sub(r" \1 ", txt)

    # txt = punctuation_pat.sub(r"", txt)
    # txt = nonlet.sub(r"", txt)

    txt = multiwhite_pat.sub(" ", txt)
    txt = txt.encode('utf-8')
    return "".join(["START ", txt.strip(), " END"])

txt = clean_text(txt).split()

vecs = np.load("/home/aaa244/storage/arxiv_glove/bigrun/data/mats/vecs.npy")
words = np.load("/home/aaa244/storage/arxiv_glove/bigrun/data/mats/vocab.npy")
word_lookup = {w:c for c,w in enumerate(words) }

print "article length:", len(txt)

X = []

mapper = {}
count = 0
for i,word in enumerate(txt):
    if word in word_lookup:
        mapper[i] = count
        count += 1
        X.append( vecs[word_lookup[word]] )

mapperr = { v:k for k,v in mapper.iteritems() }

X = np.array(X)
print "X length:", X.shape[0]

sig = splitters.gensig_model(X)
print "Splitting..."
splits,e = splitters.greedysplit(X.shape[0], K, sig)
print splits
print "Refining..."
splitsr = splitters.refine(splits, sig, 20)
print splitsr

print "Printing refined splits... "

for i,s in enumerate(splitsr[:-1]):
    k = mapperr[s]
    print
    print i,s
    print " ".join(txt[k-100:k]), "\n\n", " ".join(txt[k:k+100])

with open("result{}.txt".format(K),"w") as f:
    prev = 0
    for s in splitsr:
        k = mapperr.get(s,len(txt))
        f.write(" ".join(txt[prev:k]).replace("NL","\n"))
        f.write("\nBREAK\n")
        prev = k


print "Done"
