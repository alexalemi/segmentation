# Try to train glove and word2vec on the brown corpus


out/word2vec.100.vec:
	../arxiv_word2vec/word2vec/word2vec -train data/brown.txt -size 100 -negative 5 -threads 10 -debug 2 -save-vocab out/word2vec.vocab -output $@

out/word2vec.50.vec:
	../arxiv_word2vec/word2vec/word2vec -train data/brown.txt -size 50 -negative 5 -threads 10 -debug 2 -save-vocab out/word2vec.vocab -output $@

out/glove.vocab: data/brown.txt
	../arxiv_glove/glove/vocab_count -verbose 2 -min-count 5 < $< > $@

out/glove.co: out/glove.vocab
	../arxiv_glove/glove/cooccur -verbose 2 -vocab-file $< -memory 30 < $< > $@

out/glove.shuf.co: out/glove.p.co
	../arxiv_glove/glove/shuffle -verbose 2 -memory 30 < $< > $@

out/glove.100.vec: out/glove.shuf.co
	../arxiv_glove/glove/glove -verbose 2 -vector-size 100 -threads 10 -input-file $< -vocab-file out/glove.vocab -save-file $@ -window-size 5 -binary 0

out/glove.50.vec: out/glove.shuf.co
	../arxiv_glove/glove/glove -verbose 2 -vector-size 50 -threads 10 -input-file $< -vocab-file out/glove.vocab -save-file $@ -window-size 5 -binary 0

glove: out/glove.50.vec out/glove.100.vec

word2vec: out/word2vec.100.vec out/word2vec.50.vec
