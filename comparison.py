from numpy.random import multinomial
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gsdmm import MovieGroupProcessArray, MovieGroupProcess

np.random.seed(1)

docs = [
    'A p-value is a measure of the probability that an observed difference could have occurred just by random chance',
    'In null hypothesis significance testing, the p-value is the probability of obtaining test results at least as extreme as the results actually observed',
    'A p-value, or probability value, is a number describing how likely it is that your data would have occurred by random chance',
    'A p-value is used in hypothesis testing to help you support or reject the null hypothesis',
    'The P-value, or calculated probability, is the probability of finding the observed, or more extreme, results when the null hypothesis',
    'A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes',
    'An artificial neural network is an interconnected group of nodes, inspired by a simplification of neurons in a brain',
    'Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning ',
    'Modeled loosely on the human brain, a neural net consists of thousands or even millions of simple processing nodes that are densely',
    'Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns']

stopwords = ['this', 'is', 'a', 'the', 'of', 'an', 'that', 'or']
docs_toks = [doc.lower().replace(',', '').replace('.', '').split() for doc in docs]
docs_toks = [[w for w in doc if w not in stopwords] for doc in docs_toks]
common_dictionary = Dictionary(docs_toks)
common_corpus = [common_dictionary.doc2bow(text) for text in docs_toks]

mgp_ar = MovieGroupProcessArray(K=10, alpha=0.1, beta=0.1, n_iters=22)
mgp = MovieGroupProcess(K=10, alpha=0.1, beta=0.1, n_iters=22)

y = mgp_ar.fit(common_corpus, len(common_dictionary))
y_old = mgp.fit(common_corpus, len(common_dictionary))

print(mgp_ar.top_words(common_dictionary))
print(mgp.top_words(common_dictionary))