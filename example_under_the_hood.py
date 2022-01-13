#!/usr/bin/env python
# coding: utf-8

# In[31]:


from numpy.random import multinomial
from numpy import log, exp
from numpy import argmax
import numpy as np
import json
from sklearn.datasets import fetch_20newsgroups
# np.random.seed(1)

# In[2]:


class MovieGroupProcess:
    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):
        '''
        A MovieGroupProcess is a conceptual model introduced by Yin and Wang 2014 to
        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the
        clustering short text documents.
        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        Imagine a professor is leading a film class. At the start of the class, the students
        are randomly assigned to K tables. Before class begins, the students make lists of
        their favorite films. The teacher reads the role n_iters times. When
        a student is called, the student must select a new table satisfying either:
            1) The new table has more students than the current table.
        OR
            2) The new table has students with similar lists of favorite movies.

        :param K: int
            Upper bound on the number of possible clusters. Typically many fewer
        :param alpha: float between 0 and 1
            Alpha controls the probability that a student will join a table that is currently empty
            When alpha is 0, no one will join an empty table.
        :param beta: float between 0 and 1
            Beta controls the student's affinity for other students with similar interests. A low beta means
            that students desire to sit with students of similar interests. A high beta means they are less
            concerned with affinity and are more influenced by the popularity of a table
        :param n_iters:
        '''
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters

        # slots for computed variables
        self.number_docs = None
        self.vocab_size = None
        self.cluster_doc_count = [0 for _ in range(K)]
        self.cluster_word_count = [0 for _ in range(K)]
        self.cluster_word_distribution = [{} for i in range(K)]

    @staticmethod
    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):
        '''
        Reconstitute a MovieGroupProcess from previously fit data
        :param K:
        :param alpha:
        :param beta:
        :param D:
        :param vocab_size:
        :param cluster_doc_count:
        :param cluster_word_count:
        :param cluster_word_distribution:
        :return:
        '''
        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)
        mgp.number_docs = D
        mgp.vocab_size = vocab_size
        mgp.cluster_doc_count = cluster_doc_count
        mgp.cluster_word_count = cluster_word_count
        mgp.cluster_word_distribution = cluster_word_distribution
        return mgp

    @staticmethod
    def _sample(p):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the multinomial distribution
        :return: int
            index of randomly selected output
        '''
        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]

    def fit(self, docs, vocab_size):
        '''
        Cluster the input documents
        :param docs: list of list
            list of lists containing the unique token set of each document
        :param V: total vocabulary size for each document
        :return: list of length len(doc)
            cluster label for each document
        '''
        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size

        D = len(docs)
        self.number_docs = D
        self.vocab_size = vocab_size

        # unpack to easy var names
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        cluster_count = K
        d_z = [None for i in range(len(docs))]

        # initialize the clusters
        for i, doc in enumerate(docs):

            # choose a random  initial cluster for the doc
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)

            for word in doc:
                if word not in n_z_w[z]:
                    n_z_w[z][word] = 0
                n_z_w[z][word] += 1

        for _iter in range(n_iters):
            total_transfers = 0

            for i, doc in enumerate(docs):

                # remove the doc from it's current cluster
                z_old = d_z[i]

                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)

                for word in doc:
                    n_z_w[z_old][word] -= 1

                    # compact dictionary to save space
                    if n_z_w[z_old][word] == 0:
                        del n_z_w[z_old][word]

                # draw sample from distribution to find new cluster
                p = self.score(doc)
                z_new = self._sample(p)

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1

                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                for word in doc:
                    if word not in n_z_w[z_new]:
                        n_z_w[z_new][word] = 0
                    n_z_w[z_new][word] += 1

            cluster_count_new = sum([1 for v in m_z if v > 0])
            print("In stage %d: transferred %d clusters with %d clusters populated" % (
            _iter, total_transfers, cluster_count_new))
            if total_transfers == 0 and cluster_count_new == cluster_count and _iter>25:
                print("Converged.  Breaking out.")
                break
            cluster_count = cluster_count_new
        self.cluster_word_distribution = n_z_w
        return d_z

    def score(self, doc):
        '''
        Score a document

        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param doc: list[str]: The doc token stream
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        '''
        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        p = [0 for _ in range(K)]

        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = log(m_z[z] + alpha)
        #  lN2 = log(D - 1 + K*alpha)
        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for word in doc:
                lN2 += log(n_z_w[label].get(word, 0) + beta)
            for j in range(1, doc_size +1):
                lD2 += log(n_z[label] + V * beta + j - 1)
            p[label] = exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm>0 else 1
        return [pp/pnorm for pp in p]

    def choose_best_label(self, doc):
        '''
        Choose the highest probability label for the input document
        :param doc: list[str]: The doc token stream
        :return:
        '''
        p = self.score(doc)
        return argmax(p),max(p)


# ## Example

# In[16]:


docs=['A p-value is a measure of the probability that an observed difference could have occurred just by random chance',
     'In null hypothesis significance testing, the p-value is the probability of obtaining test results at least as extreme as the results actually observed',
     'A p-value, or probability value, is a number describing how likely it is that your data would have occurred by random chance',
     'A p-value is used in hypothesis testing to help you support or reject the null hypothesis',
     'The P-value, or calculated probability, is the probability of finding the observed, or more extreme, results when the null hypothesis',
     'A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes',
     'An artificial neural network is an interconnected group of nodes, inspired by a simplification of neurons in a brain',
     'Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning ',
     'Modeled loosely on the human brain, a neural net consists of thousands or even millions of simple processing nodes that are densely',
     'Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns']


# In[49]:


stopwords=['this','is', 'a', 'the', 'of', 'an', 'that', 'or']
docs_toks=[doc.lower().replace(',','').replace('.','').split() for doc in docs]
docs_toks=[[w for w in doc if w not in stopwords] for doc in docs_toks]


# In[50]:


docs_toks[0]


# #### train

# In[51]:


alpha=0.1
beta=0.1
K=10
n_iters=30
V = len(set([item for sublist in docs_toks for item in sublist]))

D = len(docs)
number_docs = D
vocab_size = V


# In[52]:


cluster_doc_count = [0 for _ in range(K)]
cluster_word_count = [0 for _ in range(K)]
cluster_word_distribution = [{} for i in range(K)]
# unpack to easy var names
m_z, n_z, n_z_w = cluster_doc_count, cluster_word_count, cluster_word_distribution
d_z = [None for i in range(len(docs))]


# In[71]:


def _sample(p):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the multinomial distribution
        :return: int
            index of randomly selected output
        '''

        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]


# In[72]:


# initialize the clusters
for i, doc in enumerate(docs_toks):

    # choose a random  initial cluster for the doc
    z = _sample([1.0 / K for _ in range(K)])
    d_z[i] = z
    m_z[z] += 1
    n_z[z] += len(doc)

    for word in doc:
        if word not in n_z_w[z]:
            n_z_w[z][word] = 0
        n_z_w[z][word] += 1


# In[73]:


# n_z_w[0]


# In[74]:


def score(doc, alpha, beta, K, V, D, m_z, n_z, n_z_w):
        '''
        Score a document

        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param doc: list[str]: The doc token stream
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        '''

        p = [0 for _ in range(K)]

        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = log(m_z[z] + alpha)
        #  lN2 = log(D - 1 + K*alpha)
        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for word in doc:
                lN2 += log(n_z_w[label].get(word, 0) + beta)
            for j in range(1, doc_size +1):
                lD2 += log(n_z[label] + V * beta + j - 1)
            p[label] = exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm>0 else 1
        return [pp/pnorm for pp in p]


# In[75]:


for _iter in range(n_iters):
    total_transfers = 0

    for i, doc in enumerate(docs_toks):

        # remove the doc from it's current cluster
        z_old = d_z[i]

        m_z[z_old] -= 1
        n_z[z_old] -= len(doc)

        for word in doc:
            n_z_w[z_old][word] -= 1

            # compact dictionary to save space
            if n_z_w[z_old][word] == 0:
                del n_z_w[z_old][word]

        # draw sample from distribution to find new cluster
        p = score(doc, alpha, beta, K, V, D, m_z, n_z, n_z_w)
        z_new = _sample(p)

        # transfer doc to the new cluster
        if z_new != z_old:
            total_transfers += 1

        d_z[i] = z_new
        m_z[z_new] += 1
        n_z[z_new] += len(doc)

        for word in doc:
            if word not in n_z_w[z_new]:
                n_z_w[z_new][word] = 0
            n_z_w[z_new][word] += 1

    cluster_count_new = sum([1 for v in m_z if v > 0])
    print("In stage %d: transferred %d clusters with %d clusters populated" % (
    _iter, total_transfers, cluster_count_new))
    if total_transfers == 0 and cluster_count_new == cluster_count and _iter>25:
        print("Converged.  Breaking out.")
        break
    cluster_count = cluster_count_new
cluster_word_distribution = n_z_w


# ## Get topic words

# In[62]:


def top_words(cluster_doc_count, cluster_word_distribution, values, n_topics=15, join_tok=' '):
    doc_count = np.array(cluster_doc_count)
    top_index = doc_count.argsort()[-n_topics:][::-1]
    topic_words={}
    for cluster in top_index:
        sort_dicts = sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        words=join_tok.join([ wc[0] for wc in sort_dicts])
        topic_words[cluster]=words
    return topic_words


# In[63]:


print(top_words(cluster_doc_count, cluster_word_distribution, 5, K))


# In[ ]:



