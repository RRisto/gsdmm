import os
from pathlib import Path

import numpy as np
from .dictionary import Dictionary


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
        self.cluster_doc_count = np.array([0 for _ in range(K)])
        self.cluster_word_count = np.array([0 for _ in range(K)])
        self.cluster_word_distribution = [{} for i in range(K)]

    @staticmethod
    def from_data(K, alpha, beta, dictionary, cluster_doc_count, cluster_word_count, cluster_word_distribution):
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
        mgp.number_docs = cluster_doc_count.sum()
        mgp.dictionary = dictionary
        mgp.vocab_size = len(dictionary)
        mgp.cluster_doc_count = cluster_doc_count
        mgp.cluster_word_count = cluster_word_count
        mgp.cluster_word_distribution = cluster_word_distribution
        return mgp

    def save(self, folder_path):
        '''
       Saves MovieGroupProcess model and dictionary
        :param folder_path:
        :return:
        '''
        if os.path.isdir(folder_path):
            print(f'Folder {folder_path} already exists, not overwriting it. Exiting!')

        os.mkdir(folder_path)

        with open(Path(folder_path, 'gsdmm.npy'), 'wb') as f:
            np.save(f, self.K)
            np.save(f, self.alpha)
            np.save(f, self.beta)
            np.save(f, self.cluster_doc_count)
            np.save(f, self.cluster_word_count)
            np.save(f, self.cluster_word_distribution)

        self.dictionary.save_as_text(Path(folder_path, 'dictionary.npy'))

    @staticmethod
    def load(folder_path):
        '''
          Loads MovieGroupProcess model and dictionary
           :param folder_path:
           :return:
           MovieGroupProcess class instance with correct weights
           '''

        with open(Path(folder_path, 'gsdmm.npy'), 'rb') as f:
            K = np.load(f)
            alpha = np.load(f)
            beta = np.load(f)
            cluster_doc_count = np.load(f)
            cluster_word_count = np.load(f)
            cluster_word_distribution = np.load(f)

        dictionary = Dictionary.load_from_text(Path(folder_path, 'dictionary.npy'))
        mgp = MovieGroupProcess.from_data(K, alpha, beta, dictionary, cluster_doc_count, cluster_word_count,
                                          cluster_word_distribution)
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
        return [i for i, entry in enumerate(np.random.multinomial(1, p)) if entry != 0][0]

    def create_dictionary(self, docs_toks):
        self.dictionary = Dictionary(docs_toks)
        self.corpus = [self.dictionary.doc2bow(text) for text in docs_toks]
        self.vocab_size = len(self.dictionary)

    def fit(self, docs):
        '''
        Cluster the input documents
        :param docs: list of list
            list of lists containing the unique token set of each document
        :param V: total vocabulary size for each document
        :return: list of length len(doc)
            cluster label for each document
        '''

        self.create_dictionary(docs)

        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, self.vocab_size

        D = len(self.corpus)
        self.number_docs = D

        self.cluster_word_distribution = np.zeros((K, self.vocab_size), dtype=int)

        # unpack to easy var names
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        cluster_count = K
        d_z = [None for i in range(len(self.corpus))]

        # initialize the clusters
        for i, doc in enumerate(self.corpus):
            # choose a random  initial cluster for the doc
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)

            idx, cnt = zip(*doc)
            n_z_w[z, idx] += cnt

        for _iter in range(n_iters):
            total_transfers = 0

            for i, doc in enumerate(self.corpus):

                # remove the doc from it's current cluster
                z_old = d_z[i]

                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)

                idx, cnt = zip(*doc)
                n_z_w[z_old, idx] -= cnt

                # draw sample from distribution to find new cluster
                p = self.score(doc)
                z_new = self._sample(p)

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1

                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                n_z_w[z_new, idx] += cnt

            cluster_count_new = sum([1 for v in m_z if v > 0])
            print("In stage %d: transferred %d clusters with %d clusters populated" % (
                _iter, total_transfers, cluster_count_new))
            if total_transfers == 0 and cluster_count_new == cluster_count and _iter > 25:
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

        lD1 = np.log(D - 1 + K * alpha)
        doc_size = len(doc)
        idx, cnt = zip(*doc)

        lN1 = np.log(m_z + alpha)
        lN2 = np.log(n_z_w[:, idx] + beta).sum(axis=1)
        lD2 = np.log(n_z.reshape(-1, 1) + V * beta + np.array(range(1, doc_size + 1)) - 1).sum(axis=1)
        p = np.exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm > 0 else 1
        return [pp / pnorm for pp in p]

    def choose_best_label(self, doc):
        '''
        Choose the highest probability label for the input document
        :param doc: list[str]: The doc token stream
        :return:
        '''

        doc_corpus = self.dictionary.doc2bow(doc)
        p = self.score(doc_corpus)
        return np.argmax(p), max(p)

    def top_words(self, n_toks=5, join_tok=' '):
        doc_count = np.array(self.cluster_doc_count)
        n_topics_with_docs = sum(doc_count > 0)  # dont need topics where no docs
        top_index = doc_count.argsort()[-n_topics_with_docs:][::-1]
        topic_words = {}
        top_cluster_tok_idx = np.argsort(-self.cluster_word_distribution)[:, :n_toks]

        for cluster in top_index:
            words_ = ''
            for idx in top_cluster_tok_idx[cluster]:
                words_ = f'{words_}{join_tok}{self.dictionary[idx]}'
            topic_words[cluster] = words_
        return topic_words
