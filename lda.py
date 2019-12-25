'''
The code is based on the following papers. The linked repositories were referred to during implementation as well.

Papers:

David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. In Advances in neural information processing systems, pages 601–608, 2002.

William M Darling. A theoretical and practical implementation tutorial on topic modeling and gibbs sampling. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies, pages 642–647, 2011.

Thomas L Griﬃths and Mark Steyvers. Finding scientiﬁc topics. Proceedings of the National academy of Sciences, 101(suppl 1):5228–5235, 2004.

Daniel Ramage, David Hall, Ramesh Nallapati, and Christopher D Manning. Labeled lda: A supervised topic model for credit attribution in multi-labeled corpora. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1-Volume 1, pages 248–256. Association for Computational Linguistics, 2009.


Repositories:

https://github.com/wiseodd/probabilistic-models
https://github.com/lda-project/lda
https://gist.github.com/mblondel/542786
https://github.com/kzhai/PyLDA
'''

import numpy as np
from scipy.special import logsumexp, gammaln
# from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

class LDA():
    ''' Implementation of LDA with a Collapsed Gibb's Sampler for posterior inference. '''
    def __init__(self, alpha, eta, num_topics, vocab_size=0, num_docs=0):
        # Data Properties
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.num_docs = num_docs

        # Data
        self.corpus = None
        self.corpus_model = CountVectorizer(stop_words='english')

        # Hyperparameters
        self.alpha = alpha # hyperparameter for theta_i (document-topic distribution)
        self.eta = eta # hyperparameter for eta_k (vocabulary-topic distribution)

        # Count Variables
        self.n_dk = None
        self.n_wk = None
        self.n_k = np.zeros(num_topics)
        self.docs_len = None

        # Assignments
        self.z = None


    def random_initialization(self, corpus):
        self._load_corpus(corpus)

        for i, doc in enumerate(corpus):
            word_inds = np.nonzero(self.X[i])[1]
            # for j, word in enumerate(doc):
            for j in word_inds:
                # Choose a random topic assignment
                rand_topic = np.random.randint(0, self.num_topics)

                # Add random assignment to z_ij
                self.z[i, j] = rand_topic

                # Increment respective counts for random assignment
                self.n_dk[i, rand_topic] += 1
                self.n_wk[j, rand_topic] += 1
                self.n_k[rand_topic] += 1
                self.docs_len[i] += 1
        

    def calc_log_likelihood(self):
        # Calculate p(w|z)
        # = (gamma(self.vocab_size * self.eta) / gamma(self.eta) ** self.vocab_size) ** self.num_topics

        log_likelihood = gammaln(self.vocab_size * self.eta)
        log_likelihood -= gammaln(self.eta) * self.vocab_size
        log_likelihood *= self.num_topics
        for k in range(0, self.num_topics):
            for j in range(0, self.vocab_size):
                # *= gamma(self.n_wk[j, k] + self.eta)

                log_likelihood += gammaln(self.n_wk[j, k] + self.eta)

            # /= gamma(self.n_wk[:, k] + self.vocab_size * self.eta)

            log_likelihood -= gammaln(np.sum(self.n_wk[:, k]) + (self.vocab_size + self.eta))


        # Calculate P(z)
        # *= (gamma(self.num_topics * self.alpha) / gamma(alpha) ** self.num_topics) ** self.num_docs

        log_likelihood += gammaln(self.num_topics * self.alpha)
        log_likelihood -= gammaln(self.alpha) * self.num_topics
        log_likelihood *= self.num_docs

        for i in range(0, self.num_docs):
            for k in range(0, self.num_topics):
                # *= gamma(self.n_dk[i, k] + self.alpha)

                log_likelihood += gammaln(self.n_dk[i, k] + self.alpha)

            # /= gamma(self.n_dk[i, :] + self.num_topics * self.alpha)
            log_likelihood -= gammaln(np.sum(self.n_dk[i, :]) + (self.num_topics * self.alpha))
        
        return log_likelihood

    def calc_beta(self):
        ''' Calculate distribution over words and topics given current parameters. '''
        log_beta = np.log(self.n_wk + self.eta)
        log_beta -= np.log(np.sum(self.n_wk) + self.eta * self.vocab_size)

        return np.exp(log_beta)

    def calc_theta(self):
        ''' Calculate distribution over documents and topics given current parameters. '''
        log_theta = np.log(self.n_dk + self.alpha)
        log_theta -= np.log(np.sum(self.n_dk) + self.alpha * self.num_topics)

        return np.exp(log_theta)

    def fit(self, max_iters=1000):
        ''' An implementation of the collapsed Gibb's Sampler for LDA. '''
        lls = []
        for iters in range(0, max_iters):
            if iters % 100 == 0: print(iters)
            for i in range(0, self.num_docs):
                # print(self.X[i])
                word_inds = np.nonzero(self.X[i])[1]
                # for j in range(0, self.docs_len[i]):
                for j in word_inds:
                    # Get existing topic assignment (k)
                    k = self.z[i, j]
                    # Decrement counts for this topic
                    self.n_dk[i, k] -= 1
                    self.n_wk[j, k] -= 1
                    self.n_k[k] -= 1

                    # Calculate complete conditional of z_ij
                    log_prob = np.log(self.n_wk[j, :] + self.eta)
                    log_prob -= np.log(self.n_k[:] + self.eta * self.vocab_size)
                    log_prob += np.log(self.n_dk[i, :] + self.alpha)
                    log_prob -= np.log(self.docs_len[i] + self.alpha * self.num_topics)

                    log_prob -= logsumexp(log_prob)

                    # Sample new topic using newly calculated probabilities
                    prob = np.exp(log_prob)
                    new_k = np.random.multinomial(1, prob)
                    new_k = new_k.argmax()

                    # Increment counts for new topic
                    self.n_dk[i, new_k] += 1
                    self.n_wk[j, new_k] += 1
                    self.n_k[new_k] += 1

                    # Assign z_ij to new topic
                    self.z[i, j] = new_k
            lls.append(self.calc_log_likelihood())

        return lls
    
    
    def _load_corpus(self, corpus, remove_stopwords=True):
        ''' 
        Corpus should be an iterable of strings. 

        Determines basic corpus information. 
        Initializes (empty) count variables and `self.z`.

        TODO: Not currently handling stopwords
        '''
        self.X = self.corpus_model.fit_transform(corpus)

        self.num_docs = self.X.shape[0]
        self.vocab_size = self.X.shape[1]
        self.corpus = corpus

        self.n_dk = np.zeros(shape=(self.num_docs, self.num_topics), dtype=int)
        self.n_wk = np.zeros(shape=(self.vocab_size, self.num_topics), dtype=int)
        self.n_k = np.zeros(self.num_topics, dtype=int)
        self.docs_len = np.zeros(self.num_docs, dtype=int)

        self.z = np.zeros(shape=(self.num_docs, self.vocab_size), dtype=int)
    