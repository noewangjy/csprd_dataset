import sys
import logging
import itertools
from tqdm import tqdm
import jieba
import math
import numpy as np
from statistics import mean
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.common import log_step


class TFIDFRetriever:
    def __init__(self, retrieval_corpus, corpus_ids):
        self.retrieval_corpus = retrieval_corpus
        self.corpus_ids = corpus_ids
        self.N = len(retrieval_corpus)
        self.vocab = self._build_vocabulary()
        self.idfs = self._compute_idfs()

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    @log_step
    def search_all(self, queries, top_k):
        results = list()
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id, _ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        results = dict()
        for i, doc in zip(self.corpus_ids, self.retrieval_corpus):
            results[i] = self.score(q, doc)  # NB: '+1' because doc_ids in BSARD start at 1.
        return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def score(self, q, d):
        score = 0.0
        for t in jieba.lcut(q):
            score += self._compute_tfidf(t, d)
        return score

    def _build_vocabulary(self):
        return sorted(set(itertools.chain.from_iterable([jieba.lcut(doc) for doc in tqdm(self.retrieval_corpus)])))

    def _compute_idfs(self):
        idfs = dict.fromkeys(self.vocab, 0)
        for word, _ in idfs.items():
            idfs[word] = self._compute_idf(word)
        return idfs

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10(self.N / (df + 1))

    def _compute_tf(self, t, d):
        return d.count(t)

    def _compute_tfidf(self, t, d):
        tf = self._compute_tf(t, d)
        idf = self.idfs[t] if t in self.idfs else math.log10(self.N)
        return tf * idf


class BM25Retriever(TFIDFRetriever):
    def __init__(self, retrieval_corpus, corpus_ids, k1, b):
        super().__init__(retrieval_corpus, corpus_ids)
        self.k1 = k1
        self.b = b
        self.avgdl = self._compute_avgdl()
        self.data_map = {}
        for doc in tqdm(self.retrieval_corpus):
            self.data_map[doc] = jieba.lcut(doc)

    def score(self, q, d):
        score = 0.0
        for t in jieba.lcut(q):
            tf = self._compute_tf(t, d)
            idf = self.idfs[t] if t in self.idfs else math.log10((self.N + 0.5) / 0.5)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * len(self.data_map[d]) / self.avgdl))
        return score

    def _compute_avgdl(self):
        return mean([len(jieba.lcut(doc)) for doc in self.retrieval_corpus])

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10((self.N - df + 0.5) / (df + 0.5))


# class SWSNRetriever(BM25Retriever):
#     def __init__(self, retrieval_corpus, k1, b, model):
#         super().__init__(retrieval_corpus, k1, b)
#         self.model = model
#
#     def score(self, q, d):
#         score = 0.0
#         for t in d.split():
#             sem = self._compute_sem(t, q)
#             idf = self.idfs[t] if t in self.idfs else math.log10((self.N + 0.5) / 0.5)
#             score += idf * (sem * (self.k1 + 1)) / (sem + self.k1 * (1 - self.b + self.b * len(q.split()) / self.avgdl))
#         return score
#
#     def _compute_sem(self, t, q):
#         term_embedding = self._get_word_embedding(t)
#         query_embeddings = [self._get_word_embedding(w) for w in q.split()]
#         cosines = [cosine_similarity([term_embedding], [embedding])[0, 0] for embedding in query_embeddings]
#         return np.max(cosines)
#
#     def _get_word_embedding(self, w):
#         return self.model[w] if w in self.model.vocab else np.zeros(self.model.vector_size)
