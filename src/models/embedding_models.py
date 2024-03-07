import jieba
from gensim.models import KeyedVectors, Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from src.utils.common import log_step
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


class Word2VecRetriever:
    def __init__(
            self,
            model_path: str,
            retrieval_corpus,
            corpus_ids
    ):
        self.retrieval_corpus = retrieval_corpus
        self.corpus_ids = corpus_ids
        self.tokenized_corpus = [jieba.lcut(doc) for doc in retrieval_corpus]

        # # load a pretrained word2vec embedding
        print(f'Loading word2vec model: {model_path.split("/")[-1]}...')
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        self.doc_vectors = self._compute_doc_vectors()

    @log_step
    def search_all(self, queries, top_k):
        results = []
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id, _ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        query_vector = self._compute_query_vector(q)
        similarities = cosine_similarity(query_vector, self.doc_vectors)
        results = sorted(zip(self.corpus_ids, similarities[0]), key=lambda x: x[1], reverse=True)[:top_k]
        return results

    def _compute_query_vector(self, query):
        query_terms = jieba.lcut(query)
        query_vectors = [self.model[t] for t in query_terms if t in self.model]
        if len(query_vectors) > 0:
            return np.mean(query_vectors, axis=0).reshape(1, -1)
        else:
            return np.zeros((1, self.model.vector_size))

    def _compute_doc_vectors(self):
        doc_vectors = []
        for doc in tqdm(self.retrieval_corpus, desc='Computing document vectors'):
            doc_terms = jieba.lcut(doc)
            doc_vectors.append(self._compute_query_vector(' '.join(doc_terms)))
        return np.vstack(doc_vectors)


class TrainedWord2VecRetriever:
    def __init__(self, retrieval_corpus, corpus_ids, vector_size=300, window=5, min_count=1, workers=40):
        self.retrieval_corpus = retrieval_corpus
        self.corpus_ids = corpus_ids
        self.tokenized_corpus = [jieba.lcut(doc) for doc in retrieval_corpus]
        self.model = self._train_word2vec(vector_size, window, min_count, workers)

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    @log_step
    def search_all(self, queries, top_k):
        results = list()
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id, _ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        query_vector = self._vectorize_query(q)
        scores = cosine_similarity([query_vector], self._vectorize_corpus())
        top_k_indices = np.argsort(scores[0])[-top_k:][::-1]
        return [(self.corpus_ids[i], scores[0][i]) for i in top_k_indices]

    def _train_word2vec(self, vector_size, window, min_count, workers):
        return Word2Vec(self.tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=1)

    def _vectorize_query(self, q):
        tokens = jieba.lcut(q)
        vectors = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def _vectorize_corpus(self):
        return np.array([self._vectorize_query(" ".join(doc)) for doc in self.tokenized_corpus])


class T5Retriever:
    def __init__(self, retrieval_corpus, corpus_ids, model_name='t5-small', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.retrieval_corpus = retrieval_corpus
        self.corpus_ids = corpus_ids
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.encoded_corpus = self._encode_corpus()

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    @log_step
    def search_all(self, queries, top_k):
        results = list()
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id, _ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        query_vector = self._encode_query(q)
        # print(query_vector.shape)
        # print(self.encoded_corpus.shape)
        scores = cosine_similarity(query_vector, self.encoded_corpus)
        top_k_indices = np.argsort(scores[0])[-top_k:][::-1]
        return [(self.corpus_ids[i], scores[0][i]) for i in top_k_indices]

    def _encode_corpus(self):
        return np.vstack([self._encode_query(doc) for doc in tqdm(self.retrieval_corpus, desc='Encoding corpus')])

    def _encode_query(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
        with torch.no_grad():
            encoded_query = self.model.encoder(input_ids).last_hidden_state.mean(dim=1).cpu().numpy()
        return encoded_query


class Doc2VecRetriever:
    def __init__(self, retrieval_corpus, corpus_ids, vector_size=300, window=5, min_count=1, workers=40):
        self.retrieval_corpus = retrieval_corpus
        self.corpus_ids = corpus_ids
        self.tokenized_corpus = [jieba.lcut(doc) for doc in retrieval_corpus]
        self.model = self._train_doc2vec(vector_size, window, min_count, workers)

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    @log_step
    def search_all(self, queries, top_k):
        results = list()
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id, _ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        query_vector = self._vectorize_query(q)
        scores = cosine_similarity([query_vector], self._vectorize_corpus())
        top_k_indices = np.argsort(scores[0])[-top_k:][::-1]
        return [(self.corpus_ids[i], scores[0][i]) for i in top_k_indices]

    def _train_doc2vec(self, vector_size, window, min_count, workers):
        tagged_documents = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(self.tokenized_corpus)]
        model = Doc2Vec(tagged_documents, vector_size=vector_size, window=window, min_count=min_count, workers=workers, dm=1)
        return model

    def _vectorize_query(self, q):
        tokens = jieba.lcut(q)
        return self.model.infer_vector(tokens)

    def _vectorize_corpus(self):
        return np.array([self.model.docvecs[i] for i in range(len(self.tokenized_corpus))])


class BERTRetriever:
    def __init__(self, retrieval_corpus, corpus_ids, model_name='bert-base-uncased', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.retrieval_corpus = retrieval_corpus
        self.corpus_ids = corpus_ids
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoded_corpus = self._encode_corpus()

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    @log_step
    def search_all(self, queries, top_k):
        results = list()
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id, _ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        query_vector = self._encode_query(q)
        scores = cosine_similarity(query_vector, self.encoded_corpus)
        top_k_indices = np.argsort(scores[0])[-top_k:][::-1]
        return [(self.corpus_ids[i], scores[0][i]) for i in top_k_indices]

    def _encode_corpus(self):
        return np.vstack([self._encode_query(doc) for doc in tqdm(self.retrieval_corpus, desc='Encoding corpus')])

    def _encode_query(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            encoded_query = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return encoded_query


