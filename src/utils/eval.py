import itertools
from typing import List, Dict, Optional, Type
import numpy as np
from torch import nn
from statistics import mean
from sentence_transformers import util
from torch.utils.tensorboard import SummaryWriter


class Evaluator:
    def __init__(self, metrics_at_k: Dict[str, List[int]] = {
        'recall': [1, 5, 10, 50], 'map': [1, 5, 10, 50], 'mrr': [1, 5, 10, 50], 'ndcg': [1, 5, 10, 50], 'precision': [1, 5, 10, 50]
    }):
        self.metrics_at_k = metrics_at_k

    def compute_all_metrics(self, all_results: List[List[int]], all_ground_truths: List[List[int]]):
        scores = dict()
        for k in self.metrics_at_k['recall']:
            recall_scalar = self.compute_mean_score(self.recall, all_ground_truths, all_results, k)
            scores[f'recall@{k}'] = recall_scalar

        for k in self.metrics_at_k['map']:
            map_scalar = self.compute_mean_score(self.average_precision, all_ground_truths, all_results, k)
            scores[f'map@{k}'] = map_scalar

        for k in self.metrics_at_k['mrr']:
            mrr_scalar = self.compute_mean_score(self.reciprocal_rank, all_ground_truths, all_results, k)
            scores[f'mrr@{k}'] = mrr_scalar

        for k in self.metrics_at_k['ndcg']:
            ndcg_scalar = self.compute_mean_score(self.ndcg, all_ground_truths, all_results, k)
            scores[f'ndcg@{k}'] = ndcg_scalar

        for k in self.metrics_at_k['precision']:
            ndcg_scalar = self.compute_mean_score(self.ndcg, all_ground_truths, all_results, k)
            scores[f'precision@{k}'] = ndcg_scalar


        return scores

    def compute_mean_score(self, func, all_ground_truths: List[List[int]], all_results: List[List[int]], k: int = None):
        return mean([func(truths, res, k) for truths, res in zip(all_ground_truths, all_results)])

    def precision(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances) / len(results[:k])

    def recall(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances) / len(ground_truths)

    def fscore(self, ground_truths: List[int], results: List[int], k: int = None):
        p = self.precision(ground_truths, results, k)
        r = self.recall(ground_truths, results, k)
        return (2 * p * r) / (p + r) if (p != 0.0 or r != 0.0) else 0.0

    def reciprocal_rank(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        return max([1 / (i + 1) if d in ground_truths else 0.0 for i, d in enumerate(results[:k])])

    def average_precision(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        p_k = [self.precision(ground_truths, results, k=i + 1) if d in ground_truths else 0 for i, d in
               enumerate(results[:k])]
        return sum(p_k) / len(ground_truths)

    def ndcg(self, ground_truth, results, k):
        dcg_val = self.dcg(ground_truth, results, k)
        idcg_val = self.idcg(ground_truth, k)
        if idcg_val == 0:
            return 0
        return dcg_val / idcg_val

    def dcg(self, ground_truth, results, k):
        dcg = 0
        for i, item in enumerate(results[:k]):
            if item in ground_truth:
                dcg += 1 / (np.log2(i + 2))  # +2 because index starts from 0, and log2(1) is 0.
        return dcg

    def idcg(self, ground_truth, k):
        idcg = 0
        for i in range(min(k, len(ground_truth))):
            idcg += 1 / (np.log2(i + 2))
        return idcg


class BiEncoderEvaluator(Evaluator):
    def __init__(self,
                 queries: Dict[int, str],  # qid -> query
                 documents: Dict[int, str],  # doc_id -> doc
                 relevant_pairs: Dict[int, List[int]],  # qid -> List[doc_id]
                 score_fn: str,
                 metrics_at_k: Dict[str, List[int]] = {'recall': [100, 200, 500], 'map': [100], 'mrr': [100]},
                 ):
        super().__init__(metrics_at_k)
        assert score_fn in ['dot', 'cos'], f"Unknown score function: {score_fn}"
        self.score_fn = util.dot_score if score_fn == 'dot' else util.cos_sim
        self.query_ids = list(queries.keys())
        self.queries = [queries[qid] for qid in self.query_ids]
        self.document_ids = list(documents.keys())
        self.documents = [documents[doc_id] for doc_id in self.document_ids]
        self.relevant_pairs = relevant_pairs

    def __call__(self,
                 model: Type[nn.Module],
                 device: str,
                 batch_size: int,
                 writer: Optional[Type[SummaryWriter]] = None,
                 epoch: Optional[int] = None
                 ):
        # Encode queries.
        q_embeddings = model.q_encoder.encode(texts=self.queries, device=device, batch_size=batch_size)
        d_embeddings = model.d_encoder.encode(texts=self.documents, device=device, batch_size=batch_size)

        # Retrieve top candidates -> returns a List[List[Dict[str,int]]].
        all_results = util.semantic_search(
            query_embeddings=q_embeddings,
            corpus_embeddings=d_embeddings,
            top_k=max(list(itertools.chain(*self.metrics_at_k.values()))),
            score_function=self.score_fn)
        all_results = [[result['corpus_id'] + 1 for result in results] for results in
                       all_results]  # Extract the doc_id only -> List[List[int]] (NB: +1 because article ids start at 1 while semantic_search returns indices in the given list).

        # Get ground truths.
        all_ground_truths = [self.relevant_pairs[qid] for qid in self.query_ids]

        # Compute metrics.
        scores = dict()
        for k in self.metrics_at_k['recall']:
            recall_scalar = self.compute_mean_score(self.recall, all_ground_truths, all_results, k)
            if writer is not None:
                writer.add_scalar(f'Val/recall/recall_at_{k}', recall_scalar, epoch)
            scores[f'recall@{k}'] = recall_scalar

        for k in self.metrics_at_k['map']:
            map_scalar = self.compute_mean_score(self.average_precision, all_ground_truths, all_results, k)
            if writer is not None:
                writer.add_scalar(f'Val/map/map_at_{k}', map_scalar, epoch)
            scores[f'map@{k}'] = map_scalar

        for k in self.metrics_at_k['mrr']:
            mrr_scalar = self.compute_mean_score(self.reciprocal_rank, all_ground_truths, all_results, k)
            if writer is not None:
                writer.add_scalar(f'Val/mrr/mrr_at_{k}', mrr_scalar, epoch)
            scores[f'mrr@{k}'] = mrr_scalar

        return scores
