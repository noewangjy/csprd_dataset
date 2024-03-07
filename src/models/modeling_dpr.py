import torch
import torch.nn.functional as F
from torch import Tensor as T
import pytorch_lightning as pl

from transformers import AutoConfig, AutoTokenizer, AutoModel
import random

import numpy as np
from collections import namedtuple
from omegaconf import DictConfig
from src.utils.dataset import DPRSample
from typing import List, Tuple, Dict, Optional, Union, Any, Optional, Union, Callable, Iterable


BiEncoderBatch = namedtuple(
    'BiEncoderBatch',
    [
        'query',
        'context',
        'positive_indices',
        'hard_negative_indices',
    ]
)


def dot_product_scores(q_vectors: T, p_vectors: T):
    """
    calculates the dot product scores between query vectors and passage vectors
    :param q_vectors: batch_size x hidden_size
    :param p_vectors: batch_size x hidden_size
    :return result: batch_size x batch_size
    """
    result = torch.matmul(q_vectors, torch.transpose(p_vectors, 0, 1))
    return result


class PassageMatchingModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.query_tokenizer = AutoTokenizer.from_pretrained(self.cfg.query_encoder.model_name_or_path)
        self.context_tokenizer = AutoTokenizer.from_pretrained(self.cfg.context_encoder.model_name_or_path)
        self.query_config = AutoConfig.from_pretrained(self.cfg.query_encoder.model_name_or_path)
        self.context_config = AutoConfig.from_pretrained(self.cfg.context_encoder.model_name_or_path)
        self.query_encoder = AutoModel.from_pretrained(self.cfg.query_encoder.model_name_or_path, config=self.query_config)
        self.context_encoder = AutoModel.from_pretrained(self.cfg.context_encoder.model_name_or_path, config=self.context_config)
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay
        )
        return optimizer

    def forward(
            self,
            query_ids: T,
            query_segments: T,
            query_attention_mask: T,
            context_ids: T,
            context_segments: T,
            context_attention_mask: T,
            labels: Optional[T] = None,
            hard_neg_labels: Optional[T] = None
    ):
        query_output = self.query_encoder(
            input_ids=query_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_segments
        )
        context_output = self.context_encoder(
            input_ids=context_ids,
            attention_mask=context_attention_mask,
            token_type_ids=context_segments
        )
        query_vectors = query_output.last_hidden_state[:, 0, :]
        context_vectors = context_output.last_hidden_state[:, 0, :]
        scores = dot_product_scores(query_vectors, context_vectors)
        scores = F.softmax(scores, dim=1)
        loss = None
        # We add double w
        loss_weight = torch.ones(scores.size(1)).div(scores.size(1) + 1).to(scores.device)
        if hard_neg_labels is not None:
            loss_weight[hard_neg_labels] = loss_weight[hard_neg_labels].mul(2)

        if labels is not None:
            loss = F.cross_entropy(scores, labels, weight=loss_weight)
        return scores, loss

    def create_batch(self, batch: DPRSample, shuffle: bool = False) -> BiEncoderBatch:
        query = self.query_tokenizer(
            batch.query,
            padding='max_length',
            truncation=True,
            max_length=self.cfg.query_encoder.max_length,
            return_tensors='pt'
        )

        all_contexts = list()
        positive_ctx_indices = list()
        hard_neg_ctx_indices = list()
        for sample_idx in range(len(batch.positive_context)):
            positive_ctx_indices.append(len(all_contexts))
            all_contexts.append(batch.positive_context[sample_idx])

            if shuffle:
                random.shuffle(batch.negative_contexts[sample_idx])
                random.shuffle(batch.hard_negative_contexts[sample_idx])

            all_contexts.extend(batch.negative_contexts[: self.cfg.train.num_negatives])
            hard_neg_ctx_indices.extend(
                [len(all_contexts) - 1 + i for i in range(self.cfg.train.num_hard_negatives)]
            )
            all_contexts.extend(batch.hard_negative_contexts[: self.cfg.train.num_hard_negatives])

        context = self.context_tokenizer(
            all_contexts,
            padding='max_length',
            truncation=True,
            max_length=self.cfg.context_encoder.max_length,
            return_tensors='pt'
        )

        return BiEncoderBatch(
            query=query,
            context=context,
            positive_indices=positive_ctx_indices,
            hard_negative_indices=hard_neg_ctx_indices
        )

    def training_step(self, batch: DPRSample, batch_idx: int):
        optimizer = self.optimizers(use_pl_optimizer=True)

        batch = self.create_batch(batch, shuffle=True)
        scores, loss = self(
            query_ids=batch.query.input_ids,
            query_segments=batch.query.token_type_ids,
            query_attention_mask=batch.query.attention_mask,
            context_ids=batch.context.input_ids,
            context_segments=batch.context.token_type_ids,
            context_attention_mask=batch.context.attention_mask,
            labels=torch.tensor(batch.positive_indices).to(self.device),
            hard_neg_labels=torch.tensor(batch.hard_negative_indices).to(self.device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = torch.sum(torch.argmax(scores, dim=1) == torch.tensor(batch.positive_indices).to(self.device)).float() / len(batch.positive_indices)

        self.log('train_loss', loss.detach().cpu(), prog_bar=False)
        self.log('train_accuracy', accuracy.detach().cpu(), prog_bar=False)

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def validation_step(self, batch: DPRSample, batch_idx: int):
        batch = self.create_batch(batch, shuffle=False)
        scores, loss = self(
            query_ids=batch.query.input_ids,
            query_segments=batch.query.token_type_ids,
            query_attention_mask=batch.query.attention_mask,
            context_ids=batch.context.input_ids,
            context_segments=batch.context.token_type_ids,
            context_attention_mask=batch.context.attention_mask,
            labels=torch.tensor(batch.positive_indices).to(self.device),
            hard_neg_labels=torch.tensor(batch.hard_negative_indices).to(self.device)
        )
        accuracy = torch.sum(torch.argmax(scores, dim=1) == torch.tensor(batch.positive_indices).to(self.device)).float() / len(batch.positive_indices)
        self.log('val_loss', loss.detach().cpu(), prog_bar=False)
        self.log('val_accuracy', accuracy.detach().cpu(), prog_bar=False)
        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def training_epoch_end(self, outputs: List[Dict]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, prog_bar=True)
        self.log('avg_train_acc', avg_accuracy, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True)
        self.log('avg_val_acc', avg_accuracy, prog_bar=True)













