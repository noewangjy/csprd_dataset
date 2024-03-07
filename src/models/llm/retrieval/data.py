import collections
import json
import logging
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional
import pandas as pd
import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from src.models.masked_qa.retrieval.arguments import DataArguments


def read_mapping_id(id_file):
    id_dict = {}
    for line in open(id_file, encoding='utf-8'):
        id, offset = line.strip().split('\t')
        id_dict[id] = int(offset)
    return id_dict


def read_train_file(train_file):
    train_data = []
    for line in open(train_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        pos = line[1].split(',')
        train_data.append((qid, pos))
    return train_data


def read_neg_file(neg_file):
    neg_data = collections.defaultdict(list)
    for line in open(neg_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        neg = line[1].split(',')
        neg_data[qid].extend(neg)
    return neg_data


def read_test_file(test_file, prediction_topk):
    test_data = []
    for line in open(test_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        rank = int(line[-1])
        if rank <= prediction_topk:
            test_data.append((line[0], line[1]))
    return test_data


def read_teacher_score(score_files):
    teacher_score = collections.defaultdict(dict)
    for file in score_files.split(','):
        if not os.path.exists(file):
            logging.info(f"There is no score file:{file}, skip reading the score")
            return None
        for line in open(file):
            qid, did, score = line.strip().split()
            score = float(score.strip('[]'))
            teacher_score[qid][did] = score
    return teacher_score


def generate_random_neg(qids, pids, k=30):
    print(len(qids), len(pids))
    qid_negatives = {}
    for q in qids:
        negs = random.sample(pids, k)
        qid_negatives[q] = negs
    return qid_negatives


class TrainDatasetForBiE(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        self.corpus_dataset = datasets.Dataset.load_from_disk(args.corpus_file)
        self.query_dataset = datasets.Dataset.load_from_disk(args.train_query_file)
        self.train_qrels = read_train_file(args.train_qrels)
        self.corpus_id = read_mapping_id(args.corpus_id_file)
        self.query_id = read_mapping_id(args.train_query_id_file)

        if args.neg_file:
            self.train_negative = read_neg_file(args.neg_file)
        else:
            self.train_negative = generate_random_neg(list(self.query_id.keys()), list(self.corpus_id.keys()))

        self.teacher_score = None
        if args.teacher_score_files is not None:
            self.teacher_score = read_teacher_score(args.teacher_score_files)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.train_qrels)

    def __len__(self):
        return self.total_len

    def create_query_example(self, id: Any):
        item = self.tokenizer.encode_plus(
            self.query_dataset[self.query_id[id]]['input_ids'],
            truncation='only_first',
            max_length=self.args.query_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def create_passage_example(self, id: Any):
        item = self.tokenizer.encode_plus(
            self.corpus_dataset[self.corpus_id[id]]['input_ids'],
            truncation='only_first',
            max_length=self.args.passage_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding], Optional[List[int]]]:
        group = self.train_qrels[item]

        qid = group[0]
        query = self.create_query_example(qid)

        teacher_scores = None
        passages = []
        pos_id = random.choice(group[1])
        passages.append(self.create_passage_example(pos_id))
        if self.teacher_score:
            teacher_scores = []
            teacher_scores.append(self.teacher_score[qid][pos_id])

        query_negs = self.train_negative[qid][:self.args.sample_neg_from_topk]
        if len(query_negs) < self.args.train_group_size - 1:
            negs = random.sample(self.corpus_id.keys(), k=self.args.train_group_size - 1 - len(query_negs))
            negs.extend(query_negs)
        else:
            negs = random.sample(query_negs, k=self.args.train_group_size - 1)

        if self.teacher_score:
            passages, teacher_scores = [], []
            item_scores = [(pos_id, self.teacher_score[qid][pos_id])]
            for id in negs:
                item_scores.append((id, self.teacher_score[qid][id]))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            for id, score in item_scores:
                passages.append(self.create_passage_example(id))
                teacher_scores.append(score)
        else:
            for id in negs:
                passages.append(self.create_passage_example(id))
                if self.teacher_score:
                    teacher_scores.append(self.teacher_score[qid][id])

        return query, passages, teacher_scores


class PredictionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_len=512):
        self.encode_data = datasets.Dataset.load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            self.encode_data[item]['input_ids'],
            truncation='only_first',
            max_length=self.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item


class InferenceDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            dataset_path: str,
            tokenizer: PreTrainedTokenizer,
            max_len=512
    ):
        self.dataset = {}
        self.id2key = {}
        if os.path.exists(os.path.join(dataset_path, "train_set.tsv")):
            train_set = pd.read_csv(os.path.join(dataset_path, "train_set.tsv"), sep='\t')
            for i in range(len(train_set)):
                self.dataset[i] = train_set["passage"][i]
                self.id2key[str(train_set["id"][i])] = i
        if os.path.exists(os.path.join(dataset_path, "dev_set.tsv")):
            dev_set = pd.read_csv(os.path.join(dataset_path, "dev_set.tsv"), sep='\t')
            offset = len(self.dataset)
            for i in range(len(dev_set)):
                self.dataset[i + offset] = dev_set["passage"][i]
                self.id2key[str(dev_set["id"][i])] = i + offset

        self.test_id = []
        with open(input_path, 'r') as f:
            for line in f:
                self.test_id.append(line.strip())

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.test_id)

    def __getitem__(self, item) -> BatchEncoding:
        key = self.id2key[self.test_id[item]]
        sample = self.tokenizer.encode_plus(
            self.dataset[key],
            truncation='only_first',
            max_length=self.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return sample


class CustomInferenceDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            tokenizer: PreTrainedTokenizer,
            max_len=512
    ):
        self.dataset = []
        with open(input_path, "r") as f:
            dataset = json.load(f)
        idx = 0
        for filename in dataset:
            for title in dataset[filename]:
                self.dataset.append(dataset[filename][title])
                idx += 1

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> BatchEncoding:
        sample = self.tokenizer.encode_plus(
            self.dataset[item],
            truncation='only_first',
            max_length=self.max_len,
            padding=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return sample





















@dataclass
class BiCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 512
    passage_max_len: int = 512

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        teacher_score = [f[2] for f in features]
        if teacher_score[0] is None:
            teacher_score = None
        else:
            teacher_score = torch.FloatTensor(teacher_score)

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer.pad(
            query,
            padding='max_length',
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            passage,
            padding='max_length',
            max_length=self.passage_max_len,
            return_tensors="pt",
        )

        return {"query": q_collated, "passage": d_collated, "teacher_score": teacher_score}


@dataclass
class PredictionCollator(DataCollatorWithPadding):
    is_query: bool = True

    def __call__(
            self, features
    ):
        if self.is_query:
            # print(features)
            return {"query": super().__call__(features), "passage": None}
        else:
            return {"query": None, "passage": super().__call__(features)}
