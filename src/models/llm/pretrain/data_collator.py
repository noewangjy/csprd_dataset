import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List
import torch
import torch.utils.data.dataset
from datasets import Dataset
from transformers import DataCollatorForWholeWordMask
from tqdm import tqdm
from src.utils.special_tokens import *
from src.models.masked_qa.pretrain.utils import tensorize_batch


class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.dataset = Dataset.load_from_disk(data_dir)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


@dataclass
class MaskedQACollatorForChinese(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15
    answer_mlm_probability: float = 0.50

    def get_wwm_with_prob(self, tokens: List[str], prob: float):
        self.mlm_probability = prob
        return self._whole_word_mask(tokens)

    def __call__(self, examples):
        p_ids_batch = []
        p_attention_mask_batch = []
        p_encoder_mlm_mask_batch = []
        p_decoder_labels_batch = []
        p_decoder_matrix_attention_mask_batch = []

        qa_ids_batch = []
        qa_attention_mask_batch = []
        qa_decoder_labels_batch = []
        qa_decoder_matrix_attention_mask_batch = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for example in examples:
            all_passage_tokens = example["passage_tokens"]
            query_tokens = example["query_tokens"]
            answer_tokens = example["answer_tokens"]
            blank_flag = BLANK in all_passage_tokens

            passage_tokens = all_passage_tokens[:tgt_len]
            if blank_flag and BLANK not in passage_tokens:
                passage_tokens = all_passage_tokens[-tgt_len:]

            passage_tokens = [self.tokenizer.cls_token] + passage_tokens + [self.tokenizer.sep_token]
            passage_ids = self.tokenizer.convert_tokens_to_ids(passage_tokens)

            passage_encoder_mlm_mask = self.get_wwm_with_prob(passage_tokens, self.encoder_mlm_probability)

            mask_set = []
            for _ in range(min(256, len(passage_tokens))):
                mask_set.append(self.get_wwm_with_prob(passage_tokens, self.decoder_mlm_probability))
            passage_matrix_attention_mask = []
            for i in range(len(passage_tokens)):
                idx = random.randint(0, len(mask_set) - 1)
                passage_decoder_mlm_mask = deepcopy(mask_set[idx])
                passage_decoder_mlm_mask[i] = 1
                passage_matrix_attention_mask.append(passage_decoder_mlm_mask)

            p_ids_batch.append(torch.tensor(passage_ids))
            p_attention_mask_batch.append(torch.tensor([1] * len(passage_ids)))
            passage_ids[0] = -100
            passage_ids[-1] = -100
            p_decoder_labels_batch.append(torch.tensor(passage_ids))
            p_encoder_mlm_mask_batch.append(torch.tensor(passage_encoder_mlm_mask))
            p_decoder_matrix_attention_mask_batch.append(1 - torch.tensor(passage_matrix_attention_mask))

            tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(True)
            query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
            answer_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)

            qa_ids = [self.tokenizer.cls_token_id] + query_ids + [self.tokenizer.sep_token_id] + answer_ids + [
                self.tokenizer.sep_token_id]
            if len(qa_ids) > self.max_seq_length:
                qa_ids = qa_ids[:self.max_seq_length - 1] + [self.tokenizer.sep_token_id]

            mask_set = []
            for _ in range(min(256, len(qa_ids))):
                q_wwm_mask = self.get_wwm_with_prob(query_tokens, 0.0)
                a_wwm_mask = self.get_wwm_with_prob(answer_tokens, self.answer_mlm_probability)
                qa_wwm_mask = [0] + q_wwm_mask + [0] + a_wwm_mask + [0]
                if len(qa_wwm_mask) > self.max_seq_length:
                    qa_wwm_mask = qa_wwm_mask[:self.max_seq_length - 1] + [0]
                mask_set.append(qa_wwm_mask)

            qa_decoder_matrix_attention_mask = []
            for i in range(len(qa_ids)):
                idx = random.randint(0, len(mask_set) - 1)
                qa_decoder_mlm_mask = deepcopy(mask_set[idx])
                qa_decoder_mlm_mask[i] = 1
                qa_decoder_matrix_attention_mask.append(qa_decoder_mlm_mask)

            qa_ids_batch.append(torch.tensor(qa_ids))
            qa_ids[0] = -100
            qa_ids[-1] = -100
            qa_decoder_labels_batch.append(torch.tensor(qa_ids))
            qa_decoder_matrix_attention_mask_batch.append(1 - torch.tensor(qa_decoder_matrix_attention_mask))

        p_ids_batch = tensorize_batch(p_ids_batch, self.tokenizer.pad_token_id)
        p_attention_mask_batch = tensorize_batch(p_attention_mask_batch, 0)
        decoder_p_ids_batch = p_ids_batch.clone()
        p_encoder_mlm_mask_batch = tensorize_batch(p_encoder_mlm_mask_batch, 0)
        p_encoder_input_ids_batch, p_encoder_labels_batch = self.torch_mask_tokens(p_ids_batch,
                                                                                   p_encoder_mlm_mask_batch)
        p_decoder_labels_batch = tensorize_batch(p_decoder_labels_batch, -100)
        p_decoder_matrix_attention_mask_batch = tensorize_batch(p_decoder_matrix_attention_mask_batch, 0)

        qa_ids_batch = tensorize_batch(qa_ids_batch, self.tokenizer.pad_token_id)
        qa_decoder_labels_batch = tensorize_batch(qa_decoder_labels_batch, -100)
        qa_decoder_matrix_attention_mask_batch = tensorize_batch(qa_decoder_matrix_attention_mask_batch, 0)

        batch = {
            "p_encoder_input_ids": p_encoder_input_ids_batch,
            "p_encoder_attention_mask": p_attention_mask_batch,
            "p_encoder_labels": p_encoder_labels_batch,
            "p_decoder_input_ids": decoder_p_ids_batch,
            "p_decoder_attention_mask": p_decoder_matrix_attention_mask_batch,  # [B,L,L]
            "p_decoder_labels": p_decoder_labels_batch,
            "qa_decoder_input_ids": qa_ids_batch,
            "qa_decoder_attention_mask": qa_decoder_matrix_attention_mask_batch,  # [B,L,L]
            "qa_decoder_labels": qa_decoder_labels_batch,
        }

        return batch


@dataclass
class MaskedQACollatorForEnglish(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15
    answer_mlm_probability: float = 0.50

    def get_wwm_with_prob(self, tokens: List[str], prob: float):
        self.mlm_probability = prob
        return self._whole_word_mask(tokens)

    def __call__(self, examples):
        p_ids_batch = []
        p_attention_mask_batch = []
        p_encoder_mlm_mask_batch = []
        p_decoder_labels_batch = []
        p_decoder_matrix_attention_mask_batch = []

        qa_ids_batch = []
        qa_decoder_labels_batch = []
        qa_decoder_matrix_attention_mask_batch = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for example in examples:
            all_passage_tokens = example["passage_tokens"]
            query_tokens = example["query_tokens"]
            answer_tokens = example["answer_tokens"]
            blank_flag = "[UNK]" in all_passage_tokens

            passage_tokens = all_passage_tokens[:tgt_len]
            if blank_flag and "[UNK]" not in passage_tokens:
                passage_tokens = all_passage_tokens[-tgt_len:]

            passage_tokens = [self.tokenizer.cls_token] + passage_tokens + [self.tokenizer.sep_token]
            passage_ids = self.tokenizer.convert_tokens_to_ids(passage_tokens)

            passage_encoder_mlm_mask = self.get_wwm_with_prob(passage_tokens, self.encoder_mlm_probability)

            mask_set = []
            for _ in range(min(256, len(passage_tokens))):
                mask_set.append(self.get_wwm_with_prob(passage_tokens, self.decoder_mlm_probability))
            passage_matrix_attention_mask = []
            for i in range(len(passage_tokens)):
                idx = random.randint(0, len(mask_set) - 1)
                passage_decoder_mlm_mask = deepcopy(mask_set[idx])
                passage_decoder_mlm_mask[i] = 1
                passage_matrix_attention_mask.append(passage_decoder_mlm_mask)

            p_ids_batch.append(torch.tensor(passage_ids))
            p_attention_mask_batch.append(torch.tensor([1] * len(passage_ids)))
            passage_ids[0] = -100
            passage_ids[-1] = -100
            p_decoder_labels_batch.append(torch.tensor(passage_ids))
            p_encoder_mlm_mask_batch.append(torch.tensor(passage_encoder_mlm_mask))
            p_decoder_matrix_attention_mask_batch.append(1 - torch.tensor(passage_matrix_attention_mask))

            tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(True)
            query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
            answer_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)

            qa_ids = [self.tokenizer.cls_token_id] + query_ids + [self.tokenizer.sep_token_id] + answer_ids + [
                self.tokenizer.sep_token_id]
            if len(qa_ids) > self.max_seq_length:
                qa_ids = qa_ids[:self.max_seq_length - 1] + [self.tokenizer.sep_token_id]

            mask_set = []
            for _ in range(min(256, len(qa_ids))):
                q_wwm_mask = self.get_wwm_with_prob(query_tokens, 0.0)
                a_wwm_mask = self.get_wwm_with_prob(answer_tokens, self.answer_mlm_probability)
                qa_wwm_mask = [0] + q_wwm_mask + [0] + a_wwm_mask + [0]
                if len(qa_wwm_mask) > self.max_seq_length:
                    qa_wwm_mask = qa_wwm_mask[:self.max_seq_length - 1] + [0]
                mask_set.append(qa_wwm_mask)

            qa_decoder_matrix_attention_mask = []
            for i in range(len(qa_ids)):
                idx = random.randint(0, len(mask_set) - 1)
                qa_decoder_mlm_mask = deepcopy(mask_set[idx])
                qa_decoder_mlm_mask[i] = 1
                qa_decoder_matrix_attention_mask.append(qa_decoder_mlm_mask)

            qa_ids_batch.append(torch.tensor(qa_ids))
            qa_ids[0] = -100
            qa_ids[-1] = -100
            qa_decoder_labels_batch.append(torch.tensor(qa_ids))
            qa_decoder_matrix_attention_mask_batch.append(1 - torch.tensor(qa_decoder_matrix_attention_mask))

        p_ids_batch = tensorize_batch(p_ids_batch, self.tokenizer.pad_token_id)
        p_attention_mask_batch = tensorize_batch(p_attention_mask_batch, 0)
        decoder_p_ids_batch = p_ids_batch.clone()
        p_encoder_mlm_mask_batch = tensorize_batch(p_encoder_mlm_mask_batch, 0)
        p_encoder_input_ids_batch, p_encoder_labels_batch = self.torch_mask_tokens(p_ids_batch,
                                                                                   p_encoder_mlm_mask_batch)
        p_decoder_labels_batch = tensorize_batch(p_decoder_labels_batch, -100)
        p_decoder_matrix_attention_mask_batch = tensorize_batch(p_decoder_matrix_attention_mask_batch, 0)

        qa_ids_batch = tensorize_batch(qa_ids_batch, self.tokenizer.pad_token_id)
        qa_decoder_labels_batch = tensorize_batch(qa_decoder_labels_batch, -100)
        qa_decoder_matrix_attention_mask_batch = tensorize_batch(qa_decoder_matrix_attention_mask_batch, 0)

        batch = {
            "p_encoder_input_ids": p_encoder_input_ids_batch,
            "p_encoder_attention_mask": p_attention_mask_batch,
            "p_encoder_labels": p_encoder_labels_batch,
            "p_decoder_input_ids": decoder_p_ids_batch,
            "p_decoder_attention_mask": p_decoder_matrix_attention_mask_batch,  # [B,L,L]
            "p_decoder_labels": p_decoder_labels_batch,
            "qa_decoder_input_ids": qa_ids_batch,
            "qa_decoder_attention_mask": qa_decoder_matrix_attention_mask_batch,  # [B,L,L]
            "qa_decoder_labels": qa_decoder_labels_batch,
        }

        return batch
