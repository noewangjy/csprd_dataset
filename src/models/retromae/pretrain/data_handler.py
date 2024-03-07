import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List
import torch.utils.data.dataset
from datasets import Dataset
from transformers import DataCollatorForWholeWordMask
from src.models.retromae.pretrain.utils import tensorize_batch
from ltp import LTP, StnSplit
from src.utils.special_tokens import *

word_spliter = LTP()
word_spliter.add_words(special_tokens)
sentence_spliter = StnSplit()


class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.dataset = Dataset.load_from_disk(data_dir)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


@dataclass
class RetroMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:
            e_trunc = self.tokenizer.build_inputs_with_special_tokens(e['token_ids'][:tgt_len])
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(tokens), 256)):
                mask_set.append(self._whole_word_mask(tokens))

            text_matrix_attention_mask = []
            for i in range(len(tokens)):
                idx = random.randint(0, min(len(tokens), 256) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_matrix_attention_mask.append(text_decoder_mlm_mask)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor(e_trunc))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch, encoder_mlm_mask_batch)
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return batch
#
#
# def tokenize_for_wwm(sentence):
#     tokenized_sentence = word_spliter.pipeline([sentence], tasks=["cws"], return_dict=False)[0][0]
#     tokens_for_wwm = []
#     for token in tokenized_sentence:
#         if len(token) == 1 or token == BLANK:
#             tokens_for_wwm.append(token)
#         else:
#             tokens_for_wwm.append(token[0])
#             digit_flag = token[0].isdigit()
#             for sub_token in token[1:]:
#                 if sub_token.isdigit() and digit_flag:
#                     tokens_for_wwm[-1] += sub_token
#                     digit_flag = True
#                 else:
#                     tokens_for_wwm.append("##" + sub_token)
#     return tokens_for_wwm
#

@dataclass
class RetroMAECollatorForChinese(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def get_wwm_with_prob(self, tokens: List[str], prob: float):
        self.mlm_probability = prob
        return self._whole_word_mask(tokens)

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for example in examples:

            all_passage_tokens = example["passage_tokens"]
            passage_tokens = all_passage_tokens[:tgt_len]
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

            input_ids_batch.append(torch.tensor(passage_ids))
            attention_mask_batch.append(torch.tensor([1] * len(passage_ids)))
            passage_ids[0] = -100
            passage_ids[-1] = -100
            decoder_labels_batch.append(torch.tensor(passage_ids))

            encoder_mlm_mask_batch.append(torch.tensor(passage_encoder_mlm_mask))
            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(passage_matrix_attention_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch, encoder_mlm_mask_batch)
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return batch
