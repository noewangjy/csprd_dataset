import os
import re
import spacy
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding
import pandas as pd

tqdm.pandas(desc='Processing text')


class TextPreprocessor():
    def __init__(self, spacy_model):
        self.nlp = spacy.load(spacy_model)

    def preprocess(self, series, lowercase=True, remove_punct=True,
                   remove_num=True, remove_stop=True, lemmatize=True):
        return (series.progress_apply(
            lambda text: self.preprocess_text(text, lowercase, remove_punct, remove_num, remove_stop, lemmatize)))

    def preprocess_text(self, text, lowercase, remove_punct,
                        remove_num, remove_stop, lemmatize):
        if lowercase:
            text = self._lowercase(text)
        doc = self.nlp(text)
        if remove_punct:
            doc = self._remove_punctuation(doc)
        if remove_num:
            doc = self._remove_numbers(doc)
        if remove_stop:
            doc = self._remove_stop_words(doc)
        if lemmatize:
            text = self._lemmatize(doc)
        else:
            text = self._get_text(doc)
        return text

    def _lowercase(self, text):
        return text.lower()

    def _remove_punctuation(self, doc):
        return [t for t in doc if not t.is_punct]

    def _remove_numbers(self, doc):
        return [t for t in doc if not (t.is_digit or t.like_num or re.match('.*\d+', t.text))]

    def _remove_stop_words(self, doc):
        return [t for t in doc if not t.is_stop]

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])

    def _get_text(self, doc):
        return ' '.join([t.text for t in doc])


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
            self.dataset[key]['input_ids'],
            truncation='only_first',
            max_length=self.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return sample






