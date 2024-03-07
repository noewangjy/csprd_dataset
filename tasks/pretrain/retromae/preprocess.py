import argparse
import random
from pathlib import Path
from copy import deepcopy
from datasets import load_dataset, concatenate_datasets
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm
from src.utils.special_tokens import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-chinese")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--short_seq_prob", type=float, default=0)

    return parser.parse_args()


def create_book_data(tokenizer_name: str,
                     max_seq_length: int,
                     short_seq_prob: float = 0.0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def book_tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)

    def book_pad_each_line(examples):
        blocks = []
        curr_block = []

        curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
        for sent in examples['input_ids']:
            if len(curr_block) >= curr_tgt_len:
                blocks.append(curr_block)
                curr_block = []
                curr_tgt_len = target_length if random.random() > short_seq_prob \
                    else random.randint(3, target_length)
            curr_block.extend(sent)
        if len(curr_block) > 0:
            blocks.append(curr_block)
        return {'token_ids': blocks}

    bookcorpus = load_dataset('bookcorpus', split='train')
    tokenized_bookcorpus = bookcorpus.map(book_tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    processed_bookcorpus = tokenized_bookcorpus.map(book_pad_each_line, num_proc=8, batched=True,
                                                    remove_columns=["input_ids"])
    return processed_bookcorpus


def create_wiki_data(tokenizer_name: str,
                     max_seq_length: int,
                     short_seq_prob: float = 0.0):
    import nltk
    # nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def wiki_tokenize_function(examples):
        sentences = []
        for sents in examples['sentences']:
            sentences.append(
                tokenizer(sents, add_special_tokens=False, truncation=False, return_attention_mask=False,
                          return_token_type_ids=False)['input_ids'])
        return {"input_ids": sentences}

    def sentence_wiki(examples):
        sentences = nltk.sent_tokenize(examples["text"])
        return {"sentences": sentences}

    def wiki_pad_each_line(examples):
        blocks = []
        for sents in tqdm(examples['input_ids']):
            curr_block = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            for sent in sents:
                if len(curr_block) >= curr_tgt_len:
                    blocks.append(curr_block)
                    curr_block = []
                    curr_tgt_len = target_length if random.random() > short_seq_prob \
                        else random.randint(3, target_length)
                curr_block.extend(sent)
            if len(curr_block) > 0:
                blocks.append(curr_block)
        return {'token_ids': blocks}

    wiki = load_dataset("wikipedia", language="zh", date="20221201", split="train", beam_runner="DirectRunner")
    print("wiki", len(wiki))
    # wiki = load_dataset("wikipedia", "20200501.en", split="train").filter(lambda e, i: i < 100, with_indices=True)
    wiki = wiki.map(sentence_wiki, num_proc=8, remove_columns=["title", "text"])
    print("wiki", len(wiki))
    tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=8, batched=True, remove_columns=["sentences"])
    processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=8, batched=True, remove_columns=["input_ids"], )

    return processed_wiki


def create_passage_data(tokenizer_name: str,
                        max_seq_length: int,
                        short_seq_prob: float = 0.0):
    import nltk
    # nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def passage_tokenize_function(examples):
        if len(examples['title']) > 2:
            text = examples['title'] + ' ' + examples['text']
        else:
            text = examples['text']
        sentences = nltk.sent_tokenize(text)
        return tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                         return_token_type_ids=False)

    def passage_pad_each_line(examples):
        blocks = []
        for sents in examples['input_ids']:
            curr_block = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            for sent in sents:
                if len(curr_block) >= curr_tgt_len:
                    blocks.append(curr_block)
                    curr_block = []
                    curr_tgt_len = target_length if random.random() > short_seq_prob \
                        else random.randint(3, target_length)
                curr_block.extend(sent)
            if len(curr_block) > 0:
                blocks.append(curr_block)
        return {'token_ids': blocks}

    msmarco = load_dataset("Tevatron/msmarco-passage-corpus", split="train")
    msmarco = msmarco.remove_columns("docid")
    tokenized_msmarco = msmarco.map(passage_tokenize_function, num_proc=8, remove_columns=["text", "title"])
    processed_msmarco = tokenized_msmarco.map(passage_pad_each_line, num_proc=8, batched=True, batch_size=None,
                                              remove_columns=["input_ids"])

    return processed_msmarco


def create_chinese_corpus(data_path: str):
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer.add_special_tokens({"blank_token": BLANK})
    # tokenizer.add_special_tokens({"choice_token": CHOICE})
    # tokenizer.add_tokens(special_tokens)

    from ltp import LTP, StnSplit
    word_spliter = LTP()
    word_spliter.add_words(special_tokens)
    sentence_splitter = StnSplit()

    def tokenize_for_wwm(sentence):
        tokenized_sentence = word_spliter.pipeline([sentence], tasks=["cws"], return_dict=False)[0][0]
        tokens_for_wwm = []
        for token in tokenized_sentence:
            if len(token) == 1 or token == BLANK:
                tokens_for_wwm.append(token)
            else:
                tokens_for_wwm.append(token[0])
                digit_flag = token[0].isdigit()
                for sub_token in token[1:]:
                    if sub_token.isdigit() and digit_flag:
                        tokens_for_wwm[-1] += sub_token
                        digit_flag = True
                    else:
                        tokens_for_wwm.append("##" + sub_token)
        return tokens_for_wwm

    def chinese_tokenize_function(examples):
        passage = examples['passage']
        p_sentences = sentence_splitter.split(passage)
        p_sentences_for_wwm = []
        for sentence in p_sentences:
            tokens_for_wwm = tokenize_for_wwm(sentence)
            p_sentences_for_wwm.append(tokens_for_wwm)
        passage_tokens = [token for sentence in p_sentences_for_wwm for token in sentence]

        query = "问题：" + examples['query']
        query_tokens = tokenize_for_wwm(query)

        if examples["choice"]:
            choices = examples['choices'].split(CHOICE)[1:]
            choice_tokens = []
            for choice in choices:
                choice_tokens.append(CHOICE)
                choice_tokens.extend(tokenize_for_wwm(choice))
            query_tokens = query_tokens + tokenize_for_wwm("选项：") + choice_tokens

        answer = "答案：" + examples['answer']
        answer_tokens = tokenize_for_wwm(answer)

        return {
            "passage_tokens": passage_tokens,
            "query_tokens": query_tokens,
            "answer_tokens": answer_tokens
        }

    dataset_zh = load_dataset("json", data_files=data_path)
    dataset_zh = dataset_zh.map(chinese_tokenize_function, num_proc=8, remove_columns=[
        "passage",
        "query",
        "answer",
        "task",
        "choice"
    ])

    return dataset_zh


def create_wiki_zh_data(
        max_seq_length: int,
        short_seq_prob: float = 0.0
):
    from ltp import LTP, StnSplit
    word_spliter = LTP()
    word_spliter.add_words(special_tokens)
    sentence_splitter = StnSplit()

    target_length = max_seq_length - 2

    def tokenize_for_wwm(sentence):
        tokenized_sentence = word_spliter.pipeline([sentence], tasks=["cws"], return_dict=False)[0][0]
        tokens_for_wwm = []
        for token in tokenized_sentence:
            if len(token) == 1 or token == BLANK:
                tokens_for_wwm.append(token)
            else:
                tokens_for_wwm.append(token[0])
                digit_flag = token[0].isdigit()
                for sub_token in token[1:]:
                    if sub_token.isdigit() and digit_flag:
                        tokens_for_wwm[-1] += sub_token
                        digit_flag = True
                    else:
                        tokens_for_wwm.append("##" + sub_token)
        return tokens_for_wwm

    def clean_text(text):
        text = text.replace("。\n\n", "。")
        text = text.replace("\n", "。")
        text = text.replace("\r", "。")
        text = text.replace(" ", "")
        return text

    def wiki_tokenize_function(examples):
        input_tokens = []
        for sents in examples['sentences']:
            input_tokens.append([tokenize_for_wwm(sentence) for sentence in sents])
        return {"input_tokens": input_tokens}

    def sentence_wiki(examples):
        sentences = sentence_splitter.split(clean_text(examples["text"]))
        return {"sentences": sentences}

    def wiki_pad_each_line(examples):
        passage_blocks = []
        query_blocks = []
        answer_blocks = []
        questions = [
            "[BLANK]处应该填什么句子？",
            "请在[BLANK]处填入一个最恰当的句子。",
            "结合上下文，补全[BLANK]处的句子？"
        ]

        for sents in tqdm(examples['input_tokens']):
            curr_block = []
            curr_sentences = []
            curr_len = 0
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(256, target_length)
            for sent in sents:
                if curr_len >= curr_tgt_len:
                    blank_idx = random.choice(range(len(curr_sentences)))
                    for idx in range(len(curr_sentences)):
                        if idx == blank_idx:
                            curr_block.append(BLANK)
                            answer_blocks.append(tokenize_for_wwm("答案：") + deepcopy(curr_sentences[idx]))
                            query_blocks.append(tokenize_for_wwm("问题：" + random.choice(questions)))
                        else:
                            curr_block.extend(deepcopy(curr_sentences[idx]))
                    passage_blocks.append(curr_block)
                    curr_block = []
                    curr_sentences = []
                    curr_len = 0
                    curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(256, target_length)
                curr_sentences.append(sent)
                curr_len += len(sent)
            if len(curr_block) > 0 and len(curr_sentences) > 1:
                blank_idx = random.choice(range(len(curr_sentences)))
                for idx in range(len(curr_sentences)):
                    if idx == blank_idx:
                        curr_block.append(BLANK)
                        answer_blocks.append(tokenize_for_wwm("答案：") + deepcopy(curr_sentences[idx]))
                        query_blocks.append(tokenize_for_wwm("问题：" + random.choice(questions)))
                    else:
                        curr_block.extend(deepcopy(curr_sentences[idx]))
                passage_blocks.append(curr_block)
        return {
            "passage": passage_blocks,
            "query": query_blocks,
            "answer": answer_blocks
        }

    wiki = load_dataset("wikipedia", language="zh", date="20221201", split="train", beam_runner="DirectRunner")
    wiki = wiki.map(sentence_wiki, num_proc=8, remove_columns=["title", "text"])
    tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=8, batched=True, remove_columns=["sentences"])
    processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=8, batched=True, remove_columns=["input_tokens"], )

    return processed_wiki


if __name__ == '__main__':
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # dataset_zh = create_chinese_corpus(args.data_path)
    # dataset_zh.save_to_disk(args.output_dir)
    wiki_zh = create_wiki_zh_data(args.max_seq_length)
    wiki_zh.save_to_disk(args.output_dir)
