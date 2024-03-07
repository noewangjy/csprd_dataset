import argparse
import random
from pathlib import Path
from copy import deepcopy

import datasets
from datasets import load_dataset, concatenate_datasets
import json
from tqdm import tqdm
from src.utils.special_tokens import *
# from ltp import LTP, StnSplit
import os
import re
import jieba

os.environ["HF_DATASETS_CACHE"] = "/data1/.cache/huggingface/datasets"


def has_chinese(text):
    for ch in text:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    sentences = para.split("\n")
    if isinstance(sentences, list):
        return sentences
    else:
        return [sentences]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-chinese")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--short_seq_prob", type=float, default=0)

    return parser.parse_args()


def create_chinese_corpus(data_path: str):
    for _ in special_tokens:
        jieba.add_word(_)

    def clean_text(text):
        text = text.replace("。\n\n", "。")
        text = text.replace("\n", "。")
        text = text.replace("\r", "。")
        # text = text.replace(" ", "")
        return text

    def tokenize_for_wwm(sentence):
        tokenized_sentence = jieba.lcut(sentence)
        tokens_for_wwm = []
        for token in tokenized_sentence:
            # jieba can not tokenize [BLANK] as a whole word, but as 3 tokens [ BLANK ]
            if len(token) == 1 or token == "BLANK":
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

        for i in range(len(tokens_for_wwm)):
            if tokens_for_wwm[i] == "BLANK":
                tokens_for_wwm[i] = "[BLANK]"
                tokens_for_wwm.pop(i + 1)
                tokens_for_wwm.pop(i - 1)
                break

        return tokens_for_wwm

    def chinese_tokenize_function(examples):
        passage = clean_text(examples['passage'])
        # p_sentences = sentence_splitter.split(passage)
        p_sentences = cut_sent(passage)
        p_sentences_for_wwm = []
        for sentence in p_sentences:
            tokens_for_wwm = tokenize_for_wwm(sentence)
            p_sentences_for_wwm.append(tokens_for_wwm)
        passage_tokens = [token for sentence in p_sentences_for_wwm for token in sentence]

        query = "问题：" + examples['query']
        query_tokens = tokenize_for_wwm(query)

        if examples["choice"]:
            choices = examples['choice'].split(CHOICE)[1:]
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

    dataset_zh = load_dataset("json", data_files=data_path, cache_dir="/data1/.cache/huggingface/datasets")
    dataset_zh = dataset_zh.map(chinese_tokenize_function, num_proc=80, remove_columns=[
        "passage",
        "query",
        "answer",
        "task",
        "choice"
    ])

    return dataset_zh


def create_wiki_zh_data(
        data_path: str,
        max_seq_length: int,
        short_seq_prob: float = 0.0
):
    for _ in special_tokens:
        jieba.add_word(_)

    target_length = max_seq_length - 2

    def tokenize_for_wwm(sentence):
        tokenized_sentence = jieba.lcut(sentence)
        tokens_for_wwm = []
        for token in tokenized_sentence:
            # jieba can not tokenize [BLANK] as a whole word, but as 3 tokens [ BLANK ]
            if len(token) == 1 or token == "BLANK":
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

        for i in range(len(tokens_for_wwm)):
            if tokens_for_wwm[i] == "BLANK":
                tokens_for_wwm[i] = BLANK
                tokens_for_wwm.pop(i + 1)
                tokens_for_wwm.pop(i - 1)
                break

        return tokens_for_wwm

    def clean_text(text):
        text = text.replace("\n\n", "\n")
        text = text.replace("\n", "。")
        text = text.replace("\r", "。")
        text = text.replace(" ", "")
        text = re.sub(r"。+", "。", text)
        return text

    def wiki_tokenize_function(examples):
        input_tokens = []
        for sents in examples['sentences']:
            input_tokens.append([tokenize_for_wwm(sentence) for sentence in sents])
        return {"input_tokens": input_tokens}

    def sentence_wiki(examples):
        passage = clean_text(examples["text"])
        sentences = cut_sent(passage)
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

        for sents in examples['input_tokens']:
            curr_block = []
            curr_sentences = []
            curr_len = 0
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(256, target_length)
            for sent in sents:
                if curr_len >= curr_tgt_len:
                    blank_idx = random.choice(range(len(curr_sentences)))
                    blank_sentence = "".join(deepcopy(curr_sentences[blank_idx])).replace("#", "")
                    num_sentence_pieces = len(re.findall(r"[\u4e00-\u9fa5]+", blank_sentence))
                    retry = 0
                    while num_sentence_pieces < 1:
                        blank_idx = random.choice(range(len(curr_sentences)))
                        blank_sentence = "".join(deepcopy(curr_sentences[blank_idx])).replace("#", "")
                        num_sentence_pieces = len(re.findall(r"[\u4e00-\u9fa5]+", blank_sentence))
                        retry += 1
                        if retry > 100:
                            blank_idx = 0
                            break

                    for idx in range(len(curr_sentences)):
                        if idx == blank_idx:
                            blank_sentence = "".join(deepcopy(curr_sentences[idx])).replace("#", "")
                            sentence_piece = re.findall(r"[\u4e00-\u9fa5]+", blank_sentence)
                            # if len(sentence_piece) == 0:

                            try:
                                blank_piece = random.choice(sentence_piece)
                            except:
                                print(sentence_piece)
                                raise ValueError(blank_sentence + "|" + "|".join(["".join(sent) for sent in curr_sentences]))

                            retry = 0
                            while len(blank_piece) < 4 or len(blank_piece) > 10:
                                blank_piece = random.choice(sentence_piece)
                                if retry > 10:
                                    break
                                retry += 1
                            blank_sentence = blank_sentence.replace(blank_piece, BLANK)
                            curr_block.extend(tokenize_for_wwm(blank_sentence))
                            answer_blocks.append(tokenize_for_wwm("答案：" + blank_piece))
                            query_blocks.append(tokenize_for_wwm("问题：" + random.choice(questions)))
                        else:
                            curr_block.extend(deepcopy(curr_sentences[idx]))
                    passage_blocks.append(curr_block)
                    curr_block = []
                    curr_sentences = []
                    curr_len = 0
                    curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(256, target_length)

                if has_chinese("".join(sent)):
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
            "passage_tokens": passage_blocks,
            "query_tokens": query_blocks,
            "answer_tokens": answer_blocks
        }

    print("Loading wiki_zh dataset...")
    wiki = load_dataset("json", data_files=data_path, cache_dir="/data1/.cache/huggingface/datasets")

    # print(wiki["train"]["text"][:2])
    print("wiki loaded")
    wiki = wiki.map(sentence_wiki, num_proc=80, remove_columns=["title", "text"])

    print("wiki splitted")
    tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=80, batched=True, remove_columns=["sentences"])
    print("wiki tokenized")
    processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=80, batched=True, remove_columns=["input_tokens"], )

    return processed_wiki


if __name__ == '__main__':
    args = get_args()
    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # print("Loading dataset...")
    all_dataset = open(os.path.join(args.output_dir, "raw_dataset.json"), "w")
    data_files = [
        "sohu_news.json",           # large
        "clue.json",                  # tiny
        "cmrc2019.json",              # tiny
        "dureader_robust.json",       # tiny
        "dureader-search.json",     # small
        "dureader-zhidao.json",     # small
        "news2016zh.json",          # base
    ]
    for data_file in data_files:
        print("Processing " + data_file)
        with open(os.path.join(args.data, data_file), "r") as f:
            dataset = json.load(f)
            for data in tqdm(dataset, desc=data_file):
                all_dataset.write(json.dumps(data) + "\n")

    dataset_zh = create_chinese_corpus(os.path.join(args.data, "raw_dataset.json"))
    # dataset_zh.save_to_disk(os.path.join(args.output_dir, "chinese_tiny.dataset"))
    # dataset_zh = datasets.load_from_disk(os.path.join(args.output_dir, "chinese_tiny.dataset"))
    # wiki_zh = create_wiki_zh_data(os.path.join(args.data, "wiki_zh.json"), 512)
    # print("saving data to", os.path.join(args.output_dir, "chinese_wiki-spm.dataset"))
    # wiki_zh.save_to_disk(os.path.join(args.output_dir, "chinese_wiki-spm.dataset"))
    wiki_zh = datasets.load_from_disk(os.path.join(args.output_dir, "chinese_wiki-spm.dataset"))
    # dataset_zh = datasets.load_from_disk(os.path.join(args.output_dir, "chinese_tiny.dataset"))
    # if "train" in dataset_zh:
    #     print("train dataset_zh loaded")
    # if "train" in wiki_zh:
    #     print("train wiki_zh loaded")

    dataset = concatenate_datasets([
        dataset_zh["train"],
        # dataset_zh["validation"],
        # dataset_zh["test"],
        wiki_zh["train"],
        # wiki_zh["validation"],
        # wiki_zh["test"]
    ])
    print("saving data to", os.path.join(args.output_dir, "chinese_corpus_tiny_spm.dataset"))
    dataset.save_to_disk(os.path.join(args.output_dir, "chinese_corpus_tiny_spm.dataset"))


    # from transformers import AutoTokenizer
    #
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    # tokenizer.add_tokens(special_tokens)
    # print(tokenizer.decode(tokenizer.convert_tokens_to_ids(wiki_zh["train"][100]["passage_tokens"])))
    # print(tokenizer.decode(tokenizer.convert_tokens_to_ids(wiki_zh["train"][100]["query_tokens"])))
    # print(tokenizer.decode(tokenizer.convert_tokens_to_ids(wiki_zh["train"][100]["answer_tokens"])))



    # word_spliter = LTP("/data1/policy_matching/src/LTP")
    # word_spliter.add_words(special_tokens)
    # sentence_splitter = StnSplit()
    #
    # dataset = []
    # with open(os.path.join(args.data, "zh_dataset.json"), "r") as f:
    #     for line in f:
    #         dataset.append(json.loads(line))
    #
    # new_dataset = []
    #
    # for examples in tqdm(dataset):
    #     passage = clean_text(examples['passage'])
    #     # p_sentences = sentence_splitter.split(clean_text(passage))
    #     p_sentences = cut_sent(clean_text(passage))
    #     p_sentences_for_wwm = []
    #     for sentence in p_sentences:
    #         tokens_for_wwm = tokenize_for_wwm(sentence)
    #         p_sentences_for_wwm.append(tokens_for_wwm)
    #     passage_tokens = [token for sentence in p_sentences_for_wwm for token in sentence]
    #
    #     query = "问题：" + examples['query']
    #     query_tokens = tokenize_for_wwm(query)
    #
    #     if examples["choice"]:
    #         choices = examples['choice'].split(CHOICE)[1:]
    #         choice_tokens = []
    #         for choice in choices:
    #             choice_tokens.append(CHOICE)
    #             choice_tokens.extend(tokenize_for_wwm(choice))
    #         query_tokens = query_tokens + tokenize_for_wwm("选项：") + choice_tokens
    #
    #     answer = "答案：" + examples['answer']
    #     answer_tokens = tokenize_for_wwm(answer)
    #
    #     new_dataset.append({
    #         "passage_tokens": passage_tokens,
    #         "query_tokens": query_tokens,
    #         "answer_tokens": answer_tokens
    #     })
    #
    # with open(os.path.join(args.output_dir, "chinese_ext.dataset"), "w") as f:
    #     for data in new_dataset:
    #         f.write(json.dumps(data) + "\n")
    #
