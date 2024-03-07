import argparse
import json
import os
from pathlib import Path
import pandas as pd
import re
import jieba


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_paddle", action='store_true', default=False)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--wwm", action='store_true', default=False)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    return parser.parse_args()


def dataset_save_to_json(input_file, output_file, id_file):
    with open(output_file, 'w', encoding='utf-8') as f, open(id_file, 'w', encoding='utf-8') as fid:
        cnt = -1
        for line in open(input_file, encoding='utf-8'):
            if cnt < 0:
                cnt += 1
                continue
            line = line.strip('\n').split('\t')
            data = {"id": line[0], 'text': line[5]}
            f.write(json.dumps(data) + '\n')
            fid.write(line[0] + '\t' + str(cnt) + '\n')
            cnt += 1


def corpus_save_to_json(input_file, output_file, id_file):
    with open(output_file, 'w', encoding='utf-8') as f, open(id_file, 'w', encoding='utf-8') as fid:
        cnt = -1
        for line in open(input_file, encoding='utf-8'):
            if cnt < 0:
                cnt += 1
                continue
            line = line.strip('\n').split('\t')
            data = {"id": line[0], 'text': line[7]}
            f.write(json.dumps(data) + '\n')
            fid.write(line[0] + '\t' + str(cnt) + '\n')
            cnt += 1


def generate_qrels(dataset, output_pos_file, output_neg_file):
    with open(output_pos_file, 'w', encoding='utf-8') as pos_f, open(output_neg_file, 'w', encoding='utf-8') as neg_f:
        title_flag = True
        for line in open(dataset, encoding='utf-8'):
            if title_flag:
                title_flag = False
                continue
            line = line.strip().split('\t')
            pos_ctxs = line[6].split(",")
            for pos_ctx in pos_ctxs:
                pos_f.write(line[0] + '\t' + pos_ctx + '\n')

            neg_f.write(line[0] + '\t' + line[7] + '\n')


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


def tokenize_function_wwm(examples):
    tokenized_texts = {
        "input_ids": [],
    }
    for text in examples["text"]:
        sentences = cut_sent(text)
        text_tokens = []
        for sentence in sentences:
            tokenized_sentence = jieba.lcut(sentence)
            tokens_for_wwm = []
            for token in tokenized_sentence:
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
            text_tokens.extend(tokens_for_wwm)
        tokenized_texts["input_ids"].append(tokenizer.convert_tokens_to_ids(text_tokens))

    return tokenized_texts


def tokenize_function(examples):
    return tokenizer(examples["text"], add_special_tokens=False, truncation=True, max_length=max_length,
                     return_attention_mask=False,
                     return_token_type_ids=False)


args = get_args()
if args.use_paddle:

    from paddlenlp.transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
else:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

max_length = args.max_seq_length
from datasets import load_dataset


if __name__ == '__main__':
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'corpus')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'train_query')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'dev_query')).mkdir(parents=True, exist_ok=True)

    generate_qrels(
        dataset=os.path.join(args.data_dir, 'train_set.tsv'),
        output_pos_file=os.path.join(args.output_dir, 'train_qrels.txt'),
        output_neg_file=os.path.join(args.output_dir, 'train_negs.txt')
    )

    generate_qrels(
        dataset=os.path.join(args.data_dir, 'dev_set.tsv'),
        output_pos_file=os.path.join(args.output_dir, 'dev_qrels.txt'),
        output_neg_file=os.path.join(args.output_dir, 'dev_negs.txt')
    )

    corpus_save_to_json(
        os.path.join(args.data_dir, 'policy_corpus.tsv'),
        os.path.join(args.data_dir, 'policy_corpus.json'),
        os.path.join(args.output_dir, 'corpus/mapping_id.txt')
    )

    corpus = load_dataset('json', data_files=os.path.join(args.data_dir, 'policy_corpus.json'), split='train')
    if args.wwm:
        corpus = corpus.map(tokenize_function_wwm, num_proc=8, remove_columns=["text"], batched=True)
    else:
        corpus = corpus.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    corpus.save_to_disk(os.path.join(args.output_dir, 'corpus'))
    print('corpus dataset:', corpus)

    dataset_save_to_json(
        os.path.join(args.data_dir, 'train_set.tsv'),
        os.path.join(args.data_dir, 'train_set.json'),
        os.path.join(args.output_dir, 'train_query/mapping_id.txt')
    )
    train_query = load_dataset('json', data_files=os.path.join(args.data_dir, 'train_set.json'), split='train')
    if args.wwm:
        train_query = train_query.map(tokenize_function_wwm, num_proc=8, remove_columns=["text"], batched=True)
    else:
        train_query = train_query.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)

    train_query.save_to_disk(os.path.join(args.output_dir, 'train_query'))
    print('train query dataset:', corpus)

    dataset_save_to_json(
        os.path.join(args.data_dir, 'dev_set.tsv'),
        os.path.join(args.data_dir, 'dev_set.json'),
        os.path.join(args.output_dir, 'dev_query/mapping_id.txt')
    )
    dev_query = load_dataset('json', data_files=os.path.join(args.data_dir, 'dev_set.json'), split='train')
    if args.wwm:
        dev_query = dev_query.map(tokenize_function_wwm, num_proc=8, remove_columns=["text"], batched=True)
    else:
        dev_query = dev_query.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    dev_query.save_to_disk(os.path.join(args.output_dir, 'dev_query'))
    print('dev query dataset:', corpus)
