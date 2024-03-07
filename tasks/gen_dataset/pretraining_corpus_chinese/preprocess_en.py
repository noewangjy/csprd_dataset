import argparse
import random
import re
from pathlib import Path
import os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
from copy import deepcopy
from src.utils.special_tokens import *

os.environ["HF_DATASETS_CACHE"] = "/data1/.cache/huggingface/datasets"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--short_seq_prob", type=float, default=0)

    return parser.parse_args()


def create_book_data(tokenizer_name: str,
                     max_seq_length: int,
                     short_seq_prob: float = 0.0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def get_sentence_piece(sentence):
        return re.findall(r"[\w+][\w\ ']+\w", sentence)

    def book_tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)

    def book_pad_each_line(examples):
        passage_blocks = []
        query_blocks = []
        answer_blocks = []

        questions = [
            "Please fill in the [BLANK].",
            "What is the [BLANK]?",
            "What is the right sentence in [BLANK]?",
            "What is the correct sentence in [BLANK]?",
            "Please complete the sentence in [BLANK].",
        ]

        curr_block = []
        current_sentences = []
        curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
        for sent in examples['input_ids']:
            if len(curr_block) >= curr_tgt_len:
                blank_idx = random.choice(range(len(current_sentences)))
                blank_sentence = tokenizer.decode(deepcopy(current_sentences[blank_idx]))
                sentence_pieces = get_sentence_piece(blank_sentence)
                if len(sentence_pieces) > 0:
                    blank_sentence_piece = random.choice(sentence_pieces)
                    blank_sentence = blank_sentence.replace(blank_sentence_piece, "[BLANK]")
                else:
                    blank_sentence_piece = blank_sentence
                    blank_sentence = "[BLANK]."

                blank_tokens = tokenizer.encode(blank_sentence, add_special_tokens=False)
                current_sentences[blank_idx] = deepcopy(blank_tokens)
                new_block = []
                for sent_tokens in current_sentences:
                    new_block.extend(sent_tokens)
                passage_blocks.append(tokenizer.convert_ids_to_tokens(new_block))
                query_blocks.append(tokenizer.tokenize("Question: " + random.choice(questions)))
                answer_blocks.append(tokenizer.tokenize("Answer: " + blank_sentence_piece))
                curr_block = []
                curr_tgt_len = target_length if random.random() > short_seq_prob \
                    else random.randint(3, target_length)
            curr_block.extend(sent)
            current_sentences.append(sent)

        if len(current_sentences) > 0:
            blank_idx = random.choice(range(len(current_sentences)))
            blank_sentence = tokenizer.decode(deepcopy(current_sentences[blank_idx]))
            sentence_pieces = get_sentence_piece(blank_sentence)
            if len(sentence_pieces) > 0:
                blank_sentence_piece = random.choice(sentence_pieces)
                blank_sentence = blank_sentence.replace(blank_sentence_piece, "[BLANK]")
            else:
                blank_sentence_piece = blank_sentence
                blank_sentence = "[BLANK]."

            blank_tokens = tokenizer.encode(blank_sentence, add_special_tokens=False)
            current_sentences[blank_idx] = deepcopy(blank_tokens)
            new_block = []
            for sent_tokens in current_sentences:
                new_block.extend(sent_tokens)
            passage_blocks.append(tokenizer.convert_ids_to_tokens(new_block))
            query_blocks.append(tokenizer.tokenize("Question: " + random.choice(questions)))
            answer_blocks.append(tokenizer.tokenize("Answer: " + blank_sentence_piece))
        return {
            "passage_tokens": passage_blocks,
            "query_tokens": query_blocks,
            "answer_tokens": answer_blocks,
        }

    bookcorpus = load_dataset('bookcorpus', split='train', cache_dir="/data1/.cache/huggingface/datasets")
    tokenized_bookcorpus = bookcorpus.map(book_tokenize_function, num_proc=60, remove_columns=["text"], batched=True)
    processed_bookcorpus = tokenized_bookcorpus.map(book_pad_each_line, num_proc=60, batched=True,
                                                    remove_columns=["input_ids"])

    return processed_bookcorpus


def create_wiki_data(tokenizer_name: str,
                     max_seq_length: int,
                     short_seq_prob: float = 0.0):
    import nltk
    # nltk.download('pBLANKt')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def get_sentence_piece(sentence):
        return re.findall(r"[\w+][\w\ ']+\w", sentence)

    def wiki_tokenize_function(examples):
        sentences = []
        for sents in examples['sentences']:
            results = tokenizer(sents, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                return_token_type_ids=False)['input_ids']
            if len(results) > 0:
                assert isinstance(results[0], list), "results[0] is not a list"
                # assert isinstance(results[0][0], int), "results[0][0] is not an int"
                sentences.append(results)
        return {"input_ids": sentences}

    def sentence_wiki(examples):
        sentences = nltk.sent_tokenize(examples["text"])
        # if isinstance(sentences, str):
        #     sentences = [sentences]

        return {"sentences": sentences}

    def wiki_pad_each_line(examples):
        questions = [
            "Please fill in the [BLANK].",
            "What is the [BLANK]?",
            "What is the right sentence in [BLANK]?",
            "What is the correct sentence in [BLANK]?",
            "Please complete the sentence in [BLANK].",
        ]

        passage_blocks = []
        query_blocks = []
        answer_blocks = []

        for sents in examples['input_ids']:
            # continue
            curr_block = []
            current_sentences = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            for sent in sents:
                if len(curr_block) >= curr_tgt_len:
                    blank_idx = random.choice(range(len(current_sentences)))
                    blank_sentence = tokenizer.decode(deepcopy(current_sentences[blank_idx]))
                    sentence_pieces = get_sentence_piece(blank_sentence)
                    if len(sentence_pieces) > 0:
                        blank_sentence_piece = random.choice(sentence_pieces)
                        blank_sentence = blank_sentence.replace(blank_sentence_piece, "[BLANK]")
                    else:
                        blank_sentence_piece = blank_sentence
                        blank_sentence = "[BLANK]."

                    blank_ids = tokenizer.encode(blank_sentence, add_special_tokens=False)
                    current_sentences[blank_idx] = deepcopy(blank_ids)
                    new_block = []
                    for sent_ids in current_sentences:
                        new_block.extend(sent_ids)
                    for xxx in new_block:
                        assert isinstance(xxx, int), xxx
                    passage_blocks.append(tokenizer.convert_ids_to_tokens(new_block))
                    query_blocks.append(tokenizer.tokenize("Question: " + random.choice(questions)))
                    answer_blocks.append(tokenizer.tokenize("Answer: " + blank_sentence_piece))
                    curr_block = []
                    curr_tgt_len = target_length if random.random() > short_seq_prob \
                        else random.randint(3, target_length)
                curr_block.extend(sent)
                current_sentences.append(sent)

            if len(current_sentences) > 0:
                blank_idx = random.choice(range(len(current_sentences)))
                blank_sentence = tokenizer.decode(deepcopy(current_sentences[blank_idx]))
                sentence_pieces = get_sentence_piece(blank_sentence)
                if len(sentence_pieces) > 0:
                    blank_sentence_piece = random.choice(sentence_pieces)
                    blank_sentence = blank_sentence.replace(blank_sentence_piece, "[BLANK]")
                else:
                    blank_sentence_piece = blank_sentence
                    blank_sentence = "[BLANK]."

                blank_tokens = tokenizer.encode(blank_sentence, add_special_tokens=False)
                current_sentences[blank_idx] = deepcopy(blank_tokens)
                new_block = []
                for sent_tokens in current_sentences:
                    new_block.extend(sent_tokens)
                passage_blocks.append(tokenizer.convert_ids_to_tokens(new_block))
                query_blocks.append(tokenizer.tokenize("Question: " + random.choice(questions)))
                answer_blocks.append(tokenizer.tokenize("Answer: " + blank_sentence_piece))

        return {
            'passage_tokens': passage_blocks,
            'query_tokens': query_blocks,
            'answer_tokens': answer_blocks,
        }


    wiki = load_dataset("wikipedia", "20220301.en", split="train", cache_dir="/data1/.cache/huggingface/datasets")
    # wiki = load_dataset("wikipedia", "20200501.en", split="train").filter(lambda e, i: i < 100, with_indices=True)
    wiki = wiki.map(sentence_wiki, num_proc=80, remove_columns=["title", "text", "id", "url"])
    tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=80, batched=True, remove_columns=["sentences"])
    import json
    with open("../../pretrain/masked_qa/wiki_tokenized.json", "w") as f:
        json.dump(tokenized_wiki[:5], f, indent=2)
    # print(tokenized_wiki[:5])

    processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=80, batched=True, remove_columns=["input_ids"])

    return processed_wiki


def create_passage_data(tokenizer_name: str,
                        max_seq_length: int,
                        short_seq_prob: float = 0.0):
    import nltk
    # nltk.download('pBLANKt')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def get_sentence_piece(sentence):
        return re.findall(r"[\w+][\w\ ']+\w", sentence)

    def passage_tokenize_function(examples):
        if len(examples['title']) > 2:
            text = examples['title'] + ' ' + examples['text']
        else:
            text = examples['text']
        sentences = nltk.sent_tokenize(text)
        return tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                         return_token_type_ids=False)

    def passage_pad_each_line(examples):
        passage_blocks = []
        query_blocks = []
        answer_blocks = []

        questions = [
            "Please fill in the [BLANK].",
            "What is the [BLANK]?",
            "What is the right sentence in [BLANK]?",
            "What is the correct sentence in [BLANK]?",
            "Please complete the sentence in [BLANK].",
        ]

        for sents in examples['input_ids']:
            curr_block = []
            current_sentences = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            for sent in sents:
                if len(curr_block) >= curr_tgt_len:
                    blank_idx = random.choice(range(len(current_sentences)))
                    blank_sentence = tokenizer.decode(deepcopy(current_sentences[blank_idx]))
                    sentence_pieces = get_sentence_piece(blank_sentence)
                    if len(sentence_pieces) > 0:
                        blank_sentence_piece = random.choice(sentence_pieces)
                        blank_sentence = blank_sentence.replace(blank_sentence_piece, "[BLANK]")
                    else:
                        blank_sentence_piece = blank_sentence
                        blank_sentence = "[BLANK]."

                    blank_tokens = tokenizer.encode(blank_sentence, add_special_tokens=False)
                    current_sentences[blank_idx] = deepcopy(blank_tokens)
                    new_block = []
                    for sent_tokens in current_sentences:
                        new_block.append(sent_tokens)
                    passage_blocks.append(new_block)
                    query_blocks.append(
                        tokenizer.encode("Question: " + random.choice(questions), add_special_tokens=False))
                    answer_blocks.append(tokenizer.encode("Answer: " + blank_sentence_piece, add_special_tokens=False))

                    curr_block = []
                    curr_tgt_len = target_length if random.random() > short_seq_prob \
                        else random.randint(3, target_length)
                curr_block.extend(sent)
                current_sentences.append(sent)
            if len(current_sentences) > 0:
                blank_idx = random.choice(range(len(current_sentences)))
                blank_sentence = tokenizer.decode(deepcopy(current_sentences[blank_idx]))
                sentence_pieces = get_sentence_piece(blank_sentence)
                if len(sentence_pieces) > 0:
                    blank_sentence_piece = random.choice(sentence_pieces)
                    blank_sentence = blank_sentence.replace(blank_sentence_piece, "[BLANK]")
                else:
                    blank_sentence_piece = blank_sentence
                    blank_sentence = "[BLANK]."

                blank_tokens = tokenizer.encode(blank_sentence, add_special_tokens=False)
                current_sentences[blank_idx] = deepcopy(blank_tokens)
                new_block = []
                for sent_tokens in current_sentences:
                    new_block.append(sent_tokens)
                passage_blocks.append(new_block)
                query_blocks.append(tokenizer.encode("Question: " + random.choice(questions), add_special_tokens=False))
                answer_blocks.append(tokenizer.encode("Answer: " + blank_sentence_piece, add_special_tokens=False))

        return {
            'passage_ids': passage_blocks,
            'query_ids': query_blocks,
            'answer_ids': answer_blocks,
        }

    def passage_convert_ids_to_tokens(examples):
        passage_tokens = []
        query_tokens = []
        answer_tokens = []
        for example in examples:
            passage_tokens.append(tokenizer.convert_ids_to_tokens(example["passage_ids"]))
            query_tokens.append(tokenizer.convert_ids_to_tokens(example["query_ids"]))
            answer_tokens.append(tokenizer.convert_ids_to_tokens(example["answer_ids"]))

        return {
            "passage_tokens": passage_tokens,
            "query_tokens": query_tokens,
            "answer_tokens": answer_tokens,
        }

    msmarco = load_dataset("Tevatron/msmarco-passage-corpus", split="train", cache_dir="/data1/.cache/huggingface/datasets")
    msmarco = msmarco.remove_columns("docid")
    tokenized_msmarco = msmarco.map(passage_tokenize_function, num_proc=80, remove_columns=["text", "title"])
    processed_msmarco = tokenized_msmarco.map(passage_pad_each_line, num_proc=80, batched=True, batch_size=None,
                                              remove_columns=["input_ids"])
    msmarco_dataset = processed_msmarco.map(passage_convert_ids_to_tokens, num_proc=80, batched=True,
                                            remove_columns=["passage_ids", "query_ids", "answer_ids"])

    return msmarco_dataset


if __name__ == '__main__':
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.data == 'bert_data':
        print('download and preprocess wiki and bookcorpus:')
        wiki = create_wiki_data(args.tokenizer_name, args.max_seq_length, args.short_seq_prob)
        book = create_book_data(args.tokenizer_name, args.max_seq_length, args.short_seq_prob)
        dataset = concatenate_datasets([book, wiki])
        dataset.save_to_disk(args.output_dir)
    elif args.data == 'msmarco_passage':
        print('download and preprocess msmarco-passage:')
        dataset = create_passage_data(args.tokenizer_name, args.max_seq_length, args.short_seq_prob)
        dataset.save_to_disk(args.output_dir)
    else:
        raise NotImplementedError
