import json
import logging
import os
from pathlib import Path
import pandas as pd
import argparse

from src.utils.faiss_retriever import search_by_faiss

import numpy as np
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from src.models.masked_qa.retrieval import BiEncoderModel, BiTrainer
from src.models.masked_qa.retrieval.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from src.models.masked_qa.retrieval.data import InferenceDataset, PredictionDataset, BiCollator, PredictionCollator, CustomInferenceDataset

logger = logging.getLogger(__name__)


def complete_results(
        input_file,
        result_file,
        dataset_path,
        keyword_file
):

    input_data = json.load(open(input_file, "r"))
    keywords = json.load(open(keyword_file, "r"))
    results = json.load(open(result_file, "r"))
    corpus = pd.read_csv(os.path.join(dataset_path, "policy_corpus.tsv"), sep="\t")
    query = pd.concat([
        pd.read_csv(os.path.join(dataset_path, "train_set.tsv"), sep="\t"),
        pd.read_csv(os.path.join(dataset_path, "dev_set.tsv"), sep="\t")
    ])
    output = []

    for sample in results:

        key_tokens = sample["query_id"].split("-")
        new_sample = {
            # "": sample["query_id"],
            # "category": query.loc[query["id"] == int(sample["query_id"])]["category"].values[0],
            # "subcategory": query.loc[query["id"] == int(sample["query_id"])]["subcategory"].values[0],
            "enterprise": key_tokens[0],
            "title": key_tokens[1],
            "passage": input_data[key_tokens[0]][key_tokens[1]],
            "keywords": "|".join(keywords["context"][sample["query_id"]]) if sample["query_id"] in keywords["context"] else "",
            "results": []
        }


        # new_sample = {
        #     "query_id": sample["query_id"],
        #     "category": query.loc[query["id"] == int(sample["query_id"])]["category"].values[0],
        #     "subcategory": query.loc[query["id"] == int(sample["query_id"])]["subcategory"].values[0],
        #     "enterprise": query.loc[query["id"] == int(sample["query_id"])]["name"].values[0],
        #     "title": query.loc[query["id"] == int(sample["query_id"])]["title"].values[0],
        #     "passage": query.loc[query["id"] == int(sample["query_id"])]["passage"].values[0],
        #     "results": []
        # }

        for rank in sample["results"]:
            new_rank = {
                "policy_id": rank["policy_id"],
                "content": corpus.loc[corpus["id"] == int(rank["policy_id"])]["content"].values[0],
                "keywords": "|".join(keywords["policy"][rank["policy_id"]]) if rank["policy_id"] in keywords["policy"] else "",
                "score": rank["score"],
                "category": corpus.loc[corpus["id"] == int(rank["policy_id"])]["category"].values[0],
                "subcategory": corpus.loc[corpus["id"] == int(rank["policy_id"])]["subcategory"].values[0],
                "policy": corpus.loc[corpus["id"] == int(rank["policy_id"])]["name"].values[0],
                "titles": "|".join(corpus.loc[corpus["id"] == int(rank["policy_id"])]["title_list"].values[0].split(",")) if not
                corpus.loc[corpus["id"] == int(rank["policy_id"])]["title_list"].isna().values[0] else "",
                "tag": corpus.loc[corpus["id"] == int(rank["policy_id"])]["No."].values[0] if not
                corpus.loc[corpus["id"] == int(rank["policy_id"])]["No."].isna().values[0] else "",
                "date": corpus.loc[corpus["id"] == int(rank["policy_id"])]["date"].values[0] if not
                corpus.loc[corpus["id"] == int(rank["policy_id"])]["date"].isna().values[0] else "",
            }
            new_sample["results"].append(new_rank)
        output.append(new_sample)

    with open(result_file, "w") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     cache_dir=model_args.cache_dir,
    # )
    # logger.info('Config: %s', config)

    # Get model
    model = BiEncoderModel.load(
        model_args.model_name_or_path,
        normlized=model_args.normlized,
        sentence_pooling_method=model_args.sentence_pooling_method
    )

    # Get datasets
    train_dataset = None
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=BiCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.info("*** Prediction ***")
    # if data_args.corpus_file is not None:
    logging.info("*** Corpus Prediction ***")
    passage_path = os.path.join(data_args.prediction_save_path, 'passage_reps')
    Path(passage_path).mkdir(parents=True, exist_ok=True)

    trainer.data_collator = PredictionCollator(tokenizer=tokenizer, is_query=False)
    corpus_dataset = PredictionDataset(
        data_path=data_args.corpus_file, tokenizer=tokenizer,
        max_len=data_args.passage_max_len,
    )
    pred_scores = trainer.predict(test_dataset=corpus_dataset).predictions

    if trainer.is_world_process_zero():
        assert len(corpus_dataset) == len(pred_scores)
        np.save(os.path.join(passage_path, 'passage.npy'), pred_scores)
        with open(os.path.join(passage_path, 'offset2passageid.txt'), "w") as writer:
            for line in open(data_args.corpus_id_file):
                cid, offset = line.strip().split('\t')
                writer.write(f'{offset}\t{cid}\t\n')

    # if data_args.inference_file is not None:
    logging.info("*** Inference Embedding ***")
    query_path = os.path.join(data_args.prediction_save_path, 'query_reps')
    Path(query_path).mkdir(parents=True, exist_ok=True)

    trainer.data_collator = PredictionCollator(tokenizer=tokenizer, is_query=True)
    # test_dataset = InferenceDataset(
    #     input_path=data_args.inference_file,
    #     dataset_path=data_args.dataset_path,
    #     tokenizer=tokenizer,
    #     max_len=data_args.query_max_len,
    # )

    test_dataset = CustomInferenceDataset(
        input_path=data_args.inference_file,
        tokenizer=tokenizer,
        max_len=data_args.query_max_len,
    )
    pred_scores = trainer.predict(test_dataset=test_dataset).predictions
    print(len(pred_scores))

    if trainer.is_world_process_zero():
        assert len(test_dataset) == len(pred_scores)
        np.save(os.path.join(query_path, 'query.npy'), pred_scores)
        with open(os.path.join(query_path, 'offset2queryid.txt'), "w") as writer:
            offset = 0
            # for line in open(data_args.inference_file, "r"):
            #     cid = line.strip()
            #     writer.write(f'{offset}\t{cid}\t\n')
            #     offset += 1
            with open(data_args.inference_file, "r") as f:
                dataset = json.load(f)
                for filename in dataset:
                    for title in dataset[filename]:
                        writer.write(f'{offset}\t{filename}-{title}\t\n')
                        offset += 1

    print("Start search by faiss")
    search_by_faiss(
        query_reps_path=os.path.join(data_args.prediction_save_path, 'query_reps'),
        passage_reps_path=os.path.join(data_args.prediction_save_path, 'passage_reps'),
        save_file=os.path.join(data_args.prediction_save_path, 'ranking.json'),
        batch_size=512,
        depth=1000,
        use_gpu=training_args.device.type == 'cuda'
    )

    print("Start completion from dataset")
    complete_results(
        result_file=os.path.join(data_args.prediction_save_path, 'ranking.json'),
        dataset_path=data_args.dataset_path,
        input_file=data_args.inference_file,
        keyword_file=data_args.keyword_file
    )
    print("Done!")


if __name__ == "__main__":
    main()
