import os
import json
import argparse
from os.path import abspath, join
import pandas as pd
from src.utils.eval import Evaluator
from src.models.embedding_models import Word2VecRetriever, TrainedWord2VecRetriever, T5Retriever, Doc2VecRetriever, BERTRetriever


def main(args):
    print("Loading examples and policy items...")
    df_policy = pd.read_csv(args.corpus_path, sep='\t')
    df_example_test = pd.read_csv(args.dev_path, sep='\t')
    ground_truths = df_example_test['pos_ctxs'].apply(lambda x: list(map(int, x.split(',')))).tolist()

    policy_items = df_policy['content'].tolist()
    policy_ids = df_policy['id'].tolist()
    test_examples = df_example_test['passage'].tolist()

    print("Initializing the {} retriever model...".format(args.retriever))
    if args.retriever == 't5':
        retriever = T5Retriever(
            retrieval_corpus=policy_items,
            corpus_ids=policy_ids,
            model_name=args.model_path,
            device=args.device
        )
    elif args.retriever == 'bert':
        retriever = BERTRetriever(
            retrieval_corpus=policy_items,
            corpus_ids=policy_ids,
            model_name=args.model_path,
            device=args.device
        )
    elif args.retriever == 'trained-word2vec':
        retriever = TrainedWord2VecRetriever(
            retrieval_corpus=policy_items,
            corpus_ids=policy_ids,
            vector_size=1000,
            window=5,
            min_count=1,
            workers=40
        )
    elif args.retriever == 'doc2vec':
        retriever = Doc2VecRetriever(
            retrieval_corpus=policy_items,
            corpus_ids=policy_ids,
            vector_size=1000,
            window=5,
            min_count=1,
            workers=40
        )
    elif args.retriever == 'word2vec':
        retriever = Word2VecRetriever(retrieval_corpus=policy_items, corpus_ids=policy_ids, model_path=args.model_path)
    else:
        raise ValueError("Invalid retriever name.")

    print("Running model on test examples...")
    retrieved_docs = retriever.search_all(test_examples, top_k=1000)

    print("Computing the retrieval scores...")
    evaluator = Evaluator()
    scores = evaluator.compute_all_metrics(retrieved_docs, ground_truths)

    print("Saving the scores to {} ...".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(join(args.output_dir, f'{args.retriever}_test_results.json'), 'w') as f:
        json.dump(scores, f, indent=2)
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        default="/data1/policy_matching/data/embedding/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5",
                        help="Path of the model."
                        )
    parser.add_argument("--corpus_path",
                        type=str,
                        default="/data1/policy_matching/data/cspd_v1/policy_corpus.tsv",
                        help="Path of the data file containing the law articles."
                        )
    parser.add_argument("--dev_path",
                        type=str,
                        default="/data1/policy_matching/data/cspd_v1/dev_set.tsv.tsv",
                        help="Path of the data file containing the test questions."
                        )
    parser.add_argument("--retriever",
                        type=str,
                        choices=["t5", "trained-word2vec", "word2vec", "doc2vec", "bert"],
                        required=True,
                        help="The type of model to use for retrieval"
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default=abspath(join(__file__, "/data2/zeroshot/test-run/")),
                        help="Path of the output directory."
                        )
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="The device to use for inference."
                        )
    args, _ = parser.parse_known_args()
    main(args)
