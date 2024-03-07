export PYTHONPATH=$PYTHONPATH:../../
export DATAPATH=/data1/policy_matching/data/cspd_v1

python run.py --corpus_path $DATAPATH/policy_corpus.tsv \
  --dev_path $DATAPATH/dev_set.tsv \
  --retriever bert \
  --model_path bert-base-chinese \
  --output_dir /data2/zeroshot/cspd_v1_bert-base-chinese
