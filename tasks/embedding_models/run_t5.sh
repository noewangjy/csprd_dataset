
export PYTHONPATH=$PYTHONPATH:../../
export DATAPATH=/data1/policy_matching/data/cspd_v1

python run.py --corpus_path $DATAPATH/policy_corpus.tsv \
  --dev_path $DATAPATH/dev_set.tsv \
  --retriever t5 \
  --model_path shibing624/prompt-t5-base-chinese \
  --output_dir /data2/zeroshot/t5_cspd_v1_prompt-t5-base-chinese
