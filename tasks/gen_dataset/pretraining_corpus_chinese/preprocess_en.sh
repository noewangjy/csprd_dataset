export PYTHONPATH=$PYTHONPATH:../../../
python preprocess_en.py \
  --output_dir /data1/policy_matching/data/english_corpus_bert.dataset \
  --data bert_data \
  --tokenizer_name bert-base-uncased
