export PYTHONPATH=$PYTHONPATH:../../
export DATAPATH=/data1/policy_matching/data/cspd_v1
export MODEL=sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5


python run.py --corpus_path $DATAPATH/policy_corpus.tsv \
  --dev_path $DATAPATH/dev_set.tsv \
  --retriever word2vec \
  --model_path /data1/policy_matching/data/embedding/$MODEL \
  --output_dir /data2/zeroshot/word2vec_cspd_v1_$MODEL
