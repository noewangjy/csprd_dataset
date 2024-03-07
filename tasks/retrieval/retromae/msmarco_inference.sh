export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data/msmarco-bert-base-uncased.dataset
export RAWDATAPATH=/data1/policy_matching/data/msmarco_raw.dataset
export RESDIR=results_v0

python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
--output_dir /data2/retriever/masked_qa_msmarco_passage_fintune_2 \
--tokenizer_name bert-base-uncased \
--model_name_or_path /data2/retriever/masked_qa-msmarco  \
--corpus_file $DATAPATH/corpus \
--passage_max_len 140 \
--fp16  \
--do_predict \
--prediction_save_path $RESDIR/ \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 100


python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
--output_dir /data2/retriever/masked_qa_msmarco_passage_fintune_2 \
--tokenizer_name bert-base-uncased \
--model_name_or_path /data2/retriever/masked_qa-msmarco \
--test_query_file $DATAPATH/dev_query \
--query_max_len 32 \
--fp16  \
--do_predict \
--prediction_save_path $RESDIR/ \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 100


python test.py \
--query_reps_path $RESDIR/query_reps \
--passage_reps_path $RESDIR/passage_reps \
--qrels_file $RAWDATAPATH/qrels.dev.tsv \
--ranking_file  $RESDIR/dev_ranking.txt \
--use_gpu

