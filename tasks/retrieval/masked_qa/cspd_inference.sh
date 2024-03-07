export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data/cspd_v1.dataset
export MODEL_PATH=/data2/retriever/masked_qa-mengzi-bert-base-fin-base-spm-wwm
export RESDIR=

python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
--output_dir /data2/retriever/empty \
--model_name_or_path $MODEL_PATH  \
--corpus_file $DATAPATH/corpus \
--passage_max_len 512 \
--fp16  \
--do_predict \
--prediction_save_path $RESDIR/ \
--per_device_eval_batch_size 8 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 10


python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
--output_dir /data2/retriever/empty \
--model_name_or_path $MODEL_PATH \
--test_query_file $DATAPATH/dev_query \
--query_max_len 512 \
--fp16  \
--do_predict \
--prediction_save_path $RESDIR/ \
--per_device_eval_batch_size 8 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 10


python test.py \
--query_reps_path $RESDIR/query_reps \
--passage_reps_path $RESDIR/passage_reps \
--qrels_file $DATAPATH/dev_qrels.txt \
--ranking_file  $RESDIR/dev_ranking.txt \
--use_gpu

