export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data/msmarco-bert-base-uncased.dataset
export RAWDATAPATH=/data1/policy_matching/data/msmarco_raw.dataset
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export MODEL_NAME=bert-base-uncased
export OUTPUT_NAME=/data2/retriever/masked_qa-msmarco-bert-base-10epochs-0414
export RESDIR=masked_qa-msmarco-bert-base-10epochs

#python -m torch.distributed.launch --nproc_per_node 7 \
#  run.py \
#  --output_dir $OUTPUT_NAME \
#  --model_name_or_path $MODEL_NAME \
#  --do_train  \
#  --corpus_file $DATAPATH/corpus \
#  --train_query_file $DATAPATH/train_query \
#  --train_qrels $DATAPATH/train_qrels.txt \
#  --neg_file $RAWDATAPATH/train_negs.tsv  \
#  --query_max_len 32 \
#  --passage_max_len 140 \
#  --fp16  \
#  --per_device_train_batch_size 20 \
#  --train_group_size 16 \
#  --sample_neg_from_topk 200 \
#  --learning_rate 1e-5 \
#  --num_train_epochs 10 \
#  --negatives_x_device  \
#  --overwrite_output_dir  \
#  --dataloader_num_workers 10

#python -m torch.distributed.launch --nproc_per_node 7 \
#  run.py \
#--output_dir /data2/retriever/empty \
#--model_name_or_path $OUTPUT_NAME  \
#--corpus_file $DATAPATH/corpus \
#--passage_max_len 140 \
#--fp16  \
#--do_predict \
#--prediction_save_path results/$RESDIR/ \
#--per_device_eval_batch_size 2048 \
#--dataloader_num_workers 6 \
#--eval_accumulation_steps 10
#
#python -m torch.distributed.launch --nproc_per_node 7 \
#  run.py \
#  --output_dir /data2/retriever/empty \
#  --model_name_or_path $OUTPUT_NAME \
#  --test_query_file $DATAPATH/dev_query \
#  --query_max_len 140 \
#  --fp16  \
#  --do_predict \
#  --prediction_save_path results/$RESDIR/ \
#  --per_device_eval_batch_size 2048 \
#  --dataloader_num_workers 6 \
#  --eval_accumulation_steps 10


python test.py \
  --query_reps_path results/$RESDIR/query_reps \
  --passage_reps_path results/$RESDIR/passage_reps \
  --qrels_file $RAWDATAPATH/qrels.dev.tsv \
  --ranking_file  results/$RESDIR/dev_ranking.txt \
  --use_gpu
