export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data/cspd_v1.dataset
export EPOCHS=10
#export MODEL_NAME=/data2/retromae-wwm-ext-base-spm
export MODEL_NAME=shibing624/text2vec-base-chinese-paraphrase
export OUTPUT_NAME=/data2/retriever/$MODEL_NAME-10epochs
export RESDIR=$MODEL_NAME-10epochs

python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
  --output_dir $OUTPUT_NAME \
  --model_name_or_path $MODEL_NAME \
  --do_train  \
  --corpus_file $DATAPATH/corpus \
  --train_query_file $DATAPATH/train_query \
  --train_qrels $DATAPATH/train_qrels.txt \
  --neg_file $DATAPATH/train_negs.txt  \
  --query_max_len 512 \
  --passage_max_len 512 \
  --fp16  \
  --per_device_train_batch_size 6 \
  --train_group_size 8 \
  --sample_neg_from_topk 200 \
  --learning_rate 2e-5 \
  --num_train_epochs $EPOCHS \
  --negatives_x_device  \
  --overwrite_output_dir  \
  --dataloader_num_workers 10



python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
--output_dir /data2/retriever/empty \
--model_name_or_path $OUTPUT_NAME  \
--corpus_file $DATAPATH/corpus \
--passage_max_len 512 \
--fp16  \
--do_predict \
--prediction_save_path results/$RESDIR/ \
--per_device_eval_batch_size 8 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 10


python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
--output_dir /data2/retriever/empty \
--model_name_or_path $OUTPUT_NAME \
--test_query_file $DATAPATH/dev_query \
--query_max_len 512 \
--fp16  \
--do_predict \
--prediction_save_path results/$RESDIR/ \
--per_device_eval_batch_size 8 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 10


python test.py \
--query_reps_path results/$RESDIR/query_reps \
--passage_reps_path results/$RESDIR/passage_reps \
--qrels_file $DATAPATH/dev_qrels.txt \
--ranking_file  results/$RESDIR/dev_ranking.txt \
--use_gpu

