#export CHECKPOINT_PATH=/data2
export CHECKPOINT_PATH=/data2/policy_matching_results
export DATA_PATH=/data1/policy_matching/data/csprd_chinese-bert-wwm-ext.dataset
export PYTHONPATH=$PYTHONPATH:../../
#export MODEL_NAME=$CHECKPOINT_PATH/masked_qa-wwm-ext-small-spm-new
export MODEL_NAME=hfl/chinese-bert-wwm-ext
export RETRIEVAL_PATH=/data1/policy_matching/tasks/retrieval/masked_qa
export EPOCHS=50
#export OUTPUT_NAME=$MODEL_NAME/csprd_retrieval
#export OUTPUT_NAME=$CHECKPOINT_PATH/$MODEL_NAME-csprd_retrieval
export OUTPUT_NAME=$CHECKPOINT_PATH/chinese-bert-wwm-ext-csprd_retrieval
export RESDIR=$OUTPUT_NAME/results
#export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export NUM_GPUS=8

python -m torch.distributed.launch --nproc_per_node $NUM_GPUS \
  $RETRIEVAL_PATH/run.py \
  --output_dir $OUTPUT_NAME \
  --model_name_or_path $MODEL_NAME \
  --do_train  \
  --corpus_file $DATA_PATH/corpus \
  --train_query_file $DATA_PATH/train_query \
  --train_qrels $DATA_PATH/train_qrels.txt \
  --neg_file $DATA_PATH/train_negs.txt  \
  --query_max_len 512 \
  --passage_max_len 512 \
  --fp16  \
  --per_device_train_batch_size 6 \
  --train_group_size 8 \
  --sample_neg_from_topk 200 \
  --learning_rate 1e-5 \
  --num_train_epochs $EPOCHS \
  --negatives_x_device  \
  --overwrite_output_dir  \
  --dataloader_num_workers 10

python -m torch.distributed.launch --nproc_per_node $NUM_GPUS \
  $RETRIEVAL_PATH/run.py \
--output_dir /data2/retriever/empty \
--model_name_or_path $OUTPUT_NAME  \
--corpus_file $DATA_PATH/corpus \
--passage_max_len 512 \
--fp16  \
--do_predict \
--prediction_save_path $RESDIR \
--per_device_eval_batch_size 8 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 10

python -m torch.distributed.launch --nproc_per_node $NUM_GPUS \
  $RETRIEVAL_PATH/run.py \
--output_dir $RESDIR/empty \
--model_name_or_path $OUTPUT_NAME \
--test_query_file $DATA_PATH/dev_query \
--query_max_len 512 \
--fp16  \
--do_predict \
--prediction_save_path $RESDIR/ \
--per_device_eval_batch_size 8 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 10

python $RETRIEVAL_PATH/test.py \
--query_reps_path $RESDIR/query_reps \
--passage_reps_path $RESDIR/passage_reps \
--qrels_file $DATA_PATH/dev_qrels.txt \
--ranking_file  $RESDIR/dev_ranking.txt \
--use_gpu

