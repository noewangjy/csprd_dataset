export CHECKPOINT_PATH=/data2
export DATA_PATH=data
export DATASET_PATH=$DATA_PATH/csprd_chinese-macbert-base.dataset
export MODEL_NAME_OR_PATH=hfl/chinese-macbert-base
export MODEL_NAME=$CHECKPOINT_PATH/masked_qa-chinese-macbert-base-tiny-spm-wwm
export PRETRAIN_PATH=tasks/pretrain/masked_qa
export RETRIEVAL_PATH=tasks/retrieval/masked_qa
export EPOCHS=50
export OUTPUT_NAME=$MODEL_NAME/retrieval
export RESDIR=$OUTPUT_NAME/results

## Pretrain
python -m torch.distributed.launch --nproc_per_node 8 \
  $PRETRAIN_PATH/run.py \
  --output_dir $MODEL_NAME \
  --data_dir $DATA_PATH/chinese_corpus_tiny_spm.dataset  \
  --do_train True \
  --save_steps 20000 \
  --per_device_train_batch_size 28 \
  --max_seq_length 512 \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --fp16 True \
  --warmup_ratio 0.1 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --overwrite_output_dir True \
  --dataloader_num_workers 12 \
  --weight_decay 0.01 \
  --encoder_mlm_probability 0.3 \
  --decoder_mlm_probability 0.0 \
  --decoder_answer_probability 1.0


## Fine-tuning
python -m torch.distributed.launch --nproc_per_node 8 \
  $RETRIEVAL_PATH/run.py \
  --output_dir $OUTPUT_NAME \
  --model_name_or_path $MODEL_NAME \
  --do_train  \
  --corpus_file $DATASET_PATH/corpus \
  --train_query_file $DATASET_PATH/train_query \
  --train_qrels $DATASET_PATH/train_qrels.txt \
  --neg_file $DATASET_PATH/train_negs.txt  \
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


## Retrieval
python -m torch.distributed.launch --nproc_per_node 8 \
  $RETRIEVAL_PATH/run.py \
--output_dir /data2/retriever/empty \
--model_name_or_path $OUTPUT_NAME  \
--corpus_file $DATASET_PATH/corpus \
--passage_max_len 512 \
--fp16  \
--do_predict \
--prediction_save_path $RESDIR \
--per_device_eval_batch_size 8 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 10


## Prediction
python -m torch.distributed.launch --nproc_per_node 8 \
  $RETRIEVAL_PATH/run.py \
--output_dir $RESDIR/empty \
--model_name_or_path $OUTPUT_NAME \
--test_query_file $DATASET_PATH/dev_query \
--query_max_len 512 \
--fp16  \
--do_predict \
--prediction_save_path $RESDIR/ \
--per_device_eval_batch_size 8 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 10

## Evaluation
python $RETRIEVAL_PATH/test.py \
--query_reps_path $RESDIR/query_reps \
--passage_reps_path $RESDIR/passage_reps \
--qrels_file $DATASET_PATH/dev_qrels.txt \
--ranking_file  $RESDIR/dev_ranking.txt \
--use_gpu

