export CHECKPOINT_PATH=/data2
export DATA_PATH=/data1/policy_matching/data
export PYTHONPATH=$PYTHONPATH:../../
export MODEL_NAME=$CHECKPOINT_PATH/masked_qa-base-large-spm-new/retrieval
export INFERENCE_PATH=/data1/policy_matching/tasks/inference/masked_qa
export RESDIR=outputs
export CUDA_VISIBLE_DEVICES=3

python $INFERENCE_PATH/run.py \
--output_dir /data2/retriever/empty \
--model_name_or_path $MODEL_NAME  \
--corpus_file $DATA_PATH/cspd_v1.dataset/corpus \
--dataset_path $DATA_PATH/cspd_v1 \
--inference_file all_stock_passages.json \
--keyword_file all_keywords.json \
--passage_max_len 512 \
--fp16  \
--prediction_save_path $RESDIR \
--per_device_eval_batch_size 10 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 10


