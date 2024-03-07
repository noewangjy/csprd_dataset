export CHECKPOINT_PATH=/data2/
export DATA_PATH=/data1/policy_matching/data
export PYTHONPATH=$PYTHONPATH:../../../


python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
  --output_dir $CHECKPOINT_PATH/test \
  --data_dir $DATA_PATH/chinese_corpus_base_spm.dataset  \
  --do_train True \
  --save_steps 20000 \
  --per_device_train_batch_size 36 \
  --max_seq_length 512 \
  --model_name_or_path bert-base-chinese \
  --fp16 True \
  --warmup_ratio 0.1 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --overwrite_output_dir True \
  --dataloader_num_workers 11 \
  --weight_decay 0.01 \
  --encoder_mlm_probability 0.3 \
  --decoder_mlm_probability 0.5
