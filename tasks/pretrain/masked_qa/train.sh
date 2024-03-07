export PYTHONPATH=$PYTHONPATH:../../../
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node 7 \
  run_english.py \
  --output_dir /data2/masked_qa-base-bert-english-0414 \
  --data_dir /data1/policy_matching/data/english_corpus_bert.dataset \
  --do_train True \
  --save_steps 20000 \
  --per_device_train_batch_size 100 \
  --max_seq_length 140 \
  --model_name_or_path bert-base-uncased \
  --fp16 True \
  --warmup_ratio 0.1 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --overwrite_output_dir True \
  --dataloader_num_workers 10 \
  --weight_decay 0.01 \
  --encoder_mlm_probability 0.3 \
  --decoder_mlm_probability 0.5 \
  --decoder_answer_probability 1.0
