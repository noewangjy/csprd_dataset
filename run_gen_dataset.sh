export DATAPATH=csprd_dataset
export TASK_PATH=tasks/gen_dataset/cspd

python $TASK_PATH/preprocess.py \
  --tokenizer_name hfl/chinese-macbert-base \
  --max_seq_length 512 \
  --data_dir $DATAPATH/csprd_zh \
  --output_dir data/csprd_chinese-macbert-base.dataset
