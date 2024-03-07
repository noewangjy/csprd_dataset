export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data

#bash $DATAPATH/get_data_msmarco.sh

python preprocess.py \
  --tokenizer_name hfl/chinese-macbert-base \
  --max_seq_length 512 \
  --data_dir $DATAPATH/cspd_v1 \
  --output_dir $DATAPATH/csprd_chinese-macbert-base.dataset
