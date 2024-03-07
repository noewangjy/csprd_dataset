export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data

#bash $DATAPATH/get_data_msmarco.sh

python preprocess.py \
  --tokenizer_name bert-base-uncased \
  --max_seq_length 150 \
  --data_dir $DATAPATH/msmarco_raw.dataset \
  --output_dir $DATAPATH/msmarco-bert-base-uncased.dataset \
  --use_title
