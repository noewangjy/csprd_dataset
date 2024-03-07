export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data/msmarco-bert-base-uncased.dataset
export RAWDATAPATH=/data1/policy_matching/data/msmarco_raw.dataset

python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
  --output_dir /data2/retriever/masked_qa-msmarco-continue-5epochs \
  --model_name_or_path /data2/retriever/masked_qa-msmarco \
  --do_train  \
  --corpus_file $DATAPATH/corpus \
  --train_query_file $DATAPATH/train_query \
  --train_qrels $DATAPATH/train_qrels.txt \
  --neg_file $RAWDATAPATH/train_negs.tsv  \
  --query_max_len 32 \
  --passage_max_len 140 \
  --fp16  \
  --per_device_train_batch_size 20 \
  --train_group_size 16 \
  --sample_neg_from_topk 200 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --negatives_x_device  \
  --overwrite_output_dir  \
  --dataloader_num_workers 10
