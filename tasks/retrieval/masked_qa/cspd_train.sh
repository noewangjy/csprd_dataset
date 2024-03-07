export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data/cspd_v1.dataset


python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
  --output_dir /data2/retriever/masked_qa-base-base-spm-10epochs \
  --model_name_or_path /data2/masked_qa-base-base-spm \
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
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --negatives_x_device  \
  --overwrite_output_dir  \
  --dataloader_num_workers 10
