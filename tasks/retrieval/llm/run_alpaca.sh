export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data/cspd_v1.dataset


python run.py \
  --output_dir /data2/retriever/masked_qa-base-base-spm-10epochs \
  --model_name_or_path chavinlo/gpt4-x-alpaca \
  --do_train  \
  --corpus_file $DATAPATH/corpus \
  --train_query_file $DATAPATH/train_query \
  --train_qrels $DATAPATH/train_qrels.txt \
  --neg_file $DATAPATH/train_negs.txt  \
  --query_max_len 1512 \
  --passage_max_len 1512 \
  --fp16  \
  --per_device_train_batch_size 1 \
  --train_group_size 8 \
  --sample_neg_from_topk 200 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --overwrite_output_dir  \
  --dataloader_num_workers 10
