export PYTHONPATH=$PYTHONPATH:../../../
export DATAPATH=/data1/policy_matching/data/beir-bert-base-uncased.dataset
export RAWDATAPATH=/data1/policy_matching/data/msmarco_raw.dataset

python -m torch.distributed.launch --nproc_per_node 8 \
  run.py \
--output_dir /data2/retriever/masked_qa-beir \
--model_name_or_path /data2/masked_qa-base-bert-en \
--do_train  \
--corpus_file ./data/BertTokenizer_data/corpus \
--train_query_file ./data/BertTokenizer_data/train_query \
--train_qrels ./data/BertTokenizer_data/train_qrels.txt \
--neg_file beir_neg.txt \
--query_max_len 32 \
--passage_max_len 144 \
--fp16  \
--per_device_train_batch_size 128 \
--train_group_size 2 \
--sample_neg_from_topk 200 \
--learning_rate 1e-5 \
--num_train_epochs 10 \
--negatives_x_device  \
--dataloader_num_workers 10
