# The Chinese Stock Policy Retrieval Dataset (CSPRD)

This repo (under construction) contains the codebase and datasets for the paper: **CSPRD: A Financial Policy Retrieval Dataset for Chinese Stock Market. **

❗️**Updated on 2023-09-11**: Our research paper is available on [arXiv](https://arxiv.org/abs/2309.04389).

## Abstract

In recent years, great advances in pre-trained language models (PLMs) have sparked considerable research focus and achieved promising performance on the approach of dense passage retrieval, which aims at retrieving relative passages from massive corpus with given questions. However, most of existing datasets mainly benchmark the models with factoid queries of general commonsense, while specialised fields such as finance and economics remain unexplored due to the deficiency of large-scale and high-quality datasets with expert annotations. 

In this work, we propose a new task, policy retrieval, by introducing the Chinese Stock Policy Retrieval Dataset (CSPRD), which provides 700+ prospectus passages labeled by experienced experts with relevant articles from 10k+ entries in our collected Chinese policy corpus. Experiments on lexical, embedding and fine-tuned bi-encoder models show the effectiveness of our proposed CSPRD yet also suggests ample potential for improvement. Our best performing baseline achieves 56.1% MRR@10, 28.5% NDCG@10, 37.5% Recall@10 and 80.6% Precision@10 on dev set.

> 在近些年，预训练语言模型 (PLMs) 取得了巨大的进展，引发了大量的研究焦点，并在密集段落检索的方法上取得了令人瞩目的性能。此方法旨在根据给定的问题从大型文献中检索相关段落。然而，大多数现有的数据集主要是用一般常识的事实性查询来评估模型的性能，而像金融和经济这样的专业领域由于缺乏大规模、高质量且带有专家注解的数据集而尚未被探索。

> 在这项工作中，我们提出了一个新任务，即政策检索。我们推出了“中国股票政策检索数据集”(CSPRD)，其中提供了700多个由经验丰富的专家标注的招股说明书段落，这些段落与我们收集的中国政策文献中的10k+条目中的相关文章有关。对词汇、嵌入和微调的双编码器模型的实验显示了我们提出的CSPRD的有效性，但也表明还有很大的改进潜力。在开发集上，我们表现最好的基线达到了56.1%的MRR@10，28.5%的NDCG@10，37.5%的Recall@10和80.6%的Precision@10。


## Dataset

- Chinese version released in `csprd_zh` folder
- English version released in `csprd_en` folder


## Citation

If you find this dataset and paper useful or use the codebase in your work, please cite our paper:

```bibtex
@misc{wang2023csprd,
      title={CSPRD: A Financial Policy Retrieval Dataset for Chinese Stock Market}, 
      author={Jinyuan Wang and Hai Zhao and Zhong Wang and Zeyang Zhu and Jinhao Xie and Yong Yu and Yongjian Fei and Yue Huang and Dawei Cheng},
      year={2023},
      eprint={2309.04389},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
