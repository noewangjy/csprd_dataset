import logging

import torch
from torch import nn
from transformers import BertForMaskedLM, AutoModelForMaskedLM, PreTrainedTokenizer
from transformers.modeling_outputs import MaskedLMOutput
from src.models.masked_qa.pretrain.arguments import ModelArguments
from src.models.masked_qa.pretrain.decoder import BertLayerForDecoder

logger = logging.getLogger(__name__)


class MaskedQAForPretraining(nn.Module):
    def __init__(
            self,
            bert: BertForMaskedLM,
            tokenizer: PreTrainedTokenizer,
            model_args: ModelArguments,
    ):
        super(MaskedQAForPretraining, self).__init__()
        self.encoder = bert
        self.encoder.resize_token_embeddings(len(tokenizer))

        self.decoder_embeddings = self.encoder.bert.embeddings
        self.mlm_decoder = BertLayerForDecoder(bert.config)
        self.mlm_decoder.apply(self.encoder._init_weights)
        self.qa_decoder = BertLayerForDecoder(bert.config)
        self.qa_decoder.apply(self.encoder._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.model_args = model_args

    def forward(
            self,
            p_encoder_input_ids,
            p_encoder_attention_mask,
            p_encoder_labels,
            p_decoder_input_ids,
            p_decoder_attention_mask,
            p_decoder_labels,
            qa_decoder_input_ids,
            qa_decoder_attention_mask,
            qa_decoder_labels,
    ):
        # return (torch.sum(self.encoder.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]), )
        lm_out: MaskedLMOutput = self.encoder(
            p_encoder_input_ids,
            p_encoder_attention_mask,
            labels=p_encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = lm_out.hidden_states[-1][:, :1]  # B 1 D

        mlm_decoder_embedding_output = self.decoder_embeddings(input_ids=p_decoder_input_ids)
        mlm_hiddens = torch.cat([cls_hiddens, mlm_decoder_embedding_output[:, 1:]], dim=1)

        mlm_decoder_position_ids = self.encoder.bert.embeddings.position_ids[:, :p_decoder_input_ids.size(1)]
        mlm_decoder_position_embeddings = self.encoder.bert.embeddings.position_embeddings(mlm_decoder_position_ids)  # B L D
        mlm_query = mlm_decoder_position_embeddings + cls_hiddens  # B L D

        mlm_matrix_attention_mask = self.encoder.get_extended_attention_mask(
            p_decoder_attention_mask,
            p_decoder_attention_mask.shape,
            p_decoder_attention_mask.device
        )

        mlm_decoder_output = self.mlm_decoder(query=mlm_query,
                                              key=mlm_hiddens,
                                              value=mlm_hiddens,
                                              attention_mask=mlm_matrix_attention_mask)[0]
        mlm_pred_scores, mlm_loss = self.compute_loss(mlm_decoder_output, p_decoder_labels)

        qa_decoder_embedding_output = self.decoder_embeddings(input_ids=qa_decoder_input_ids)
        qa_hiddens = torch.cat([cls_hiddens, qa_decoder_embedding_output[:, 1:]], dim=1)

        qa_decoder_position_ids = self.encoder.bert.embeddings.position_ids[:, :qa_decoder_input_ids.size(1)]
        qa_decoder_position_embeddings = self.encoder.bert.embeddings.position_embeddings(qa_decoder_position_ids)  # B L D
        qa_query = qa_decoder_position_embeddings + cls_hiddens  # B L D

        qa_matrix_attention_mask = self.encoder.get_extended_attention_mask(
            qa_decoder_attention_mask,
            qa_decoder_attention_mask.shape,
            qa_decoder_attention_mask.device
        )

        qa_decoder_output = self.qa_decoder(query=qa_query,
                                            key=qa_hiddens,
                                            value=qa_hiddens,
                                            attention_mask=qa_matrix_attention_mask)[0]
        qa_pred_scores, qa_loss = self.compute_loss(qa_decoder_output, qa_decoder_labels)

        total_loss = (mlm_loss + lm_out.loss + qa_loss) / 3

        return (total_loss,)

    def compute_loss(self, hiddens, labels):
        pred_scores = self.encoder.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.encoder.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
            cls,
            model_args: ModelArguments,
            tokenizer: PreTrainedTokenizer,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, tokenizer, model_args)
        return model
