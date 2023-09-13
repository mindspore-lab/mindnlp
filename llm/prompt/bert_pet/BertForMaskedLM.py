import math
from mindspore import nn, ops
from mindspore.common.initializer import initializer, HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindnlp.models import BertModel, BertLMPredictionHead

class BertOnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def construct(self, sequence_output):
        prediction_scores = self.predictions(sequence_output, None)
        return prediction_scores


class BertForMaskedLM(nn.Cell):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.embedding_table
        self.vocab_size = config.vocab_size
        self.reset_dense_parameters()
    
    def reset_dense_parameters(self):
        self.cls.predictions.decoder.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.cls.predictions.decoder.weight.shape))
        if self.cls.predictions.decoder.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.cls.predictions.decoder.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.cls.predictions.decoder.bias.set_data(initializer(Uniform(bound), [self.cls.predictions.decoder.out_channels]))

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                  masked_lm_labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[:2]
        if masked_lm_labels is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.vocab_size),
                                               masked_lm_labels.view(-1),
                                               ignore_index=-1)
            outputs = (masked_lm_loss,) + outputs

        return outputs