from mindnlp.transformers import BertModel
from mindnlp.core import nn 

class SentimentClassifier(nn.Module):
    def __init__(self, base_model_name_or_path = 'bert-base-uncased', freeze_bert = True):
        super().__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel.from_pretrained(base_model_name_or_path)

        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.cls_layer = nn.Linear(768, 1)
    
    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model
        last_hs = self.bert_layer(seq, attention_mask = attn_masks).last_hidden_state

        #Obtaining the representation of [CLS] head
        cls_rep = last_hs[:, 0]
        
        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)
        
        return logits
