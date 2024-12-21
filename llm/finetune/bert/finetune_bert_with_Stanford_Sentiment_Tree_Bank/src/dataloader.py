import mindspore
from mindnlp.transformers import BertTokenizer
import pandas as pd

class SSTDataset():
    def __init__(self, base_model_name_or_path, filename, maxlen):

        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter = '\t')

        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name_or_path)

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'label']

        #Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence) #Tokenize the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        
        return tokens_ids, label

def get_loader(dataset, batchsize, shuffle=True, num_workers=1, drop_remainder=True):
    data_loader = mindspore.dataset.GeneratorDataset(source=dataset,
                                      column_names=['tokens_ids', 'label'],
                                      shuffle=shuffle,
                                      num_parallel_workers=num_workers
                                      )
    data_loader = data_loader.batch(batch_size=batchsize, 
                                    drop_remainder=drop_remainder,
                                    )
    return data_loader.create_dict_iterator()