from tokenizers import Tokenizer
import os

class MiniCPMTokenizer:
    def __init__(self, tokenizer_file: str):
        self.tokenizer_file = tokenizer_file
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.tokenizer.token_to_id(tok) for tok in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.tokenizer.id_to_token(i) for i in ids]

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

if __name__ == "__main__":
    tokenizer_path = "D:/个人/MiniCPM_Llama3/minicpm_assets/tokenizer.json"
    tokenizer = MiniCPMTokenizer(tokenizer_path)

    text = "你好 MiniCPM"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded = tokenizer.decode(ids)

    print("原文：", text)
    print("分词：", tokens)
    print("Token IDs：", ids)
    print("解码：", decoded)
