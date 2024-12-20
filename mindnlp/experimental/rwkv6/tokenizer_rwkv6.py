# coding=utf-8
# Copyright 2024 BlinkDL, et al.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications copyright 2024 [Huawei Technologies Co., Ltd]
# Changes: Migrated to MindSpore interface
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import urllib
import tempfile
import urllib


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret


class RWKV_TOKENIZER():
    def __init__(self, file_name=None):
        vocab = file_name if file_name else self.get_vocab()

        self.idx2token = {}
        sorted = [] # must be already sorted
        with open(vocab, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def get_vocab(self):
        VOCAB_NAME = "rwkv_vocab_v20230424.txt"
        VOCAB_SRC = [
            "https://www.modelscope.cn/models/EliwiiKeeya/RWKV-x060-World-1B6-v2.1-20240328-ctx4096/resolve/master/rwkv_vocab_v20230424.txt"
        ]
        temp_dir = tempfile.gettempdir()
        temp_vocab_path = os.path.join(temp_dir, "mindnlp", "rwkv6")
        temp_vocab = os.path.join(temp_vocab_path, VOCAB_NAME)

        if os.path.exists(temp_vocab) and os.path.getsize(temp_vocab) > 0:
            print("Use cached vocab: " + temp_vocab)
            return temp_vocab
        else:
            print("Download the vocab as: " + temp_vocab)
            if not os.path.exists(temp_vocab_path):
                os.makedirs(temp_vocab_path)

            for url in VOCAB_SRC:
                try:
                    urllib.request.urlretrieve(url, temp_vocab)
                except Exception as e:
                    print(e)
                else:
                    return temp_vocab
            else:
                raise(RuntimeError("Download failed."))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))            
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        if isinstance(src, str):
            return [self.encodeBytes(src.encode("utf-8"))]
        elif isinstance(src, list):
            return [self.encodeBytes(s.encode("utf-8")) for s in src]

    def decode(self, tokens):
        return [self.decodeBytes(batch).decode('utf-8') for batch in tokens]    
        # try:
        #     return self.decodeBytes(tokens).decode('utf-8')
        # except:
        #     return '\ufffd' # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()