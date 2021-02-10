# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def get_tokenized_tokens(words, tokenizer):
    all_tokens = []    
    tok_to_orig_index = []
    orig_to_tok_index = []

    for i, token in enumerate(words):
        orig_to_tok_index.append(len(all_tokens))            
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            all_tokens.append(sub_token)
            tok_to_orig_index.append(i)

    return all_tokens, tok_to_orig_index, orig_to_tok_index
