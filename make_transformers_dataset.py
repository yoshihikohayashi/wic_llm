# make transformers datasets
# called from: cl_conf1.py; cl_conf2.py
# last maintained: 2024-05-12 14:53:58

import re
import pandas as pd
import torch
import datasets
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer

#
disable_progress_bar()

bert_model_name = 'bert-base-uncased'
#bert_model_name = 'bert-large-uncased'
#bert_model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#
def make_transformers_dataset(verb, llm, 
                              train_fname, dev_fname, test_fname,
                              train_idxs, dev_idxs, test_idxs,
                              ):
    print('>>>>', train_fname, dev_fname, test_fname)
    train_ds = create_dataset(verb, llm, train_fname, train_idxs)
    dev_ds = create_dataset(verb, llm, dev_fname, dev_idxs)
    test_ds = create_dataset(verb, llm, test_fname, test_idxs)
    if (not train_ds) or (not dev_ds) or (not test_ds): 
        print('### Something wrong in make_transformers dataset')
        return []
    return datasets.DatasetDict({'train':train_ds, 'validation':dev_ds, 'test':test_ds})
    
def create_dataset(verb, llm, fname, idxs):
    # recover target word indexes (quick hack; index info shuold have been included in wic_fname file!!)
    wic_df_ = pd.read_csv(fname, delimiter='\t', na_filter=None)
    print('fname, idxs:', fname, idxs)
    if not idxs: idxs = range(len(wic_df_))
    print('len:', len(wic_df_))
    wic_df = wic_df_
    # set text and label
    wic_df['text'] = format_descriptions(wic_df, verb, llm)
    w1_ids = [int(_.split('-')[0]) for _ in list(wic_df['new_idx'])]
    w2_ids = [int(_.split('-')[1]) for _ in list(wic_df['new_idx'])]
    wic_df['w1_idx'] = w1_ids
    wic_df['w2_idx'] = w2_ids
    #
    ds = datasets.Dataset.from_pandas(wic_df[['text', 'label', 'w1_idx', 'w2_idx']])
    ds = ds.class_encode_column('label')
    tokenized_ds = ds.map(tokenize_ds, batched=True, batch_size=None)
    return tokenized_ds

def format_descriptions(df, verb, llm):
    texts = []
    if verb=='none':
        for w, p, c1, c2 in zip(df['w'], df['p'], df['new_c1'], df['new_c2']):
            text = w + ' [SEP] ' + p + ' [SEP] ' + c1 + ' [SEP] ' + c2
            texts.append(text)
    else:
        desc_name = '_'.join([verb, llm, 'resp'])
        for w, p, c1, c2, resp in zip(df['w'], df['p'], df['new_c1'], df['new_c2'], df[desc_name]):
            text = w + ' [SEP] ' + p + ' [SEP] ' + c1 + ' [SEP] ' + c2 + ' [SEP] ' + resp
            texts.append(text)
    #
    return texts

def tokenize_ds(ds):
    return tokenizer(ds['text'], max_length=384, padding=True, truncation=True)
