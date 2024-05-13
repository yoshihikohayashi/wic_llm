# Anayzing the LLM-generated descriptions using spaCy for tokenization
# last maintained: 2024-05-12 11:06:16

import pandas as pd
import statistics

import spacy
nlp = spacy.load('en_core_web_md')

def basic_stats_all():
    for dn in ['contrast_gpt3_resp', 'contrast_gpt4_resp',
               'direct_gpt3_resp', 'direct_gpt4_resp',
               'direct2_gpt3_resp', 'direct2_gpt4_resp',]:
        print('-----', dn)
        for fn in ['./data_tsv/train.tsv', './data_tsv/dev.tsv', './data_tsv/test.tsv']:
            print('---', fn)
            X = basic_stats(nlp_descs(fn, dn))
        print()
    
def basic_stats(docs):
    n_docs = len(docs)
    mean_doc_length = statistics.mean([len(doc.text) for doc in docs])
    stdev_doc_length = statistics.stdev([len(doc.text) for doc in docs])
    mean_n_sents = statistics.mean([len(list(doc.sents)) for doc in docs])
    stdev_n_sents = statistics.stdev([len(list(doc.sents)) for doc in docs])
    mean_n_tokens = statistics.mean([len(doc) for doc in docs])
    stdev_n_tokens = statistics.stdev([len(doc) for doc in docs])
    ##
    print('Description length:', mean_doc_length, stdev_doc_length)
    print('Number of sentences:', mean_n_sents, stdev_n_sents)
    print('Number of tokens:', mean_n_tokens, stdev_n_tokens)
    
def nlp_descs(fname, descname, label=None, pos=None):
    docs = get_descs(fname, descname, label, pos)
    return [nlp(_) for _ in docs]

def get_descs(fname, descname, label, pos):
    df = get_desc_df(fname, descname, label, pos)
    return list(df[descname])

def get_desc_df(fname, descname, label=None, pos=None):
    df = pd.read_csv(fname, delimiter='\t', na_filter=None)
    if label and (label=='T' or label=='F'):
        df = df[df['l']==label]
    if pos and (pos=='N' or pos=='V'):
        df = df[df['p']==pos]
    return pd.DataFrame(df[descname])

#####
if __name__ == '__main__':
    basic_stats_all()
