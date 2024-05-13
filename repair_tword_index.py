# repair the tokenization of contextual sentences in WiC instance file
# last maintained: 2024-05-12 13:36:43
# Usage example: $ python repair_tword_index.py train

import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()
 
def repair_tword_index(dataset='train', outfname=None):
    infname = './WiC_dataset/{dataset}/{dataset}.data.txt'.format(dataset=dataset)
    if not outfname:
        outfname = './WiC_dataset_repaired/{dataset}/{dataset}_new.data.txt'.format(infset=dataset)
    #
    df = pd.read_csv(infname, delimiter='\t', names=['w', 'p', 'idxs', 'c1', 'c2'], na_filter=None)
    with open(outfname, 'w') as outf:
        outf.write('\t'.join(['w', 'p', 'idxs', 'c1', 'c2'])+'\n')
        for inst in df.iloc:
            w1_idx, w2_idx = [int(_) for _ in inst['idxs'].split('-')]
            new_w1_idx, new_c1 = repair_tword_index_(w1_idx, inst['c1'])
            new_w2_idx, new_c2 = repair_tword_index_(w2_idx, inst['c2'])
            new_idx = '-'.join([str(new_w1_idx), str(new_w2_idx)])
            outfline = '\t'.join([inst['w'], inst['p'], new_idx, new_c1, new_c2])
            outf.write(outfline+'\n')
    #

import re
def repair_tword_index_(orig_index, orig_sent):
    orig_tokens = orig_sent.split()
    tw_form = orig_tokens[orig_index]
    #print('tw_form:', tw_form)
    repaired_sentence = detokenizer.detokenize(orig_tokens)
    #print('repaired sentence:', repaired_sentence)
    repaired_tokens = re.findall(r'\w+|[^\w\s]', repaired_sentence)
    #print('repaired tokens:', repaired_tokens)
    try:
        new_index = repaired_tokens.index(tw_form)
    except ValueError:
        print('Whao!')
        print('tw_form:', tw_form)
        print('repaired sentence:', repaired_sentence)
        print('repaired tokens:', repaired_tokens)
        new_index = seek_index(repaired_tokens, tw_form)
        if new_index==-1:
            new_index = seek_index2(repaired_tokens, tw_form)
        print('new index:', new_index)        
    #return new_index, repaired_sentence
    return new_index, ' '.join(repaired_tokens)

def seek_index(tokens, tw_form):
    for i, token in enumerate(tokens):
        if token.startswith(tw_form):
            return i
    return -1

import Levenshtein
import numpy as np
def seek_index2(tokens, tw_form):
    print('last resort using Levenshtein:', tokens, tw_form)
    sims = [Levenshtein.ratio(t, tw_form) for t in tokens]
    return np.argmax(sims)

#####
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        outfname = None
    else:
        outfname = sys.argv[2]
    if len(sys.argv) < 2:
        dataset = 'train'
    else:
        dataset = sys.argv[1]
    repair_tword_index(dataset=dataset, outfname=outfname)
