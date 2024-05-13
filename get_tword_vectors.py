# get the contexualized word vectors for a target word
# last maintained: 2024-05-12 14:51:17

import torch
cossim = torch.nn.CosineSimilarity(dim=0)

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
device = "cuda:0" if torch.cuda.is_available() else "cpu"

bert_model = 'bert-base-uncased'
#bert_model = 'bert-large-uncased'
#bert_model = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(bert_model)
pr_model = AutoModel.from_pretrained(bert_model, output_hidden_states=True)

#
sample = 'domesticity [SEP] N [SEP] Making a hobby of domesticity . [SEP] A royal family living in unpretentious domesticity . [SEP] In the first sentence, "domesticity" refers to home-related activities or chores being done for pleasure, while in the second sentence, it refers to a royal family leading a simple, home-centered life. Thus, "domesticity" can refer to both the enjoyment of home-related tasks and the simplicity of home life. [SEP]'
sample_ids = '4-6'

sample2 = 'circulate [SEP] V [SEP] Circulate a rumor . [SEP] This letter is being circulated among the faculty . [SEP] hogegegegege [SEP]'
sample2_ids = '0-4'

sample3 = 'dissipation [SEP] N [SEP] The dissipation of the mist . [SEP] Mindless dissipation of natural resources . [SEP] hogegege'
samples3_idx='1-1'

sample4 = "go [SEP] V [SEP] The washing machine wo n't go unless it 's plugged in . [SEP] The policemen went from door to door looking for the suspect . [SEP] higehige"
sample4_idx='5-2'

# for testing
import pandas as pd
import Levenshtein

train_df = pd.read_csv('./WiC_dataset/train/train.data.txt', delimiter='\t', names=['w', 'p', 'idxs', 'c1', 'c2'])

def test(id, verbose=False):
    inst = train_df.iloc[id-1]
    text = inst['w'] + ' [SEP] ' + inst['p'] + ' [SEP] ' + inst['c1'] + ' [SEP] ' + inst['c2'] + ' [SEP] '
    vectors, idxs = bert_encode_sent(text, pr_model)
    w1_idx, w2_idx = [int(_) for _ in inst['idxs'].split('-')]
    w1v, w2v, w1, w2 = get_tword_vectors_test(vectors, idxs, w1_idx, w2_idx, verbose=verbose)
    lv_sim = Levenshtein.ratio(w1, w2)
    if w1!=w2 and lv_sim < 0.5: 
        print(id, 'tword/pos:', inst['w'], inst['p'])
        print('ERROR?', w1, w2, lv_sim)
        print('Sim:', cossim(w1v, w2v))
        print()
    return w1v, w2v
                     
# batch version called from forward of the model (cl_conf1.py; cl_conf2.py)
def get_tword_vectors_(v, t, w1, w2):
    r_ = [get_tword_vectors(v_, t_, w1_, w2_, verbose=False) for v_, t_, w1_, w2_ in zip(v, t, w1, w2)]
    return [_[0] for _ in r_], [_[1] for _ in r_]

def get_tword_vectors(vectors, token_ids, w_idx_c1, w_idx_c2, verbose=False):
    # Input: vectors, idxs = ce.bert_encode_sent(ce.sample, ce.pr_model)
    c1_span, c2_span = get_csent_spans(token_ids)
    c1_vector = vectors[c1_span[0]:c1_span[-1]+1]
    c2_vector = vectors[c2_span[0]:c2_span[-1]+1]
    c1_idxs, c1_tokens_wo_subwords = decode_tokens_wo_subwords(token_ids[c1_span[0]:c1_span[-1]+1])
    c2_idxs, c2_tokens_wo_subwords = decode_tokens_wo_subwords(token_ids[c2_span[0]:c2_span[-1]+1])
    w1_vector = get_embedding_vector(c1_vector, c1_idxs[w_idx_c1])
    w2_vector = get_embedding_vector(c2_vector, c2_idxs[w_idx_c2])
    w1 = c1_tokens_wo_subwords[w_idx_c1]
    w2 = c2_tokens_wo_subwords[w_idx_c2]
    if verbose:
        print('C1 span:', c1_span, 'C2 span:', c2_span)
        print('C1:', c1_idxs, c1_tokens_wo_subwords)
        print('C2:', c2_idxs, c2_tokens_wo_subwords)
        print('w1:', c1_idxs[w_idx_c1], w1)
        print('w2:', c2_idxs[w_idx_c2], w2)    
    return w1_vector, w2_vector

def get_tword_vectors_test(vectors, token_ids, w_idx_c1, w_idx_c2, verbose=False):
    # Input: vectors, idxs = ce.bert_encode_sent(ce.sample, ce.pr_model)
    c1_span, c2_span = get_csent_spans(token_ids)
    c1_vector = vectors[c1_span[0]:c1_span[-1]+1]
    c2_vector = vectors[c2_span[0]:c2_span[-1]+1]
    c1_idxs, c1_tokens_wo_subwords = decode_tokens_wo_subwords(token_ids[c1_span[0]:c1_span[-1]+1])
    c2_idxs, c2_tokens_wo_subwords = decode_tokens_wo_subwords(token_ids[c2_span[0]:c2_span[-1]+1])
    w1_vector = get_embedding_vector(c1_vector, c1_idxs[w_idx_c1])
    w2_vector = get_embedding_vector(c2_vector, c2_idxs[w_idx_c2])
    w1 = c1_tokens_wo_subwords[w_idx_c1]
    w2 = c2_tokens_wo_subwords[w_idx_c2]
    if verbose:
        print('C1 span:', c1_span, 'C2 span:', c2_span)
        print('C1:', c1_idxs, c1_tokens_wo_subwords)
        print('C2:', c2_idxs, c2_tokens_wo_subwords)
        print('w1:', c1_idxs[w_idx_c1], w1)
        print('w2:', c2_idxs[w_idx_c2], w2)    
    return w1_vector, w2_vector, w1, w2

def get_embedding_vector(vectors, span):
    if type(span)==type(1):
        return vectors[span]
    else:
        return torch.mean(vectors[span[0]:span[-1]+1], axis=0)

def get_csent_spans(token_ids):
    c1_sep_start = 2; c1_sep_end = 3
    c2_sep_start = 3; c2_sep_end = 4
    c1_span = []
    c2_span = []
    sep_c = 0
    in_c1_span = False
    in_c2_span = False
    for i, id in enumerate(token_ids):
        if id==102:
            sep_c += 1
        if sep_c==c1_sep_start:
            in_c1_span = True
        if sep_c==c2_sep_start:
            in_c2_span = True
        if in_c1_span:
            c1_span.append(i)
        if in_c2_span:
            c2_span.append(i)
        if sep_c==c1_sep_end:
            in_c1_span = False
            continue
        if sep_c==c2_sep_end:
            break
    return c1_span[1:-1], c2_span[1:-1]

def decode_tokens_wo_subwords(idxs):
    # idxs: tensor([  101,   146, 22480,  1103, 12862,  5838,  1757,  2686,   119,   102])
    # ret-1 idx_list: [0, 1, 2, 3, [4, 5, 6], 7, 8, 9]
    # ret-2 wstr_list: ['[CLS]', 'I', 'verified', 'the', 'informativeness', 'results', '.', '[SEP]']
    #
    idx_list = []
    wstr_list = []
    # tstr_list: ['[CLS]', 'I', 'verified', 'the', 'inform', '##ative', '##ness', 'results', '.', '[SEP]']
    tstr_list = decode_tokens(idxs)
    #
    def seek_end(start):
        if start == len(tstr_list)-1:
            return start+1
        k = start
        while(True):
            if tstr_list[k+1].startswith('##'):
                k += 1
            if not tstr_list[k+1].startswith('##'):
                return k+1
        return 'poi' # OK?
    def recover_word(tokens):
        w_str = ''
        for t in tokens:
            if t.startswith('##'):
                w_str += t[2:]
            else:
                w_str += t
        return w_str
    #
    i = 0
    while(i < len(tstr_list)):
        end_index = seek_end(i)
        __id = list(range(i, end_index))
        _ids = idxs[i:end_index]
        _tks = [tokenizer.decode(x) for x in _ids]
        if len(__id) == 1:
            idx_list.append(__id[0])
        else:
            idx_list.append(__id)
        wstr_list.append(recover_word(_tks))
        i = end_index
    return idx_list, wstr_list

###
def bert_encode_sent(s, bert_model):
    tokens = tokenizer(s, return_tensors='pt').to(device)
    bert_model.to(device)
    with torch.no_grad():
        bert_output = bert_model(**tokens)
    # returns
    # - contextualized vectors (torch.tensor): shape=[#tokens, 768]
    # - token ids (torch.tensor): shape=[#tokens]
    return bert_output[0].cpu().squeeze(), tokens['input_ids'][0].cpu()

#######
def decode_tokens(idxs):
    # idxs: tensor([  101,   146, 22480,  1103, 12862,  5838,  1757,  2686,   119,   102])
    # returns: ['[CLS]', 'I', 'verified', 'the', 'inform', '##ative', '##ness', 'results', '.', '[SEP]']
    return [tokenizer.decode(idx) for idx in idxs.tolist()]

def get_embedding(s, idx):
    embs, token_ids_w_subwords = bert_encode_sent(s, pr_model)
    token_ids_wo_subwords, decoded_tokens = decode_tokens_wo_subwords(token_ids_w_subwords)
    if type(token_ids_wo_subwords[idx+1]) == type(1): # not subwords
        return embs[idx+1]
    else: # list of subwords ids
        return torch.mean(embs[token_ids_wo_subwords[idx+1]], axis=0)

def get_embedding_(idx, embs, token_ids_w_subwords):
    token_ids_wo_subwords, decoded_tokens = decode_tokens_wo_subwords(token_ids_w_subwords)
    if type(token_ids_wo_subwords[idx+1]) == type(1): # not subwords
        print('Target:', decoded_tokens[idx])
        return embs[idx+1]
    else: # list of subwords ids
        return torch.mean(embs[token_ids_wo_subwords[idx+1]], axis=0)

def compare_embeddings(s1, idx1, s2, idx2):
    v1 = get_embedding(s1, idx1)
    v2 = get_embedding(s2, idx2)
    return cossim(v1, v2), v1, v2
