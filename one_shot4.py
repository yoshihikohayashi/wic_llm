# WiC
import pandas
import time, ray
import openai
#
import os, itertools
os.environ['OPENAI_API_KEY'] = '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'
#
import tinydb
instance_q = tinydb.Query()

# one-shot prompt template 2023-09-05 14:32:30
one_shot_template = '''
Your task is to identify if the meanings of the target word \"{word}\" in the following c1 and c2 sentences correspond to {adj} meanings or not.
That is, it is the Word-in-Context task.

Consider the following example in decision.
[Example]
Target word: {word_}
c1: {c1_}
c2: {c2_}
Answer: {ans_}

Please simply answer T, if the meanings correspond to {adj} meanings.
Otherwise, simply answer F.
[Question]
Target word: {word}
c1: {c1}
c2: {c2}
Answer: 
'''

class prompt_template:
    def __init__(self, name, template):
        self.name = name
        self.template_str = template
one_shot_prompt = prompt_template('one-shot-default', one_shot_template)

###
def eval_wic_fs(adj='identical', 
                target_dataset='test', s=0, e=0,
                ref_dataset='dev',  
                pr_template=one_shot_prompt, 
                sel_TF='TF', sel_POS='same', 
                sel_Dsim=True, sel_Ssim=False,
                llm='gpt-3.5-turbo-0613',
                trial=0,
                save_db=True, temperature=0, max_tokens=384, nc=8, verbose=True):
    # globals
    global _adj, _target_dataset, _ref_dataset, _pr_template
    global _sel_TF, _sel_POS, _sel_Dsim, _sel_Ssim, _llm
    _adj=adj; _target_dataset=target_dataset; _ref_dataset=ref_dataset; _pr_template=pr_template
    _sel_TF=sel_TF; _sel_POS=sel_POS; _sel_Dsim=sel_Dsim; _sel_Ssim=sel_Ssim
    _llm=llm
    #
    #wic_db_path = '/home/hayashi/work/wic/TinyDB/'
    wic_db_path = 'c:/Users/yoshi/python/wic/TinyDB/'
    _db_name_ = '_'.join([llm, target_dataset, ref_dataset, str(trial)])
    wic_db_target = tinydb.TinyDB(wic_db_path + 'wic_ospr_' + _db_name_ + '.json') 
    _table_name_ = '_'.join([llm, pr_template.name, adj, sel_POS, sel_TF, str(sel_Dsim), str(sel_Ssim), str(trial)])
    pr_table = wic_db_target.table(_table_name_)
    #
    if e==0: 
        target_df = make_df(target_dataset)[s:]
        e = len(target_df)
    else:
        target_df = make_df(target_dataset)[s:e]
    target_df['id'] = range(s, e)
    golds = target_df['label']
    #
    ref_df = make_df(ref_dataset)
    ref_df['id'] = range(len(ref_df))
    #
    if verbose: print('>>> Target, Reference datasets:', target_dataset, ref_dataset)
    pred_list = wic(adj, target_df, s, e, ref_df,
                    pr_template.template_str, 
                    temperature, max_tokens, nc, verbose)
    accuracy = make_results_summary(pred_list, golds)
    #
    if verbose: print('Accuracy:', accuracy)
    if save_db: save_preds(pr_table, target_df, pred_list)
    #
    return pred_list, golds

def wic(adj, target_df, s, e, ref_df, 
        pr_template, temperature, max_tokens, nc, verbose):
    words = target_df['word']; pos_list = target_df['pos']
    c1_list = target_df['c1']; c2_list = target_df['c2']
    golds = target_df['label']
    #
    ray.shutdown()
    ray.init(num_cpus=nc)
    #
    okay = ng = 0
    if verbose: print('\nStart WiC with one-shot examples >', time.ctime())
    begin_t = time.time()
    pred_list = []
    for i, (word, pos, c1, c2) in enumerate(zip(words, pos_list, c1_list, c2_list)):
        print('\n---------------')
        for trial in range(5):
            wic_res = wic_(adj, s+i, word, pos, c1, c2, 
                           pr_template, target_df, ref_df,
                           temperature=temperature, max_tokens=max_tokens, nc=nc, verbose=verbose)
            if wic_res: 
                break
            else:
                print('>>> retrying wic_ >>>', trial+1)
        pred_list.append(wic_res)
        print('finished: i, pred, gold:', s+i, wic_res, golds[s+i])
        if wic_res==golds[s+i]: okay += 1
        else: ng += 1
        print('Acc so far:', okay/(okay+ng))
    #
    end_t = time.time()
    if verbose:
        print('>>> All finished! Total elapsed (sec):', end_t-begin_t)
        print(time.ctime())
        print('\n')
    #
    return pred_list

#
def wic_(adj, target_id, word, pos, c1, c2, 
         pr_template_body, target_df, ref_df, 
         temperature, max_tokens, nc, verbose,
         timeout=3, trial_limit=5):
    time.sleep(1)
    begin_t = time.time()
    #
    if verbose: print('Selecting examples>')
    sel_word, sel_pos, sel_c1, sel_c2, sel_ans =\
        select_example(target_df, target_id, ref_df, word, c1, c2)
    #
    query = pr_template_body.format(adj=adj, word=word, c1=c1, c2=c2, 
                                    word_=sel_word, c1_=sel_c1, c2_=sel_c2, ans_=sel_ans)
    if verbose: print(query)
    #
    r_ = rgpt.remote(query, temperature, max_tokens)
    #
    cnt = 1
    _finished, _not_finished = ray.wait([r_], num_returns=1, timeout=timeout)
    while _not_finished and (cnt < trial_limit):
        cnt += 1
        print('>>> wic_: incomplete', cnt)
        print('    ... finished items:', len(_finished), 1)
        print('    ... elapsed time:', time.time()-begin_t)        
        _finished, _not_finished = ray.wait([r_], num_returns=1, timeout=timeout)
    #
    if cnt >= trial_limit:
        print('<<< wic_ trial count exceedsd <<<')
        return []
    #
    try:
        results = ray.get(_finished)
    except openai.error.RateLimitError as err:
        print('<<< RateLimitError <<< sleep for: 10 seconds')
        time.sleep(10)
        return []
    except openai.error.APIError as err:
        print('<<< APIError <<< sleep for: 10 seconds')
        time.sleep(10)
        return []
    #
    end_t = time.time()
    print('Elapsed (sec):', end_t-begin_t)
    #
    pred_list = [c["choices"][0]["message"]["content"] for c in results]
    #
    if not pred_list:
        return []
    #
    return pred_list[0]

#
import numpy as np
import random

def select_example_with_sim(cands, target_sim, T_or_F):
    cand_sims = list(cands['sim'])
    cand_diff_sims = [abs(target_sim - cand_sim) for cand_sim in cand_sims]
    if T_or_F=='T':
        sel_pos = np.argmax(cand_diff_sims)
    else:
        sel_pos = np.argmin(cand_diff_sims)
    return sel_pos

def select_example_random(cands, target_sim, T_or_F):
    return random.choice(range(len(cands)))

#
def select_example(target_df, target_id, ref_df, word, c1, c2, select_func=select_example_with_sim):
    #print('### Ref size:', len(ref_df))
    #print(target_df)
    t = target_df[target_df['id']==target_id]
    target_sim = t['sim']
    target_pos = t['pos']    
    #
    if _sel_POS=='N': POS = 'N'
    elif _sel_POS=='V': POS = 'V'
    elif _sel_POS=='same': 
        if target_pos.all()=='N': POS = 'N'
        else: POS = 'V'
    else:
        if target_pos.all()=='N': POS = 'V'
        else: POS = 'V'
    #print('>>>>> TF, POS', _sel_TF, POS, flush=True)
    #cands = ref_df[(ref_df['label']==_sel_TF) & (ref_df['pos']==_sel_POS)]
    if _sel_TF=='TF':
        T_or_F = random.choice(['T', 'F'])
        #cands = ref_df[(ref_df['pos']==POS)]
        cands = ref_df[(ref_df['label']==T_or_F) & (ref_df['pos']==POS)]
    else:
        cands = ref_df[(ref_df['label']==_sel_TF) & (ref_df['pos']==POS)]
        T_or_F = _sel_TF
    print('# of candidates:', len(cands), 'from:', len(ref_df), 'records', flush=True)
    #
    if _sel_Dsim:
        select_func=select_example_with_sim
        print('> selecting one example with similarity difference')
    else:
        select_func=select_example_random
        print('> selecting one example randomly')
    sel_pos = select_func(cands, target_sim, T_or_F)
    sel = cands.iloc[sel_pos]
    #
    print('> selected id:', sel['id'])
    print('> selected word POS:', sel['word'], POS)
    #
    return sel['word'], sel['pos'], sel['c1'], sel['c2'], sel['label']

### WiC dataset
def make_df(dataset='train'):
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'data.txt', encoding="utf-8") as f:
        data_df = pandas.read_csv(f, delimiter='\t', na_filter=None, names=['word', 'pos', 'index', 'c1', 'c2'])
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'gold.txt',  encoding="utf-8") as f:
        gold_df = pandas.read_csv(f, delimiter='\t', na_filter=None, header=None)
    with open('./add_sim_' + dataset + '.tsv', encoding="utf-8") as f:
        asim_df = pandas.read_csv(f, delimiter='\t', na_filter=None, names=['id', 'sim', 'c1v', 'c2v'])
    data_df['label'] = gold_df
    data_df['sim'] = asim_df['sim']
    return data_df

### results summary
from sklearn.metrics import confusion_matrix, classification_report
def make_results_summary(predicted_list, gold_list):
    print(confusion_matrix(gold_list, predicted_list))  
    print(classification_report(gold_list, predicted_list))
    #
    corr = 0
    for p, g in zip(predicted_list, gold_list):
        if p[:2] == g[:2]: corr += 1
    return corr/len(predicted_list)

### save prediction results
def save_preds(pr_table, target_df, preds):
    words = target_df['word']; pos_list = target_df['pos']
    c1_list = target_df['c1']; c2_list = target_df['c2']
    golds = target_df['label']; ids = target_df['id']
    #
    for id, word, pos, gold, c1, c2, pred in zip(ids, words, pos_list, golds, c1_list, c2_list, preds):
        pr_table.upsert({'id':id, 'w':word, 'p':pos, 'l':gold, 'c1':c1, 'c2':c2, 'pred':pred}, 
                        ((instance_q.id==id) & (instance_q.w==word) & (instance_q.p==pos)))

### OpenAI API with ray support
@ray.remote
def rgpt(query, temperature, max_tokens, trial_limit=5, first_wait_time=10):
    global _llm
    for i in range(trial_limit):
        try:
            api_res = openai.ChatCompletion.create(
                #model = 'gpt-3.5-turbo-0613',
                model = _llm,
                messages = [{'role':'user', 'content':query}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return api_res
        except openai.error.ServiceUnavailableError as err:
            if i==trial_limit - 1:
                raise
            print(f"Error: {err}")
            wait_time_seconds = first_wait_time * (2**i)
            print(f"Waiting for {wait_time_seconds} secs.")
            time.sleep(wait_time_seconds)
