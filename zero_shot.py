# WiC zero shot baseline
# last maintained: 2024-05-12 11:04:39
# # You have to define your own OPENAI_API_KEY environment variable
# Usage example: $ python zero_shot.py --llm gpt-4-0613

import pandas
import time, ray
import openai
#
import tinydb
instance_q = tinydb.Query()

# zero-shot prompt template 2023-09-09 16:13:56
zero_shot_template = '''
Your task is to identify if the meanings of the target word \"{word}\" in the following c1 and c2 sentences correspond to {adj} meanings or not.
That is, it is the Word-in-Context task.

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
zero_shot_prompt = prompt_template('zero-shot-default', zero_shot_template)

###
def eval_wic_zs(adj='identical', 
                target_dataset='test', s=0, e=0,
                pr_template=zero_shot_prompt, 
                llm='gpt-3.5-turbo-0613',
                trial=0,
                save_db=True, temperature=0, max_tokens=384, nc=8, verbose=True):
    # globals
    global _adj, _target_dataset, _pr_template, _llm
    _adj=adj; _target_dataset=target_dataset; _pr_template=pr_template
    _llm=llm
    #
    wic_db_path = './wic_results_db/'
    _db_name_ = '_'.join([llm, target_dataset, str(trial)])
    wic_db_target = tinydb.TinyDB(wic_db_path + 'zs_' + _db_name_ + '.json') 
    _table_name_ = '_'.join([llm, pr_template.name, adj, str(trial)]) # added llm, trial
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
    if verbose: print('>>> Target datasets:', target_dataset)
    pred_list = wic(adj, target_df, s, e, 
                    pr_template.template_str, 
                    temperature, max_tokens, nc, verbose)
    accuracy = make_results_summary(pred_list, golds)
    #
    if verbose: print('Accuracy:', accuracy)
    if save_db: save_preds(pr_table, target_df, pred_list)
    #
    return pred_list, golds

def wic(adj, target_df, s, e,
        pr_template, temperature, max_tokens, nc, verbose):
    words = target_df['word']; pos_list = target_df['pos']
    c1_list = target_df['c1']; c2_list = target_df['c2']
    golds = target_df['label']
    #
    ray.shutdown()
    ray.init(num_cpus=nc)
    #
    okay = ng = 0
    if verbose: print('\nStart WiC with zero-shot setting >', time.ctime(), flush=True)
    begin_t = time.time()
    pred_list = []
    for i, (word, pos, c1, c2) in enumerate(zip(words, pos_list, c1_list, c2_list)):
        print('\n---------------', flush=True)
        for trial in range(5):
            wic_res = wic_(adj, s+i, word, pos, c1, c2, 
                           pr_template, target_df, 
                           temperature=temperature, max_tokens=max_tokens, nc=nc, verbose=verbose)
            if wic_res: 
                break
            else:
                print('>>> retrying wic_ >>>', trial+1, flush=True)
        pred_list.append(wic_res)
        print('finished: i, pred, gold:', s+i, wic_res, golds[s+i])
        if wic_res==golds[s+i]: okay += 1
        else: ng += 1
        print('Acc so far:', okay/(okay+ng), flush=True)
    #
    end_t = time.time()
    if verbose:
        print('>>> All finished! Total elapsed (sec):', end_t-begin_t)
        print(time.ctime())
        print('\n', flush=True)
    #
    return pred_list

#
def wic_(adj, target_id, word, pos, c1, c2, 
         pr_template_body, target_df, 
         temperature, max_tokens, nc, verbose,
         timeout=3, trial_limit=5):
    time.sleep(1)
    begin_t = time.time()
    #
    query = pr_template_body.format(adj=adj, word=word, c1=c1, c2=c2)
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
        print('    ... elapsed time:', time.time()-begin_t, flush=True) 
        _finished, _not_finished = ray.wait([r_], num_returns=1, timeout=timeout)
    #
    if cnt >= trial_limit:
        print('<<< wic_ trial count exceedsd <<<', flush=True)
        return []
    #
    try:
        results = ray.get(_finished)
    except openai.error.RateLimitError as err:
        print('<<< RateLimitError <<< sleep for: 10 seconds', flush=True)
        time.sleep(10)
        return []
    #
    if len(results)!=1:
        print('# of not finished calls:', 1-len(results), flush=True)
    #
    end_t = time.time()
    print('Elapsed (sec):', end_t-begin_t, flush=True)
    #
    pred_list = [c["choices"][0]["message"]["content"] for c in results]
    #
    if not pred_list:
        return []
    #
    return pred_list[0]

### WiC dataset
def make_df(dataset='train'):
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'data.txt', encoding="utf-8") as f:
        data_df = pandas.read_csv(f, delimiter='\t', na_filter=None, names=['word', 'pos', 'index', 'c1', 'c2'])
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'gold.txt',  encoding="utf-8") as f:
        gold_df = pandas.read_csv(f, delimiter='\t', na_filter=None, header=None)
    data_df['label'] = gold_df
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
            print(f"Waiting for {wait_time_seconds} secs.", flush=True)
            time.sleep(wait_time_seconds)

##
import argparse

def main():
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument('--adj', type=str, default='identical', help='adjective to specify the degree of semantic sameness')
    arg_p.add_argument('--target_dataset', type=str, default='test', help='train, dev, or test')
    arg_p.add_argument('--s', type=int, default=0, help='start index')
    arg_p.add_argument('--e', type=int, default=0, help='end index')
    arg_p.add_argument('--trial', type=int, default=0, help='trial number')
    arg_p.add_argument('--llm', type=str, default='gpt-3.5-turbo-0613', help='gpt-3 or gpt-4')
    arg_p.add_argument('--save_db', type=str, default="True", help='save the detailed json file')
    arg_p.add_argument('--verbose', type=str, default="True", help='verbose')
    args = arg_p.parse_args()
    #
    eval_wic_zs(adj=args.adj, target_dataset=args.target_dataset,
                s=args.s, e=args.e, trial=args.trial,
                pr_template=zero_shot_prompt, llm=args.llm, verbose=args.verbose)

#####
if __name__ == '__main__':
    main()
