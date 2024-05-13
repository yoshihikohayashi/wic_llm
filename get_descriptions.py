# get descriptions for WiC data instances 
# last maintained: 2024-05-12 11:04:50
# You have to define your own OPENAI_API_KEY environment variable
# Usage example: $ python get_descriptions.py --pr_type direct --dataset train --llm gpt-4-0613

import pandas
import time, ray
import re
import openai

#
class prompt_tamplate:
    def __init__(self, name, template):
        self.name = name
        self.template_str = template

### prompt templates
contrast_template_str = '''
Contrast the core senses of \"{:s}\" in these two sentences and summarize the result in a short sentence.
1: {:s}
2: {:s}
'''

direct_template_str = '''
Identify if the target word "{:s}"  in the following sentences correspond to the identical meanings or not.
Answer Yes or No, and provide one brief sentence to describe the rationale behind the decision.
1: {:s}
2: {:s}
'''

contrast_template = prompt_tamplate('contrast', contrast_template_str)
direct_template = prompt_tamplate('direct', direct_template_str)

### top
def get_desc(pr_type, 
             dataset='test', s=0, e=10, 
             llm='gpt-3.5-turbo-0613',
             temperature=0, max_tokens=384, nc=8, save_tsv=True, verbose=True):
    #
    if pr_type == 'contrast':
        pr_template = contrast_template
    elif pr_type == 'direct':
        pr_template = direct_template
    else:
        pr_template = contrast_template
    #
    global _llm
    _llm = llm
    #
    if e==0:
        if dataset=='train':
            e = 5428
        elif dataset=='dev':
            e = 638
        else:
            e = 1400
    words, pos_list, c1_list, c2_list, golds = get_data(dataset, s, e)
    #
    desc_list = wic(words, pos_list, c1_list, c2_list, pr_template.template_str, temperature, max_tokens, nc, verbose)
    if save_tsv:
        fname = './descriptions/wic_desc_{ftype}_{prtype}_{model}.tsv'.format(ftype=dataset, prtype=pr_template.name, model=llm)
        save_to_tsv(fname, words, pos_list, c1_list, c2_list, golds, desc_list, s)
    return desc_list

#
def save_to_tsv(fname, words, pos_list, c1_list, c2_list, golds, desc_list, s):
    with open(fname, 'w', encoding="utf-8") as f:
        line = '\t'.join(['id', 'w', 'p', 'c1', 'c2', 'l', 'resp'])
        f.write(line+'\n')
        for i, (w, p, c1, c2, g, desc) in enumerate(zip(words, pos_list, c1_list, c2_list, golds, desc_list)):
            line = '\t'.join([str(s+i), w, p, c1, c2, g, desc])
            f.write(line+'\n')

#
def wic(words, pos_list, c1_list, c2_list, pr_template_str, temperature, max_tokens, nc, verbose):
    #
    ray.shutdown()
    ray.init(num_cpus=nc)
    #
    begin_t = time.time()
    desc_list = []
    cnt = 0
    for i, (word, pos, c1, c2) in enumerate(zip(words, pos_list, c1_list, c2_list)):
        for trial in range(10):
            wic_res = wic_(word, pos, c1, c2, pr_template_str, temperature, max_tokens, nc, verbose)
            if wic_res: 
                break
            else:
                print('>>> retrying wic_ >>>', wic_res, trial+1)
        wic_res_ = re.sub(r'\n+', ' ', wic_res)
        desc_list.append(wic_res_)
        print(i, 'finished:', cnt, wic_res)
        cnt += 1
    #
    end_t = time.time()
    if verbose:
        print('Elappsed (sec):', end_t-begin_t)
        print(desc_list)
    return desc_list

def wic_(word, pos, c1, c2, pr_template_str, temperature, max_tokens, nc, verbose, 
         to=30, trial_limit=2):
    time.sleep(1)
    begin_t = time.time()
    #
    query = pr_template_str.format(word, c1, c2)
    if verbose:
        print('c1:', c1)
        print('c2:', c2)
        print(query)
    #
    r_ = rgpt.remote(query, temperature, max_tokens)
    #
    cnt = 1
    _finished, _not_finished = ray.wait([r_], num_returns=1, timeout=to)
    while _not_finished and (cnt < trial_limit):
        cnt += 1
        print('>>> wic_: incomplete', cnt)
        print('    ... finished items:', len(_finished), 1)
        print('    ... elappled time:', time.time()-begin_t)        
        _finished, _not_finished = ray.wait([r_], num_returns=1, timeout=to)
    #
    if cnt >= trial_limit:
        print('<<< wic_ trial count exceedsd <<<')
        return []
        #pass
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
    if len(results)!=1:
        print('# of not finished calls:', 1-len(results))
    #
    end_t = time.time()
    print('Elappsed (sec):', end_t-begin_t)
    #
    response = [c["choices"][0]["message"]["content"] for c in results][0]
    if verbose: print('Response:', response)
    return response

def all_nots(l):
    for _ in l:
        if _: return False
    return True

### WiC dataset
def get_data(dataset, s, e):
    data_df_ = make_df(dataset=dataset)
    if e:
        data_df = data_df_.iloc[s:e]
    else:
        data_df = data_df_
    #
    words = list(data_df['word'])
    pos_list = list(data_df['pos'])
    c1_list = list(data_df['c1'])
    c2_list = list(data_df['c2'])
    golds = list(data_df['label'])
    #
    return words, pos_list, c1_list, c2_list, golds

def make_df(dataset='train'):
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'data.txt', encoding="utf-8") as f:
        data_df = pandas.read_csv(f, delimiter='\t', na_filter=None, names=['word', 'pos', 'index', 'c1', 'c2'])
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'gold.txt',  encoding="utf-8") as f:
        gold_df = pandas.read_csv(f, delimiter='\t', na_filter=None, header=None)
    data_df['label'] = gold_df
    return data_df

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

#########################
import argparse



def main():
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument('--pr_type', type=str, default='contrast', help='contrast or direct')
    arg_p.add_argument('--dataset', type=str, default='test', help='train, dev, or test')
    arg_p.add_argument('--s', type=int, default=0, help='start index')
    arg_p.add_argument('--e', type=int, default=0, help='end index')
    arg_p.add_argument('--llm', type=str, default='gpt-3.5-turbo-0613', help='gpt-3 or gpt-4')
    arg_p.add_argument('--save_tsv', type=str, default="True", help='save the detailed json file')
    arg_p.add_argument('--verbose', type=str, default="True", help='verbose')
    args = arg_p.parse_args()
    #
    desc_list = get_desc(pr_type=args.pr_type, dataset=args.dataset,
                         s=args.s, e=args.e, 
                         llm=args.llm, verbose=args.verbose)
    return desc_list

#####
if __name__ == '__main__':
    main()
