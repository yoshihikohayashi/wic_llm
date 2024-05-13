# additional analysis tools for the lrec-coling2024 paper
# ... reading the detailed results json (tinydb) file
# last maintained: 2024-05-12 14:56:44

import tinydb

#
def accuracies(json_file='./wic_results_db/cl_config_1.json', desc='contrast', llm='gpt3', 
               seeds = [23, 123, 223, 323, 3407], wv_dim=128, nrec=5428):
    db = tinydb.TinyDB(json_file)
    for i, seed in enumerate(seeds):
        dict = get_table(json_file, desc, llm, wv_dim, seed, nrec)
        acc = dict['accuracy']
        seed = dict['seed']
        print(seed, acc)
    return True
    
def get_table(json_file='./wic_results_db/cl_config_1.json', desc='contrast', llm='gpt3', wv_dim=128, seed=23, nrec=5428):
    db = tinydb.TinyDB(json_file)
    tbl_name = '_'.join([desc, llm, str(wv_dim), str(seed), str(nrec)])
    return db.table(tbl_name).all()[0]

def best_acc_seed(json_file='./wic_results_db/cl_config_1.json', desc='contrast', llm='gpt3', wv_dim=128, 
                  seeds = [23, 123, 223, 323, 3407], nrec=5428):
    best_acc = 0.0
    best_seed=seeds[0]
    for i, seed in enumerate(seeds):
        dict = get_table(json_file, desc, llm, wv_dim, seed, nrec)
        acc = dict['accuracy']
        if acc > best_acc:
            best_seed = dict['seed']
            best_acc = acc
            return_dict = dict
    return best_seed, return_dict

from sklearn.metrics import classification_report, confusion_matrix
def metrics(golds, preds):
    print (classification_report(golds, preds))
    print (confusion_matrix(golds, preds))

def ok_ng(golds, preds):
    oks = []
    ngs = []
    for i, (g, p) in enumerate(zip(golds, preds)):
        if g==p:
            oks.append(i)
        else:
            ngs.append(i)
    return oks, ngs

import itertools
def comparisons(golds, names_list, preds_list):
    names_combs = itertools.combinations(names_list, 2)
    preds_combs = list(itertools.combinations(preds_list, 2))
    both_ng_list = []
    print()
    for i, (x_name, y_name) in enumerate(names_combs):
        xx, yy = preds_combs[i]
        #xx = globals()[x]; yy = globals()[y]
        #xx = eval(x, globals()); yy = eval(y, globals())
        print(x_name, len(xx), y_name, len(yy))
        ok_x, ng_x = ok_ng(golds, xx)
        ok_y, ng_y = ok_ng(golds, yy)
        both_ok = set(ok_x).intersection(set(ok_y))
        only_x = set(ok_x).difference(set(ok_y))
        only_y = set(ok_y).difference(set(ok_x))
        both_ng = set(ng_x).intersection(set(ng_y))
        print('both ok', len(both_ok), '/', 'only', x_name, len(only_x), '/', 'only', y_name, len(only_y), '/', 'both ng', len(both_ng))
        and_acc = len(both_ok)/1400
        or_acc = (1400-len(both_ng))/1400
        print('And acc:', and_acc, 'Or acc:', or_acc)
        both_ng_list.append(both_ng)
        print()
    return both_ng_list

def comparisons2(golds, names_list, preds_list):
    names_combs = itertools.combinations(names_list, 2)
    preds_combs = list(itertools.combinations(preds_list, 2))
    both_ok_list = []
    both_ng_list = []
    only_x_list = []
    only_y_list = []
    print()
    for i, (x_name, y_name) in enumerate(names_combs):
        xx, yy = preds_combs[i]
        #xx = globals()[x]; yy = globals()[y]
        #xx = eval(x, globals()); yy = eval(y, globals())
        print(x_name, len(xx), y_name, len(yy))
        ok_x, ng_x = ok_ng(golds, xx)
        ok_y, ng_y = ok_ng(golds, yy)
        both_ok = set(ok_x).intersection(set(ok_y))
        only_x = set(ok_x).difference(set(ok_y))
        only_y = set(ok_y).difference(set(ok_x))
        both_ng = set(ng_x).intersection(set(ng_y))
        print('both ok', len(both_ok), '/', 'only', x_name, len(only_x), '/', 'only', y_name, len(only_y), '/', 'both ng', len(both_ng))
        and_acc = len(both_ok)/1400
        or_acc = (1400-len(both_ng))/1400
        print('And acc:', and_acc, 'Or acc:', or_acc)
        both_ok_list.append(both_ok)
        both_ng_list.append(both_ng)
        only_x_list.append(only_x)
        only_y_list.append(only_y)
        print()
    return both_ok_list, both_ng_list, only_x_list, only_y_list
