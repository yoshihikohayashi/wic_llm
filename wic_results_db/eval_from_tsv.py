# summarize the zero shot evaluation results from *.tsv file
# Usage example: $ python eval_from_tsv.py zs_gpt-4-0613_test_2.tsv 
# last maintained: 2024-05-12 10:20:52

import pandas as pd
def make_df(fname):
    df = pd.read_csv(fname, delimiter='\t')
    return df

from sklearn.metrics import confusion_matrix, classification_report
def make_results_summary(predicted_list, gold_list):
    print(confusion_matrix(gold_list, predicted_list))  
    print(classification_report(gold_list, predicted_list))
    #
    corr = 0
    for p, g in zip(predicted_list, gold_list):
        if p[:2] == g[:2]: corr += 1
    return corr/len(predicted_list)

def main(fname='./zs_gpt-3.5-turbo-0613_test_2.tsv'):
    df = make_df(fname)
    gold = df['l']
    pred = df['pred']
    accuracy = make_results_summary(pred, gold)
    print('Accuracy:', accuracy)
    return accuracy

#####
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()



