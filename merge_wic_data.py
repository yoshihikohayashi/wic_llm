# merge necessary data files  
# last maintained: 2024-05-12 13:46:59
# Usage example: $ python merge_wic_data.py data_tsv

import pandas as pd
import re
#
def remove_answer(df):
    r = []
    for x in list(df):
        r.append(re.sub('^ ', '', re.sub('Rationale: ', '', re.sub(r'Yes. |No. ', '', x))))
    rpd = pd.DataFrame(r)
    return rpd
#
WiC_dataset_path = './WiC_dataset/'
WiC_dataset_repaired_path = './WiC_dataset_repaired/'
descriptions_path = './descriptions/'

train_orig_data_df = pd.read_csv(WiC_dataset_path + 'train/train.data.txt', delimiter='\t', names=['w', 'p', 'idx', 'c1', 'c2'])
train_orig_gold_df = pd.read_csv(WiC_dataset_path + 'train/train.gold.txt', delimiter='\t', names=['l'])
train_new_data_df = pd.read_csv(WiC_dataset_repaired_path + 'train_new.data.txt', delimiter='\t')

dev_orig_data_df = pd.read_csv(WiC_dataset_path + 'dev/dev.data.txt', delimiter='\t', names=['w', 'p', 'idx', 'c1', 'c2'])
dev_orig_gold_df = pd.read_csv(WiC_dataset_path + 'dev/dev.gold.txt', delimiter='\t', names=['l'])
dev_new_data_df = pd.read_csv(WiC_dataset_repaired_path + 'dev_new.data.txt', delimiter='\t')

test_orig_data_df = pd.read_csv(WiC_dataset_path + 'test/test.data.txt', delimiter='\t', names=['w', 'p', 'idx', 'c1', 'c2'])
test_orig_gold_df = pd.read_csv(WiC_dataset_path + 'test/test.gold.txt', delimiter='\t', names=['l'])
test_new_data_df = pd.read_csv(WiC_dataset_repaired_path + 'test_new.data.txt', delimiter='\t')
#
train_contrast_gpt3_df = pd.read_csv(descriptions_path + 'wic_desc_train_contrast_gpt-3.5-turbo-0613.tsv', delimiter='\t')
train_direct_gpt3_df = pd.read_csv(descriptions_path + 'wic_desc_train_direct_gpt-3.5-turbo-0613.tsv', delimiter='\t')
train_contrast_gpt4_df = pd.read_csv(descriptions_path + 'wic_desc_train_contrast_gpt-4-0613.tsv', delimiter='\t')
train_direct_gpt4_df = pd.read_csv(descriptions_path + 'wic_desc_train_direct_gpt-4-0613.tsv', delimiter='\t')

dev_contrast_gpt3_df = pd.read_csv(descriptions_path + 'wic_desc_dev_contrast_gpt-3.5-turbo-0613.tsv', delimiter='\t')
dev_direct_gpt3_df = pd.read_csv(descriptions_path + 'wic_desc_dev_direct_gpt-3.5-turbo-0613.tsv', delimiter='\t')
dev_contrast_gpt4_df = pd.read_csv(descriptions_path + 'wic_desc_dev_contrast_gpt-4-0613.tsv', delimiter='\t')
dev_direct_gpt4_df = pd.read_csv(descriptions_path + 'wic_desc_dev_direct_gpt-4-0613.tsv', delimiter='\t')

test_contrast_gpt3_df = pd.read_csv(descriptions_path + 'wic_desc_test_contrast_gpt-3.5-turbo-0613.tsv', delimiter='\t')
test_direct_gpt3_df = pd.read_csv(descriptions_path + 'wic_desc_test_direct_gpt-3.5-turbo-0613.tsv', delimiter='\t')
test_contrast_gpt4_df = pd.read_csv(descriptions_path + 'wic_desc_test_contrast_gpt-4-0613.tsv', delimiter='\t')
test_direct_gpt4_df = pd.read_csv(descriptions_path + 'wic_desc_test_direct_gpt-4-0613.tsv', delimiter='\t')

#
train_merged_df = train_orig_data_df.copy()
dev_merged_df = dev_orig_data_df.copy()
test_merged_df = test_orig_data_df.copy()
#
train_merged_df['label'] = train_orig_gold_df['l']
dev_merged_df['label'] = dev_orig_gold_df['l']
test_merged_df['label'] = test_orig_gold_df['l'] 
# id column is missed in original files
train_merged_df['id'] = train_contrast_gpt4_df['id']
dev_merged_df['id'] = dev_contrast_gpt4_df['id']
test_merged_df['id'] = test_contrast_gpt4_df['id'] 

#
train_merged_df['new_idx'] = train_new_data_df['idxs']
train_merged_df['new_c1'] = train_new_data_df['c1']
train_merged_df['new_c2'] = train_new_data_df['c2']

dev_merged_df['new_idx'] = dev_new_data_df['idxs']
dev_merged_df['new_c1'] = dev_new_data_df['c1']
dev_merged_df['new_c2'] = dev_new_data_df['c2']

test_merged_df['new_idx'] = test_new_data_df['idxs']
test_merged_df['new_c1'] = test_new_data_df['c1']
test_merged_df['new_c2'] = test_new_data_df['c2']

# descriptions for contrast/direct ; gpt-3 ; gpt-4
train_merged_df['contrast_gpt3_resp'] = train_contrast_gpt3_df['resp']
train_merged_df['contrast_gpt4_resp'] = train_contrast_gpt4_df['resp']
train_merged_df['direct_gpt3_resp'] = train_direct_gpt3_df['resp']
train_merged_df['direct_gpt4_resp'] = train_direct_gpt4_df['resp']

dev_merged_df['contrast_gpt3_resp'] = dev_contrast_gpt3_df['resp']
dev_merged_df['contrast_gpt4_resp'] = dev_contrast_gpt4_df['resp']
dev_merged_df['direct_gpt3_resp'] = dev_direct_gpt3_df['resp']
dev_merged_df['direct_gpt4_resp'] = dev_direct_gpt4_df['resp']

test_merged_df['contrast_gpt3_resp'] = test_contrast_gpt3_df['resp']
test_merged_df['contrast_gpt4_resp'] = test_contrast_gpt4_df['resp']
test_merged_df['direct_gpt3_resp'] = test_direct_gpt3_df['resp']
test_merged_df['direct_gpt4_resp'] = test_direct_gpt4_df['resp']

# remoce Yes/No from the direct descriptions
# Remark: direcct2 descriptions are "direct-type" descriptions in the paper!!
train_merged_df['direct2_gpt3_resp'] = remove_answer(train_merged_df['direct_gpt3_resp'])
train_merged_df['direct2_gpt4_resp'] = remove_answer(train_merged_df['direct_gpt4_resp'])

dev_merged_df['direct2_gpt3_resp'] = remove_answer(dev_merged_df['direct_gpt3_resp'])
dev_merged_df['direct2_gpt4_resp'] = remove_answer(dev_merged_df['direct_gpt4_resp'])

test_merged_df['direct2_gpt3_resp'] = remove_answer(test_merged_df['direct_gpt3_resp'])
test_merged_df['direct2_gpt4_resp'] = remove_answer(test_merged_df['direct_gpt4_resp'])

#
to_file = True

#####
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_tsv_dir = sys.argv[1]
    else:
        data_tsv_dir = None
    #
    if data_tsv_dir:
        train_merged_df.to_csv('./{:s}/train.tsv'.format(data_tsv_dir), sep='\t')
        dev_merged_df.to_csv('./{:s}/dev.tsv'.format(data_tsv_dir), sep='\t')
        test_merged_df.to_csv('./{:s}/test.tsv'.format(data_tsv_dir), sep='\t')
