# supervised text classification: configuration#1
# last maintained: 2024-05-12 15:22:51
 
import torch
cossim = torch.nn.CosineSimilarity(dim=0)

import datasets
import transformers

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.optim import AdamW
import pandas as pd
import json
import tinydb

from datasets.utils.logging import disable_progress_bar
#disable_progress_bar()

import get_tword_vectors as gtv

#
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#
from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

#
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback
#bert_base = 'bert-large-uncased'
#bert_base = 'roberta-base'
#bert_base = 'roberta-large'
bert_base = 'bert-base-uncased'
#
bert_model_name = bert_base
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
#
from transformers import AutoConfig
from transformers import BertConfig
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from torch import nn
#
class WiC_with_Bert(BertPreTrainedModel):
    config_class = BertConfig
    #
    def __init__(self, config, wv_dim, dv_dim=512):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.wv_dim = wv_dim
        self.dv_dim = dv_dim
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, dv_dim),
        )
        if wv_dim==0:
            self.classifier = nn.Sequential(
                nn.Linear(dv_dim, 128), # 2023-10-04 20:28:00
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(128, config.num_labels), # 2023-10-04 20:41:47
            )
        elif wv_dim==-1:
            self.classifier = nn.Sequential(
                nn.Linear(dv_dim+config.hidden_size*2, 128), # 2023-10-04 20:28:00
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(128, config.num_labels), # 2023-10-04 20:41:47
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(dv_dim+self.wv_dim*2, 128), # 2023-10-04 20:28:00
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(128, config.num_labels), # 2023-10-04 20:41:47
            )
        if wv_dim!=0 and wv_dim!=-1: # use FF layeer; wv_dim==0: not using vectors; wv_dim=-1: not using FF
            self.feedforward = nn.Linear(config.hidden_size*2, self.wv_dim*2, config.num_labels)
        print('Classifier:', self.classifier)
        self.init_weights()
    #
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, w1_idx=None, w2_idx=None, **kwargs):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids, 
                                        **kwargs)
        dropout_output = self.dropout(self.linear(bert_outputs[1])) # 2023-10-04 20:23:46 just for probing
        if self.wv_dim==0: # do not use bert word vectors
            merged_output = dropout_output
        else: # use bert word vectors
            w1_vector, w2_vector = gtv.get_tword_vectors_(bert_outputs[0], input_ids, w1_idx, w2_idx)
            w1_vector = torch.stack(w1_vector, dim=0)
            w2_vector = torch.stack(w2_vector, dim=0)
            if self.wv_dim==-1: # use bert word vectors as is by concanenation (not using FF)
                merged_output = torch.concat([dropout_output, w1_vector, w2_vector], dim=1)
            else: # use bert word vectors with a FF layer
                mult_vector = w1_vector * w2_vector
                diff_vector = torch.abs(w1_vector - w2_vector)
                ff_vector = self.feedforward(torch.cat([mult_vector, diff_vector], dim=1))
                merged_output = torch.concat([dropout_output, ff_vector], dim=1)
        #
        logits = self.classifier(merged_output)
        if labels is not None:
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        #
        return SequenceClassifierOutput(loss=loss, logits=logits)

#
def evaluate(trainer_obj, test_ds, verbose=True):
    eval_obj = trainer_obj.predict(test_ds)
    golds = eval_obj.label_ids
    preds = [np.argmax(_) for _ in eval_obj.predictions]
    conf_matrix = confusion_matrix(golds, preds)      
    class_report = classification_report(golds, preds)
    accuracy = accuracy_score(golds, preds)
    if verbose:
        print(conf_matrix)
        print(class_report)
    print('Accuracy:', accuracy)
    return eval_obj, golds, preds, accuracy, conf_matrix, class_report

#
def train(ds, epochs, lr, batch_size, patience):
    global wic_with_bert_model
    #
    model = wic_with_bert_model
    optimizer = AdamW(model.parameters(), lr=lr)
    #
    train_ds = ds['train']
    val_ds = ds['validation']
    test_ds = ds['test']
    #
    batch_size = batch_size
    logging_steps = len(train_ds) // batch_size
    model_name = "cl_conf2"
    #
    training_args = TrainingArguments(
        output_dir=model_name,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_on_each_node=1,
        metric_for_best_model='accuracy',
        load_best_model_at_end=True,
        disable_tqdm=False,
        #disable_tqdm=True,
        logging_steps=logging_steps,
        push_to_hub=False,
        log_level='error',
        lr_scheduler_type='linear',
        warmup_ratio=0.1,
    )
    #
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    #
    trainer.train()
    #
    return trainer

#
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

###
import make_transformers_dataset as mtd
import argparse
import tinydb

###
def main(verbose=True, save_db=True):
    global wic_with_bert_model
    #
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument('--verb', type=str, default='contrast', help='contrast, direct, direct2, or none')
    arg_p.add_argument('--train_fname', type=str, default='./data_tsv/train.tsv', help='train data file')
    arg_p.add_argument('--dev_fname', type=str, default='./data_tsv/dev.tsv', help='dev data file')
    arg_p.add_argument('--test_fname', type=str, default='./data_tsv/test.tsv', help='test data file')
    arg_p.add_argument('--llm', type=str, default='gpt3', help='model name; gpt3 or gpt4')
    arg_p.add_argument('--seed', type=int, default=23, help='random seed')
    arg_p.add_argument('--batch_size', type=int, default=16, help='batch size')
    arg_p.add_argument('--epochs', type=int, default=20, help='number of maximum epochs')
    arg_p.add_argument('--lr', type=float, default=4e-5, help='learning rate')
    arg_p.add_argument('--patience', type=int, default=5, help='early stopping patience')
    arg_p.add_argument('--wv_dim', type=int, default=0, help='word vector dimensionality')
    arg_p.add_argument('--dv_dim', type=int, default=512, help='[CLS] vector dimensionality') # 2023-10-04 20:54:22
    arg_p.add_argument('--n_train', type=int, default=5428, help='number of training data instances') # first n instances
    arg_p.add_argument('--dev_idxs', type=list, default=[], help='list of instance id indices') 
    arg_p.add_argument('--test_idxs', type=list, default=[], help='list of instance id indices') 
    arg_p.add_argument('--save_db', type=str, default="True", help='save the evaluation results in db')
    arg_p.add_argument('--verbose', type=str, default="True", help='verbose')
    args = arg_p.parse_args()
    #
    verbose = args.verbose
    #
    if args.save_db == 'True' or args.save_db == 'true':
        save_db = True
    else:
        save_db = False
    #
    transformers.set_seed(args.seed)
    #
    wic_with_bert_config = AutoConfig.from_pretrained(bert_model_name, num_labels=2)
    wic_with_bert_model = (WiC_with_Bert.from_pretrained(bert_model_name, 
                                                         config=wic_with_bert_config, wv_dim=args.wv_dim, dv_dim=args.dv_dim).to(device))
    #    
    if verbose: print('Parameters:', args)
    #
    # pretrained contenxt-dependent embeddings for w1 and w2
    print('>>> In Preparation')
    #
    if verbose:
        print('train, dev, test files:', args.train_fname, args.dev_fname, args.test_fname)
    #
    ds = mtd.make_transformers_dataset(args.verb, args.llm, 
                                       args.train_fname, args.dev_fname, args.test_fname, 
                                       range(args.n_train), args.dev_idxs, args.test_idxs,
                                       )
    #
    if verbose: print(ds)

    #
    print('>>> In Training')
    trainer_obj = train(ds, 
                        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, patience=args.patience)
    print('>>> Evaluation')
    val_result = trainer_obj.evaluate()
    print('--- Validation results ---')
    print(val_result)
    test_result = trainer_obj.evaluate(ds['test']) 
    print('--- Test results ---')
    print(test_result)
    eval_obj, golds, preds, accuracy, confusion_matrix, classification_report = evaluate(trainer_obj, ds['test'], verbose)
    #
    if save_db:
        wic_db_path = './wic_results_db/'
        wic_cl_results_db = tinydb.TinyDB(wic_db_path + 'cl_config_2' + '.json') 
        rslt_table = wic_cl_results_db.table('_'.join([args.verb, args.llm, str(args.wv_dim), str(args.seed), str(args.n_train)]))
        save_results_in_db(rslt_table, args.verb, args.llm, args.seed, args.wv_dim, 
                           json.dumps(golds.tolist()), str(preds), 
                           accuracy, json.dumps(confusion_matrix.tolist()), classification_report, 
                           json.dumps(val_result), json.dumps(test_result))
    #
    print('Accuracy:', accuracy, flush=True)
    return trainer_obj, eval_obj, golds, preds

#
def save_results_in_db(rslt_table, desc, llm, seed, wv_dim,
                       golds, preds, accuracy, confusion_matrix, classification_report,
                       val_result, test_result):
    rslt_table.insert({'desc':desc, 'llm':llm, 'seed':seed, 'wv_dim':wv_dim,
                       'golds':golds, 'preds':preds, 
                       'accuracy':accuracy, 'confusion_matrix':confusion_matrix,
                       'classfication_report':classification_report,
                       'val_result':val_result, 'test_result':test_result})

#####
if __name__ == '__main__':
    main()
