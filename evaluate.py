#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:16:34 2022

@author: premkumar
"""
import argparse
import logging
import os
import random
import glob

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
#from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME,
                          BertConfig, BertForQuestionAnswering, BertTokenizer,
                          XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer,
                          DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)

from utils_squad import (
    RawResultExtended, write_predictions_extended, RawResult, write_predictions)

from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad
from train import load_and_cache_examples

import json

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(opt, model, tokenizer, results_dir, prefix=""):
    
    dataset, examples, features = load_and_cache_examples(opt, tokenizer, validation=True, output_examples=True)
    eval_sampler = SequentialSampler(dataset)
    
    batch_size = 4
    
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)
    
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(opt.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1]
                      }
            if opt.model_type != 'distilbert':
                inputs['token_type_ids'] = None if opt.model_type == 'xlm' else batch[2]
            example_indices = batch[3]
            if opt.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            
            outputs = model(**inputs)
        
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if opt.model_type in ['xlnet', 'xlm']:
                result = RawResultExtended(unique_id=unique_id,
                                           start_top_log_probs=to_list(outputs[0][i]),
                                           start_top_index=to_list(outputs[1][i]),
                                           end_top_log_probs=to_list(outputs[2][i]),
                                           end_top_index=to_list(outputs[3][i]),
                                           cls_logits=to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
            
            all_results.append(result)
    
    output_prediction_file = os.path.join(results_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(results_dir, "nbest_predictions_{}.json".format(prefix))
    
    output_null_log_odds_file = None
    
    if opt.model_type in ['xlnet', 'xlm']:
        write_predictions_extended(examples, features, all_results, opt.n_best_size,
                        opt.max_answer_length, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, opt.predict_file,
                        model.config.start_n_top, model.config.end_n_top,
                        opt.version_2_with_negative, tokenizer, opt.verbose_logging)
    else:
        write_predictions(examples, features, all_results, opt.n_best_size,
                        opt.max_answer_length, opt.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, opt.verbose_logging,
                        opt.version_2_with_negative, opt.null_score_diff_threshold)
    
    evaluate_options = EVAL_OPTS(data_file=opt.predict_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    
    return results

class EvaluationModelParams:
    def __init__(self,
                 train_file='./data/train-v1.1.json',
                 predict_file='./data/dev-v1.1.json',
                 model_type='bert',
                 # output_dir='/home/bhaskaran.p/NLP_project',
                 output_dir='.',
                 max_seq_length=382,
                 doc_stride=128,
                 max_query_length=64,
                 gradient_accumulation_steps=1,
                 num_train_epochs=3,
                 weight_decay=0.0,
                 learning_rate=5e-5,
                 adam_epsilon=1e-8,
                 max_grad_norm=1.0,
                 alpha_ce=0.5,
                 alpha_squad=0.5,
                 temperature=2.0,
                 logging_steps=50,
                 save_steps=50,
                 version_2_with_negative=False,
                 n_best_size=20,
                 max_answer_length=30,
                 do_lower_case=True,
                 verbose_logging=False,
                 null_score_diff_threshold=0.0
                 ):
        self.train_file = train_file
        self.predict_file = predict_file
        self.model_type = model_type
        self.output_dir = output_dir
        self.doc_stride = doc_stride
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.null_score_diff_threshold = null_score_diff_threshold
        self.verbose_logging = verbose_logging
        self.do_lower_case = do_lower_case

        self.alpha_ce = alpha_ce
        self.alpha_squad = alpha_squad
        self.temperature = temperature
        self.save_steps = save_steps
        self.logging_steps = logging_steps

        self.version_2_with_negative = version_2_with_negative


MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)}

pretrained_models = {
    'distilbert': 'distilbert-base-uncased',
    'bert': "google/bert_uncased_L-6_H-512_A-8",
    'xlnet': 'xlnet-base-cased'}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Provide model as input: [bert, distilbert, xlnet]')
    # Required argument
    parser.add_argument('--model', type=str,
                        help='Model type should be: "bert", "distilbert" or "xlnet"')
    args = parser.parse_args()

    # Initialize evaluation model params.
    model_type = args.model
    opt = EvaluationModelParams(model_type=model_type)

    checkpoints_dir = ('./Final_Checkpoints_' + model_type)

    # Because DistilBERT is trained with BERT as the teacher.
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert' if opt.model_type == 'distilbert' else opt.model_type]
    pretrained = pretrained_models['bert' if opt.model_type == 'distilbert' else opt.model_type]
    
    tokenizer = tokenizer_class.from_pretrained(
        pretrained_model_name_or_path=pretrained, do_lower = opt.do_lower_case)

    # Output Directory results - [bert, distilbert, xlnet]
    model_spec = model_type
    results_dir = os.path.join(os.getcwd(), 'results', model_spec) # Add path here.
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results = {}
    print(checkpoints_dir)
    checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(checkpoints_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    for checkpoint in checkpoints:
        model = model_class.from_pretrained(checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
        result = evaluate(
            opt,
            model,
            tokenizer,
            prefix='',
            results_dir=results_dir)
        result = dict((k, v) for k, v in result.items())
        results.update(result)

    with open(os.path.join(results_dir, 'result.json'), 'w') as file:
        json.dump(results, file)
    