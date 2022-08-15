#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=gpu_run
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
"""
Created on Mon Aug  8 21:54:48 2022

@author: premkumar, sakthikripa, srikanth
"""

import logging
import os

import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm, trange

import argparse

from tensorboardX import SummaryWriter

from transformers import (
    BertConfig, BertForQuestionAnswering, BertTokenizer,
    XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer,
    DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)

from transformers import AdamW

from utils_squad import (
    read_squad_examples, convert_examples_to_features)

logger = logging.getLogger(__name__)

class TrainModelParams: # Has only init function, do we need this class then ?

    def __init__(
        self,
        train_file='/data/train-v1.1.json',
        predict_file='/data/dev-v1.1.json',
        model_type='xlnet',
        teacher_type = 'bert',
        output_dir='./',
        max_seq_length = 382,
        doc_stride = 128,
        max_query_length = 64,
        gradient_accumulation_steps = 1,
        num_train_epochs = 3,
        weight_decay = 0.0,
        learning_rate = 5e-5,
        adam_epsilon = 1e-8,
        max_grad_norm = 1.0,
        alpha_ce = 0.5,
        alpha_squad = 0.5,
        temperature = 2.0,
        logging_steps = 50,
        save_steps = 50):

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
        
        self.alpha_ce = alpha_ce
        self.alpha_squad = alpha_squad
        self.temperature = temperature
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        
        self.teacher_type = teacher_type

def to_list(tensor): # TODO: Not used
    return tensor.detach().cpu().tolist()

def load_and_cache_examples(opt, tokenizer, validation=False, output_examples=False):

    # Train or dev file path.
    input_file = opt.train_file if not validation else opt.predict_file
    
    cached_features_file = os.path.join(
        os.path.dirname(input_file), '{}_cached_{}'.format(
            opt.model_type, 'dev' if validation else 'train'))

    # Check for cached examples.
    if os.path.exists(cached_features_file):
        # Examples have been cached.
        print('Extracting cached examples.')
        features = torch.load(cached_features_file)

        # Fetch the squad dataset examples.
        examples = read_squad_examples(
            input_file=input_file,
            validation= validation)
        
    else:
        print('Extracting features')
        examples = read_squad_examples(
            input_file=input_file,
            validation=validation)
        
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=opt.max_seq_length,
                                                doc_stride=opt.doc_stride,
                                                max_query_length=opt.max_query_length,
                                                is_training=not validation)
        # Save the new calculated features as cached features.
        torch.save(features, cached_features_file)
    
    print('Features Extracted')

    # TODO: No idea what any of this does.
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_input_mask = torch.tensor([feature.input_mask for feature in features], dtype=torch.long)
    all_segment_ids = torch.tensor([feature.segment_ids for feature in features], dtype=torch.long)
    all_cls_index = torch.tensor([feature.cls_index for feature in features], dtype=torch.long)
    all_p_mask = torch.tensor([feature.p_mask for feature in features], dtype=torch.float)

    # TODO: No idea what's happening here.
    if validation:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)
    
    if output_examples:
        return dataset, examples, features
    
    return dataset

def train(opt, train_dataset, model, tokenizer, teacher=None):
    torch.cuda.empty_cache()

    
    summary_writer = SummaryWriter()
    
    batch_size = 8
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    
    t_total = len(train_dataloader) // opt.gradient_accumulation_steps * opt.num_train_epochs
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, eps=opt.adam_epsilon)
    
    

    model = torch.nn.DataParallel(model)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", opt.num_train_epochs)
    
    model_spec = '{}(s)-{}(t)'.format(opt.model_type, opt.teacher_type)
    checkpoint_dir = os.path.join(opt.output_dir, 'Checkpoints', model_spec)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(opt.num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if teacher is not None:
                teacher.eval()
            batch = tuple(t.to(opt.device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1], 
                      'start_positions': batch[3], 
                      'end_positions':   batch[4]}
            if opt.model_type != 'distilbert':
                inputs['token_type_ids'] = None if opt.model_type == 'xlm' else batch[2]
            if opt.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask':       batch[6]})
            outputs = model(**inputs)
            loss = outputs.loss
            start_logits_stu = outputs.start_logits
            end_logits_stu = outputs.end_logits

            if teacher is not None:
                if 'token_type_ids' not in inputs:
                    inputs['token_type_ids'] = None if opt.teacher_type == 'xlm' else batch[2]
                with torch.no_grad():
                    outputs = teacher(input_ids=inputs['input_ids'],
                                                               token_type_ids=inputs['token_type_ids'],
                                                               attention_mask=inputs['attention_mask'])
                
                start_logits_tea = outputs.start_logits
                end_logits_tea = outputs.end_logits
                
                loss_fct = nn.KLDivLoss(reduction='batchmean')
                loss_start = loss_fct(F.log_softmax(start_logits_stu/opt.temperature, dim=-1),
                                      F.softmax(start_logits_tea/opt.temperature, dim=-1)) * (opt.temperature**2)
                loss_end = loss_fct(F.log_softmax(end_logits_stu/opt.temperature, dim=-1),
                                    F.softmax(end_logits_tea/opt.temperature, dim=-1)) * (opt.temperature**2)
                loss_ce = (loss_start + loss_end)/2.
                loss = opt.alpha_ce*loss_ce + opt.alpha_squad*loss

            # Mean loss - for parallelism on GPUs.
            if opt.n_gpu > 1:
                loss = loss.mean() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            
            tr_loss += loss.item()
            if (step + 1) % opt.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1
            
            
            if global_step % opt.save_steps == 0:
                output_dir = os.path.join(checkpoint_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.module.save_pretrained(output_dir)
            
    
    output_dir = os.path.join(checkpoint_dir, 'Final_Checkpoints')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.module.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return global_step, tr_loss / global_step

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)}

# These are the names of the pretrained models we are using.
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

    opt = TrainModelParams(model_type=args.model)

    # Importing pretrained models trained on larger corpus.
    # Fetch the libraries based on the model we are using. [distilbert, bert, xlnet]
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opt.model_type]
    config = config_class.from_pretrained(pretrained_model_name_or_path=pretrained_models[opt.model_type])
    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name_or_path=pretrained_models[opt.model_type])
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=pretrained_models[opt.model_type],
        config=config)

    # Fetch the training dataset.
    train_dataset = load_and_cache_examples(
        opt,
        tokenizer,
        validation=False,
        output_examples=False)

    # Knowledge distillation with teacher BERT.
    pretrained_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    if opt.model_type == 'distilbert':
        teacher_config = BertConfig.from_pretrained(pretrained_model_name)
        teacher = BertForQuestionAnswering.from_pretrained(pretrained_model_name, config=teacher_config)

        teacher.to(opt.device)

    model.to(opt.device)

    logger.info('**** Starting training. ****')
    train(opt, train_dataset, model, tokenizer, teacher=None)
    logger.info('**** Finished training *****')

    
    
